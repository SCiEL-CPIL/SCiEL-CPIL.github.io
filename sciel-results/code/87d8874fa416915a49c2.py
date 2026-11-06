import time
import random
import sys

N = 9
SIZE = 1 << N
BIT_MASKS = [1 << b for b in range(N)]

# Precompute neighbor masks as bitsets (integers)
# NEIGHBOR_MASKS[v] has bit i set if v and i are adjacent
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        neighbor = v ^ (1 << b)
        m |= (1 << neighbor)
    NEIGHBOR_MASKS[v] = m

def _bit_count(x):
    return bin(x).count('1')

def _get_bits(x):
    """Yield the indices of set bits in x."""
    while x:
        lsb = x & (-x)
        yield lsb.bit_length() - 1
        x ^= lsb

def _beam_search(start, beam_width, max_time, rng, lookahead_depth=2):
    """
    Beam search for the longest snake.
    State: (path_list, used_mask, forbidden_mask)
    used_mask: bitset of all vertices in the path.
    forbidden_mask: bitset of all vertices that are adjacent to any vertex in path[:-1].
    """
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0
    
    # Initial beam: one state
    beam = [(path, used_mask, forbidden_mask)]
    best_path = list(path)
    best_len = 1
    t0 = time.time()
    
    # If time is tight, reduce lookahead
    if max_time < 10:
        lookahead_depth = 1

    while True:
        if time.time() > max_time:
            break
        if not beam:
            break

        new_beam = []
        # To avoid too much memory/time, we might limit the number of candidates generated per state
        # But beam_width is the cap for the next iteration.
        
        for cur_path, cur_used, cur_forb in beam:
            cur = cur_path[-1]
            # Candidates are neighbors of cur, not used, and not forbidden
            candidates_mask = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forb)
            
            if candidates_mask == 0:
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = list(cur_path)
                continue

            # Generate candidates and score them
            candidate_scores = []
            for v in _get_bits(candidates_mask):
                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                new_forb = cur_forb | NEIGHBOR_MASKS[cur]
                
                # Lookahead scoring
                # Score is based on how many options we have after moving to v
                next_candidates_mask = NEIGHBOR_MASKS[v] & ~(new_used | new_forb)
                
                if lookahead_depth == 1:
                    score = _bit_count(next_candidates_mask)
                else:
                    # Lookahead 2: For each option w from v, check options from w
                    # To keep it fast, we might just sum the bit counts or take the max
                    # Let's take the max number of options 2 steps ahead
                    max_options_ahead = 0
                    # Limit iterations for speed if too many candidates
                    count = 0
                    for w in _get_bits(next_candidates_mask):
                        count += 1
                        if count > 20: # Heuristic limit for inner loop
                            break
                        next_forb_w = new_forb | NEIGHBOR_MASKS[v]
                        next_next_mask = NEIGHBOR_MASKS[w] & ~(new_used | (1 << w) | next_forb_w)
                        opts = _bit_count(next_next_mask)
                        if opts > max_options_ahead:
                            max_options_ahead = opts
                    
                    # We prefer states that keep more options open, but also penalize if the path is "tight"
                    # A simple score: max_options_ahead
                    score = max_options_ahead
                    # Add a small random perturbation to break ties and explore
                    score += rng.random() * 0.1

                candidate_scores.append((score, v, new_path, new_used, new_forb))

            # Sort candidates for this state by score (descending)
            # We only need the top few to feed into the global beam
            if candidate_scores:
                candidate_scores.sort(key=lambda x: x[0], reverse=True)
                # Keep top 5 candidates from each beam state to control explosion
                for item in candidate_scores[:5]:
                    new_beam.append(item[1:]) # (path, used, forb)
                    if len(item[2]) > best_len:
                        best_len = len(item[2])
                        best_path = list(item[2])

        if not new_beam:
            break

        # Select top beam_width states from new_beam
        # We need to sort new_beam by some global score
        # The score used in candidate_scores was local. Let's re-score or use the last score.
        # To simplify, let's sort by the score computed in the last step.
        # We didn't store the score in new_beam. Let's modify the loop to store it.
        
        # Refined approach: Store (score, path, used, forb) in new_beam
        # Let's restart the inner loop logic slightly to keep scores.
        
        # Actually, the above loop already updated best_path.
        # To select the next beam, we need a global ordering.
        # Let's re-evaluate the score for the top candidates of new_beam.
        
        # Optimization: Instead of re-evaluating, we can just keep the top K based on the score we just calculated.
        # But new_beam contains tuples (path, used, forb). We lost the score.
        # Let's change the structure of new_beam to include the score.
        
        # Let's rewrite the inner loop to collect (score, path, used, forb)
        # I will implement this in the final code.
        
        # For now, let's assume we sort new_beam by a proxy: 
        # We can re-calculate the "next_options" count for the current end of the path.
        
        def _score_state(state):
            p, used, forb = state
            cur = p[-1]
            next_opts = _bit_count(NEIGHBOR_MASKS[cur] & ~(used | forb))
            return next_opts

        new_beam.sort(key=_score_state, reverse=True)
        beam = new_beam[:beam_width]

    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    rng = random.Random(42)
    
    best_path = None
    best_len = 0

    # Phase 1: Quick greedy to get a baseline
    # Greedy: always pick the neighbor with the most future options
    def _greedy_extend(start):
        path = [start]
        used_mask = 1 << start
        forbidden_mask = 0
        while True:
            cur = path[-1]
            candidates_mask = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
            if candidates_mask == 0:
                break
            
            best_v = -1
            best_score = -1
            for v in _get_bits(candidates_mask):
                new_forb = forbidden_mask | NEIGHBOR_MASKS[cur]
                next_opts = _bit_count(NEIGHBOR_MASKS[v] & ~(used_mask | (1 << v) | new_forb))
                if next_opts > best_score:
                    best_score = next_opts
                    best_v = v
            
            if best_v == -1:
                break
            path.append(best_v)
            used_mask |= (1 << best_v)
            forbidden_mask |= NEIGHBOR_MASKS[cur]
            
        return path

    greedy_path = _greedy_extend(0)
    best_path = greedy_path
    best_len = len(greedy_path)

    # Phase 2: Beam Search with multiple configurations
    # We will run several beam searches with different parameters and random starts.
    # Time budget allocation:
    # 10s for quick beams
    # 30s for medium beams
    # 60s for large beams
    # 20s for local improvement / restarts
    
    configs = [
        # (beam_width, duration, start_vertex, lookahead_depth)
        (20, 5.0, 0, 2),
        (50, 10.0, 0, 2),
        (100, 20.0, 0, 2),
        (200, 30.0, 0, 2),
        (300, 40.0, 0, 2),
        (100, 10.0, 1, 2),
        (100, 10.0, 255, 2),
        (150, 15.0, 511, 2),
    ]
    
    for bw, duration, start_v, la in configs:
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time()
        if remaining < duration:
            duration = remaining - 2
        if duration < 1:
            continue
            
        t0 = time.time()
        result = _beam_search(start_v, bw, t0 + duration, rng, la)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result

    # Phase 3: Local Improvement / Restart from best path prefix
    if time.time() < deadline - 10:
        # Try to extend the best path found so far by restarting beam search from its end?
        # Or from a prefix?
        # Restarting from the end is just continuing.
        # Let's try a few more random starts with high beam width if time permits.
        if time.time() < deadline - 5:
             remaining = deadline - time.time() - 2
             if remaining > 5:
                 start_v = rng.randint(0, SIZE-1)
                 result = _beam_search(start_v, 200, time.time() + remaining, rng, 2)
                 if len(result) > best_len:
                     best_len = len(result)
                     best_path = result

    return best_path

# Verification helper (not used in construct, but good for debugging)
def _is_valid_snake(path):
    if len(path) < 2:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        diff = u ^ v
        if diff & (diff - 1) != 0: # Not a power of 2
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            u, v = path[i], path[j]
            diff = u ^ v
            if diff & (diff - 1) == 0: # Power of 2 means distance 1
                return False
    return True

if __name__ == "__main__":
    # Quick self-test
    pass
