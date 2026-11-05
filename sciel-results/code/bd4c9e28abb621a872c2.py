"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
TIME_LIMIT = 110  # safety margin below 120s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_path = None
    best_len = 0

    def elapsed():
        return time.time() - start_time

    def time_left():
        return TIME_LIMIT - elapsed()

    def get_neighbors(v):
        """Generate all neighbors of v in Q_N."""
        for b in range(N):
            yield v ^ (1 << b)

    def score_candidate(v, path, used, forbidden):
        """Score a candidate vertex by counting its valid extensions."""
        cur = path[-1]
        score = 0
        for b in range(N):
            v2 = v ^ (1 << b)
            if v2 == cur:
                continue
            if v2 not in used and v2 not in forbidden:
                score += 1
        return score

    def beam_search(start_vertex, beam_width=50, max_steps=512):
        """Beam search maintaining multiple partial paths."""
        # Each beam entry: (path, used_set, forbidden_set, score)
        beams = []
        initial_path = [start_vertex]
        initial_used = {start_vertex}
        initial_forbidden = set()
        beams.append((initial_path, initial_used, initial_forbidden, 0))

        for step in range(max_steps):
            if time_left() < 5.0:
                break

            new_beams = []
            for path, used, forbidden, _ in beams:
                cur = path[-1]
                candidates = []
                for b in range(N):
                    v = cur ^ (1 << b)
                    if v not in used and v not in forbidden:
                        # Score this candidate
                        s = score_candidate(v, path, used, forbidden)
                        candidates.append((s, v))
                
                # Sort by score descending, take top few
                candidates.sort(key=lambda x: -x[0])
                for s, v in candidates[:min(3, len(candidates))]:
                    # Create new beam state
                    new_path = path + [v]
                    new_used = used | {v}
                    new_forbidden = set(forbidden)
                    for nb in get_neighbors(cur):
                        new_forbidden.add(nb)
                    new_beams.append((new_path, new_used, new_forbidden, s))
            
            if not new_beams:
                break

            # Keep top beam_width beams
            new_beams.sort(key=lambda x: -len(x[0]))
            beams = new_beams[:beam_width]

        # Return the longest path from all beams
        if not beams:
            return [start_vertex]
        
        longest = max(beams, key=lambda x: len(x[0]))
        return longest[0]

    def try_local_improvement(path, max_rounds=5):
        """Try to extend or improve path by trying different starting points."""
        current_best = list(path)
        current_len = len(current_best)
        
        for _ in range(max_rounds):
            if time_left() < 10.0:
                break
            
            # Try extending from the end
            extended = list(current_best)
            used = set(extended)
            forbidden = set()
            for i in range(len(extended) - 1):
                for nb in get_neighbors(extended[i]):
                    forbidden.add(nb)
            
            cur = extended[-1]
            improved = False
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    extended.append(v)
                    used.add(v)
                    for nb in get_neighbors(cur):
                        forbidden.add(nb)
                    cur = v
                    improved = True
                    break
            
            if not improved and len(extended) > current_len:
                current_best = extended
                current_len = len(current_best)
        
        return current_best

    # Run beam search with different parameters
    best_path = None
    best_len = 0

    # First, try a simple greedy to get a baseline
    def greedy_extend(start_v):
        path = [start_v]
        used = {start_v}
        forbidden = set()
        while time_left() > 2.0:
            cur = path[-1]
            found = False
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    for nb in get_neighbors(cur):
                        forbidden.add(nb)
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        return path

    baseline = greedy_extend(0)
    best_path = baseline
    best_len = len(baseline)

    # Now run beam search multiple times with random starts
    while time_left() > 15.0:
        sv = random.randint(0, 2**N - 1)
        beam_width = random.choice([30, 50, 80, 120])
        p = beam_search(sv, beam_width=beam_width)
        
        # Try local improvement
        p = try_local_improvement(p)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    return best_path
# EVOLVE-BLOCK-END
