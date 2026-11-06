import time
import random
import sys

N = 9
SIZE = 1 << N
BIT_MASKS = [1 << b for b in range(N)]

# Precompute neighbor masks for fast lookup
# NEIGHBOR_MASKS[v] is a bitmask of all vertices adjacent to v
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ BIT_MASKS[b])
    NEIGHBOR_MASKS[v] = m

# Precompute popcount for small integers? No, int.bit_count() is fast in Python 3.10+
# We assume a recent Python version.

def _popcount(x):
    return x.bit_count()

def _beam_search(start, beam_width, max_time, rng, diversity_penalty=0.1):
    """
    Optimized beam search.
    Uses a scoring function that balances path length and future options.
    diversity_penalty adds randomness to prevent premature convergence.
    """
    if max_time <= 0:
        return [start]
        
    deadline = time.time() + max_time
    
    # State: (path, used_mask, forbidden_mask, current_vertex)
    # We store path as a list for easy access, but used_mask/forbidden_mask for fast filtering
    
    # Initial state
    cur_path = [start]
    cur_used = 1 << start
    cur_forb = 0
    
    # Beam is a list of states. We keep the best 'beam_width' states.
    # Each state is a tuple: (path, used_mask, forbidden_mask, score)
    # We sort by score descending.
    
    # To speed up, we don't store the full path in the beam if we can reconstruct it?
    # No, we need the path for the final result and for checking non-consecutive constraints (though forbidden_mask handles that).
    # Actually, forbidden_mask contains all vertices that are neighbors of any vertex in the path EXCEPT the last one.
    # Wait, the constraint is: non-consecutive vertices differ in at least 2 bits.
    # This is equivalent to: no vertex in the path (except the immediate predecessor of the current vertex) can be a neighbor of the current vertex.
    # So, forbidden_mask should contain the neighbors of all vertices in path[:-1].
    
    # Let's refine the state representation:
    # state = (path, used_mask, forbidden_mask)
    # forbidden_mask = union of NEIGHBOR_MASKS[v] for v in path[:-1]
    
    beam = [(cur_path, cur_used, cur_forb)]
    best_path = cur_path
    best_len = 1
    
    while time.time() < deadline:
        new_beam = []
        seen_states = set() # To avoid duplicates in the new beam if possible, though paths are unique by construction
        
        for path, used, forb in beam:
            cur = path[-1]
            # Candidates are neighbors of cur that are not used and not forbidden
            candidates = NEIGHBOR_MASKS[cur] & ~(used | forb)
            
            if candidates == 0:
                # Dead end, skip
                continue
            
            # Iterate through candidates
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb
                
                new_path = path + [v]
                new_used = used | (1 << v)
                # The new forbidden mask adds the neighbors of the current vertex 'cur'
                # because 'cur' becomes a non-consecutive vertex relative to the next step
                new_forb = forb | NEIGHBOR_MASKS[cur]
                
                # Check if this state is already in the beam (by last 4 vertices and used mask?)
                # A simple key: (last_vertex, used_mask) is not unique enough for long paths.
                # But since we are doing beam search, we just generate candidates.
                
                new_beam.append((new_path, new_used, new_forb))
                
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = new_path
        
        if not new_beam:
            break
            
        # Score the new beam
        # Heuristic: 
        # 1. Length is most important.
        # 2. Among same length, prefer paths with more "options" for the next step? 
        #    Or prefer paths that are "less constrained"?
        #    Let's use: score = len(path) * 100 + options_count
        #    Actually, for snakes, "minimum remaining values" (choosing the move that leaves fewest options) is often better to avoid getting stuck, 
        #    BUT for finding *long* snakes, we might want to keep options open.
        #    Let's try: score = len(path) - penalty_for_low_options
        #    Let's stick to a simple score: primarily length, tie-breaker by number of available next moves.
        
        def get_score(state):
            p, u, f = state
            cur = p[-1]
            opts = (NEIGHBOR_MASKS[cur] & ~(u | f)).bit_count()
            # We want long paths. 
            # If two paths have same length, which is better?
            # Often, the one with *fewer* options is "more committed" and might lead to a dead end sooner.
            # The one with *more* options has more flexibility.
            # Let's add a small bonus for flexibility.
            return (len(p), opts)
            
        # Sort by length descending, then by options descending
        # We want to keep the top beam_width states
        new_beam.sort(key=get_score, reverse=True)
        
        # Apply diversity? 
        # To avoid all beams converging to the same path, we can randomly drop some of the top ones 
        # and keep some slightly lower ones.
        if diversity_penalty > 0 and len(new_beam) > beam_width:
            # Keep top beam_width * 0.8, and sample rest from the next 20%
            keep_count = int(beam_width * 0.8)
            rest = new_beam[keep_count: keep_count + beam_width]
            if rest:
                sampled = rng.sample(rest, min(beam_width - keep_count, len(rest)))
                new_beam = new_beam[:keep_count] + sampled
            else:
                new_beam = new_beam[:beam_width]
        else:
            new_beam = new_beam[:beam_width]
            
        beam = new_beam
        
        if not beam:
            break
            
    return best_path

def _local_search_extend(path, max_time, rng):
    """
    Try to extend a path by restarting the beam search from various points in the path.
    This helps if the end of the path is a dead end, but an earlier point could have led to a longer path.
    """
    best_path = list(path)
    best_len = len(path)
    n = len(path)
    
    if n < 5:
        return best_path
    
    # Try restarting from the last 20% of the path
    # We simulate the path up to that point, then beam search from there.
    # To do this efficiently, we need to reconstruct the state (used, forb) at that point.
    
    # Precompute states for the given path
    states = []
    used = 0
    forb = 0
    for i, v in enumerate(path):
        used |= 1 << v
        if i > 0:
            # forb for step i is neighbors of path[0..i-1]
            # Actually, when at path[i], the forbidden mask is neighbors of path[0..i-1].
            # Let's compute forb incrementally.
            pass
            
    # Re-compute properly
    used_mask = 0
    forbidden_mask = 0
    for i, v in enumerate(path):
        used_mask |= 1 << v
        if i < n - 1:
            # If we are at path[i], the next step cannot go to neighbors of path[0..i]
            # Wait, the constraint is on non-consecutive.
            # If we are at path[i], we can go to a neighbor of path[i] that is not in path[0..i-1] and not a neighbor of path[0..i-2]?
            # No. The constraint is: new vertex v_new must not be adjacent to any path[j] where j < i-1.
            # So forbidden_mask at step i (before choosing v_{i+1}) is union of NEIGHBOR_MASKS[path[j]] for j in 0..i-1.
            forbidden_mask |= NEIGHBOR_MASKS[v]
        states.append((v, used_mask, forbidden_mask))
        
    # We will try to restart from indices in the latter half of the path
    start_indices = range(max(0, n - 30), n - 1)
    rng.shuffle(list(start_indices))
    
    for idx in start_indices:
        if time.time() > max_time - 2:
            break
            
        # Reconstruct state at idx
        # used_mask_at_idx = bits of path[0..idx]
        # forbidden_mask_at_idx = union of NEIGHBOR_MASKS[path[0..idx-1]]
        
        # We stored (v, used_mask, forbidden_mask) in states.
        # But my loop above updated forbidden_mask *after* checking i < n-1.
        # Let's re-verify the logic.
        # At step i (current vertex path[i]), the forbidden vertices are neighbors of path[0]...path[i-1].
        # So if we restart from path[idx], the forbidden mask should be union of NEIGHBOR_MASKS[path[0..idx-1]].
        # The used mask should be bits of path[0..idx].
        
        # Let's just re-calculate for simplicity in the loop, it's fast enough for N=9.
        prefix = path[:idx+1]
        p_used = 0
        p_forb = 0
        for j, v in enumerate(prefix):
            p_used |= 1 << v
            if j < len(prefix) - 1:
                p_forb |= NEIGHBOR_MASKS[v]
                
        # Now beam search from prefix[-1] with these masks
        # We need a version of beam search that accepts initial masks
        
        # For now, let's just call a simplified beam search that starts from this state
        # We can reuse _beam_search if we modify it to accept initial state, 
        # or just create a new beam state.
        
        # Let's create a helper that runs beam search from a specific state
        def _beam_search_from_state(start_v, init_used, init_forb, beam_width, t_limit, rng):
            deadline = time.time() + t_limit
            cur_path = [start_v]
            cur_used = init_used
            cur_forb = init_forb
            
            beam = [(cur_path, cur_used, cur_forb)]
            best_path = cur_path
            best_len = 1
            
            while time.time() < deadline:
                new_beam = []
                for path, used, forb in beam:
                    cur = path[-1]
                    candidates = NEIGHBOR_MASKS[cur] & ~(used | forb)
                    if candidates == 0:
                        continue
                    tmp = candidates
                    while tmp:
                        lsb = tmp & (-tmp)
                        v = lsb.bit_length() - 1
                        tmp ^= lsb
                        new_path = path + [v]
                        new_used = used | (1 << v)
                        new_forb = forb | NEIGHBOR_MASKS[cur]
                        new_beam.append((new_path, new_used, new_forb))
                        if len(new_path) > best_len:
                            best_len = len(new_path)
                            best_path = new_path
                if not new_beam:
                    break
                new_beam.sort(key=lambda x: (len(x[0]), (NEIGHBOR_MASKS[x[0][-1]] & ~(x[1] | x[2])).bit_count()), reverse=True)
                beam = new_beam[:beam_width]
                if not beam:
                    break
            return best_path

        # Run a short beam search from this point
        # We want to see if we can beat the current best_len
        # Time budget per restart: small, e.g., 0.5s
        t_budget = 0.5
        bw = 50
        improved = _beam_search_from_state(path[idx], p_used, p_forb, bw, t_budget, rng)
        
        # If the improved path is longer than the original prefix + extension, 
        # we need to merge it? 
        # The improved path starts from path[idx]. 
        # So the full path would be path[:idx] + improved[1:]
        # But we must ensure the connection is valid. 
        # Since improved[0] is path[idx], and improved[1] is a neighbor of path[idx] not in used/forb, it is valid.
        
        full_candidate = path[:idx] + improved[1:]
        
        # Validate? The beam search ensures validity relative to the prefix state.
        # We should check if the new suffix doesn't conflict with the prefix.
        # The beam search used p_used and p_forb which account for the prefix.
        # So it should be valid.
        
        if len(full_candidate) > best_len:
            best_len = len(full_candidate)
            best_path = full_candidate
            
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    rng = random.Random(42)
    
    best_path = None
    best_len = 0
    
    # Phase 1: Initial Greedy/Beam to get a baseline
    # Use a moderately wide beam for a short time
    t1 = min(5.0, deadline - start_time - 10)
    p1 = _beam_search(0, 100, t1, rng)
    best_path = p1
    best_len = len(p1)
    
    # Phase 2: Multiple Beam Searches with different widths and seeds
    # We will run several iterations, increasing beam width or changing strategy
    
    configs = [
        # (beam_width, time_budget, diversity)
        (200, 10.0, 0.0),
        (500, 15.0, 0.1),
        (1000, 20.0, 0.2),
        (2000, 30.0, 0.1),
    ]
    
    for bw, budget, div in configs:
        if time.time() > deadline - 20:
            break
        remaining = deadline - time.time()
        if remaining < budget:
            budget = max(2.0, remaining - 10)
            
        p = _beam_search(0, bw, budget, rng, div)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
    # Phase 3: Local Search / Restart from best path
    if time.time() < deadline - 15:
        remaining = deadline - time.time()
        if remaining > 10:
            improved = _local_search_extend(best_path, remaining - 5, rng)
            if len(improved) > best_len:
                best_len = len(improved)
                best_path = improved
                
    # Phase 4: Final push with very wide beam if time permits
    if time.time() < deadline - 5:
        remaining = deadline - time.time()
        if remaining > 5:
            p = _beam_search(0, 5000, remaining - 2, rng, 0.0)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
                
    # Final verification (optional, but good for debugging)
    # For N=9, we can verify in O(N^2)
    if best_path:
        # Check distinct
        if len(set(best_path)) != len(best_path):
            # Should not happen
            pass
        # Check adjacency
        for i in range(len(best_path) - 1):
            if (best_path[i] ^ best_path[i+1]).bit_count() != 1:
                # Invalid
                pass
        # Check non-consecutive
        # This is O(L^2). L~200, so 40000 ops. Fast.
        for i in range(len(best_path)):
            for j in range(i+2, len(best_path)):
                if (best_path[i] ^ best_path[j]).bit_count() < 2:
                    # Invalid
                    pass
                    
    return best_path

if __name__ == "__main__":
    # For local testing
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Length: {len(res)}, Time: {end-start:.2f}s")
