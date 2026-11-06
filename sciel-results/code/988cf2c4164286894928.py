"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 9
TOTAL = 1 << N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a snake in Q_9 using Greedy Best-First Search with Lookahead.
    This approach selects the next vertex based on the number of valid successors,
    avoiding dead ends more effectively than pure random backtracking in the early stages.
    """
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    best_path = [0]
    best_len = 1
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(TOTAL)]
    for v in range(TOTAL):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    def greedy_solve(rng):
        nonlocal best_path, best_len
        
        # Start from a random vertex to break symmetry
        start_v = rng.randint(0, TOTAL - 1)
        path = [start_v]
        used = [False] * TOTAL
        used[start_v] = True
        
        # To efficiently check validity, we maintain a set of "blocked" vertices.
        # A vertex is blocked if it is in the path or adjacent to a vertex in the path (except the last one).
        # Actually, for the next step, we only care about vertices that are NOT in the path 
        # and NOT adjacent to any vertex in path[:-1].
        
        # Let's maintain `blocked` as a boolean array.
        # Initially, `blocked` contains all neighbors of start_v? No.
        # If we are at `v`, the next vertex `u` must not be in `path` and must not be adjacent to any `w in path[:-1]`.
        
        # Efficient update:
        # When we add `u` to the path, the previous last vertex `v` becomes "old".
        # `v` now blocks its neighbors (except `u` and `path[-2]`? No, `v` is adjacent to `path[-2]` and `u`).
        # Any neighbor `w` of `v` (other than `path[-2]` and `u`) becomes blocked for future steps.
        
        blocked = [False] * TOTAL
        
        # Initialize: start_v is in path.
        # Its neighbors are not yet blocked in the sense of "cannot be chosen next", 
        # but they are adjacent to start_v. 
        # Wait, the rule is: non-consecutive vertices differ by >= 2.
        # So if path is [v0], next is v1. v1 is adjacent to v0.
        # Next is v2. v2 must be distance >= 2 from v0.
        # So v0 blocks its neighbors (except v1).
        
        # Let's refine the `blocked` logic.
        # `blocked[x]` is True if x cannot be the *next* vertex.
        # x cannot be the next vertex if:
        # 1. x is in the path.
        # 2. x is adjacent to any vertex in path[:-1] (the "old" part of the path).
        
        # Initially path = [v0]. path[:-1] is empty.
        # So no vertex is blocked by adjacency to old path.
        # But v0 is in path, so v0 is "used".
        
        # When we extend path by u (new last):
        # The old last v becomes part of path[:-1].
        # So v now blocks its neighbors.
        # Specifically, for each neighbor w of v:
        #   if w is not the new last u and w is not the previous previous (path[-2]? No, path[-2] is v's neighbor in the path).
        #   Actually, v is adjacent to path[-2] (let's call it v_prev) and u (new last).
        #   v blocks all its neighbors w such that w is not in the path.
        #   Since v_prev and u are in the path, we don't need to block them via this mechanism (they are already used).
        #   But we must ensure that v does NOT block v_prev or u in a way that prevents future steps?
        #   No, v_prev and u are already in the path.
        #   The key is: any future vertex w must not be adjacent to v.
        #   So we mark all neighbors of v as blocked, EXCEPT those already in the path (which are handled by `used`).
        
        # Algorithm for adding u to path ending in v:
        # 1. Mark u as used.
        # 2. For each neighbor w of v:
        #      if w is not used:
        #          blocked[w] = True
        #      (Note: w might be u, but u is now used, so we skip it or it's handled by used check later)
        #      (Note: w might be path[-2], which is used, so skipped)
        
        # To "unblock" or backtrack? We are doing greedy, so no backtracking.
        
        current = start_v
        while True:
            # Find valid candidates
            candidates = []
            for u in neighbors[current]:
                if not used[u] and not blocked[u]:
                    candidates.append(u)
                    
            if not candidates:
                break
                
            # Evaluate candidates
            best_score = -1
            best_u = None
            best_ties = []
            
            for u in candidates:
                # Score: number of valid next-next vertices
                score = 0
                # Simulate adding u
                # The new "old" vertex will be current.
                # So current will block its neighbors.
                # The valid next-next vertices w must be:
                # 1. Neighbors of u
                # 2. Not used
                # 3. Not blocked (current logic)
                # 4. Not blocked by the new blocking of current's neighbors
                
                # Let's calculate score more carefully.
                # If we pick u, the next vertex w must be a neighbor of u.
                # w must not be used.
                # w must not be in `blocked`.
                # w must not be a neighbor of `current` (because current becomes "old" and blocks its neighbors).
                # Exception: if w is `current`? No, w is neighbor of u, current is neighbor of u. 
                # If w == current, then dist(w, current) = 1, but they are consecutive? 
                # Path: ... -> current -> u -> w.
                # w is consecutive to u. w is non-consecutive to current.
                # So dist(w, current) must be >= 2.
                # If w is a neighbor of current, dist(w, current) = 1. So w cannot be a neighbor of current.
                
                # So, valid w for next step after u:
                # w in neighbors[u]
                # not used[w]
                # not blocked[w]
                # w not in neighbors[current]  (i.e., dist(w, current) >= 2)
                
                # We can precompute a set of neighbors of current for fast lookup
                # Or just check.
                
                for w in neighbors[u]:
                    if used[w] or blocked[w]:
                        continue
                    # Check if w is adjacent to current
                    # This is equivalent to checking if w is in neighbors[current]
                    # Since N=9, neighbors list is small.
                    # A faster check: if w is in neighbors[current], skip.
                    # We can use a set for neighbors[current] if needed, but list is fine for small N.
                    # Actually, we can just check if (w ^ current).bit_count() == 1
                    if (w ^ current).bit_count() == 1:
                        continue
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_u = u
                    best_ties = [u]
                elif score == best_score:
                    best_ties.append(u)
            
            if best_u is None:
                break
                
            # Pick randomly from best ties to introduce diversity
            u = rng.choice(best_ties)
            
            # Commit to u
            path.append(u)
            used[u] = True
            
            # Update blocked status
            # current becomes "old", so it blocks its neighbors
            for w in neighbors[current]:
                if not used[w]:
                    blocked[w] = True
                    
            current = u
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                
            if time.time() - start_time > time_limit:
                return best_path
                
        return best_path

    # Run multiple trials
    for trial in range(10000):
        if time.time() - start_time > time_limit:
            break
        rng = random.Random(trial)
        p = greedy_solve(rng)
        # p is the path found in this trial. 
        # Note: greedy_solve updates best_path nonlocally, but also returns it.
        # We rely on the nonlocal update inside greedy_solve.
        
    return best_path

# EVOLVE-BLOCK-END
