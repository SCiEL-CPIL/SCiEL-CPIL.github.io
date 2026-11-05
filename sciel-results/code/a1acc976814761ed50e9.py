import numpy as np
import time

def priority(adjacency):
    N = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Compute Degrees
    degrees = adjacency.sum(axis=1).astype(int)
    
    # 2. Compute Core Numbers
    # Simple peeling algorithm
    core = np.zeros(N, dtype=int)
    deg = degrees.copy()
    active = np.ones(N, dtype=bool)
    
    # We iterate k from 0 to max_deg
    max_deg = deg.max()
    for k in range(max_deg + 1):
        # Find vertices with current degree == k that are active
        # Note: degree changes as we peel
        # This simple loop is O(N * MaxDeg) which is fine for N=200
        while True:
            candidates = np.where((deg == k) & active)[0]
            if len(candidates) == 0:
                break
            for v in candidates:
                if active[v]:
                    core[v] = k
                    active[v] = False
                    # Decrement neighbors
                    neighbors = np.where(adjacency[v])[0]
                    for u in neighbors:
                        if active[u]:
                            deg[u] -= 1
            # Check if new vertices reached degree k
            # The while loop handles this implicitly if we re-check
            # But the standard algorithm is slightly different.
            # Let's use a more robust method:
            # Actually, the above while loop might miss vertices that drop to k later in the same k-iteration?
            # No, if a vertex drops to k, it will be caught in the next iteration of the while loop?
            # Yes, because we re-evaluate `candidates`.
            
    # 3. Initialize Scores
    # Priority: Lower Core is better. Lower Degree is better.
    # Score = - (Core * 10000 + Degree)
    scores = -(core * 10000 + degrees).astype(float)
    
    # Helper: Run Greedy Decoder
    def run_greedy(sc):
        order = np.argsort(-sc) # Descending score
        picked = []
        blocked = np.zeros(N, dtype=bool)
        for v in order:
            if not blocked[v]:
                picked.append(v)
                blocked[v] = True
                neighbors = np.where(adjacency[v])[0]
                blocked[neighbors] = True
        return len(picked), picked, blocked

    # 4. Local Search
    best_size, _, _ = run_greedy(scores)
    current_size = best_size
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Find a candidate swap
        # We want to find a pair (v, u) such that:
        # v is picked, u is blocked by v (or someone else), and u is "better" than v.
        # If we swap scores, u might be picked before v, improving the set.
        
        # Get current selection details
        order = np.argsort(-scores)
        picked = []
        blocked = np.zeros(N, dtype=bool)
        picked_by = {} # v -> who blocked it? Or just track picked order.
        
        # We need to know which picked vertex blocked which skipped vertex?
        # Actually, if u is blocked, it's blocked by *some* picked vertex.
        # If u is "better" than the picker, swapping might help.
        
        # Let's identify "bad picks":
        # A pick v is "bad" if it blocked a vertex u that is better than v.
        
        candidate_v = -1
        candidate_u = -1
        
        # Iterate through picked vertices in order
        # We need to simulate to know who blocked whom?
        # Or just check: for each picked v, check neighbors u.
        # If u is NOT picked, u was blocked.
        # Was u blocked by v?
        # If u is neighbor of v, and v was picked before u (u wasn't picked), then v blocked u?
        # Yes, if v was picked, and u is neighbor, u is blocked.
        # So for any picked v and neighbor u not picked, v "caused" u to be blocked.
        
        for v in order:
            if blocked[v]:
                continue # v was not picked
            # v was picked
            neighbors = np.where(adjacency[v])[0]
            for u in neighbors:
                if not blocked[u]: # u was not blocked? i.e., u was picked?
                    continue
                # u is blocked.
                # Check if u is "better" than v.
                # Better: Lower Core, then Lower Degree.
                if core[u] < core[v]:
                    candidate_v = v
                    candidate_u = u
                    break
                elif core[u] == core[v]:
                    if degrees[u] < degrees[v]:
                        candidate_v = v
                        candidate_u = u
                        break
            if candidate_v != -1:
                break
        
        if candidate_v == -1:
            # No obvious bad pick found based on static metrics.
            # Try a random swap to escape local minima?
            # Or stop if stuck too long.
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            
            # Random Swap
            i, j = np.random.choice(N, 2, replace=False)
            scores[i], scores[j] = scores[j], scores[i]
            
            new_size, _, _ = run_greedy(scores)
            if new_size >= current_size:
                current_size = new_size
                if new_size > best_size:
                    best_size = new_size
                    no_improve_count = 0
            else:
                # Revert
                scores[i], scores[j] = scores[j], scores[i]
            continue
            
        # Apply swap for candidate_v, candidate_u
        v, u = candidate_v, candidate_u
        scores[v], scores[u] = scores[u], scores[v]
        
        new_size, _, _ = run_greedy(scores)
        if new_size >= current_size:
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            scores[v], scores[u] = scores[u], scores[v]
            no_improve_count += 1
            
            # If this swap failed, maybe try another candidate?
            # To keep it simple, we just move to next iteration which will find a new candidate
            # (potentially the same one, causing a loop).
            # To avoid infinite loop on same bad swap, we can mark (v,u) as tabu?
            # Or just accept that we might waste a few steps.
            # Given N=200, we have many candidates.
            
    return scores
