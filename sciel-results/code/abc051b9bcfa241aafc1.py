import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # Precompute conflict matrix or just use the set for O(1) lookup
    # conflict[i][j] = 1 if (i-j)%m in forbidden
    # Since m=205, a 205x205 matrix is fine (42k entries)
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i != j:
                d = (i - j) % m
                if d in forbidden:
                    conflict[i][j] = True
                    conflict[j][i] = True # Symmetric

    def is_valid_addition(S_set, v):
        # Check if v can be added to S_set
        for u in S_set:
            if conflict[v][u]:
                return False
        return True

    def greedy_init(random_order=True):
        S = []
        S_set = set()
        candidates = list(range(m))
        if random_order:
            random.shuffle(candidates)
        
        for v in candidates:
            if is_valid_addition(S_set, v):
                S.append(v)
                S_set.add(v)
        return S, S_set

    def local_search(S, S_set, max_steps=1000):
        # Try to improve S by adding vertices or swapping
        # S is a list, S_set is a set for fast lookup
        # We return the best S found
        
        best_S = list(S)
        best_S_set = set(S)
        best_size = len(S)
        
        # We will iterate, trying to add, then swap
        # To avoid infinite loops, we limit steps or use a tabu-like mechanism
        # Simple approach: try all valid adds. If none, try all valid swaps.
        
        steps = 0
        while steps < max_steps:
            steps += 1
            
            # 1. Try to add any vertex
            added = False
            # Randomize order of candidates not in S to break symmetry
            candidates = [v for v in range(m) if v not in S_set]
            random.shuffle(candidates)
            
            for v in candidates:
                if is_valid_addition(S_set, v):
                    S.append(v)
                    S_set.add(v)
                    added = True
                    break # Restart the loop to try adding more
                    
            if added:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                    best_S_set = set(S)
                continue # Restart inner logic
            
            # 2. No vertex can be added. Try swaps.
            # Pick a random u in S and a random v not in S
            # Check if S - {u} + {v} is valid
            swap_made = False
            u_candidates = list(S)
            random.shuffle(u_candidates)
            
            for u in u_candidates:
                # Temporarily remove u
                S_set.remove(u)
                S.remove(u)
                
                # Try to add any v not in S
                # We want to see if we can add at least one v
                # If we add v, size is same. But maybe we can add more later.
                # For now, just do a random valid swap to move in state space.
                candidates = [v for v in range(m) if v not in S_set]
                random.shuffle(candidates)
                
                for v in candidates:
                    if is_valid_addition(S_set, v):
                        S.append(v)
                        S_set.add(v)
                        swap_made = True
                        break
                
                if swap_made:
                    break
                else:
                    # Revert
                    S.append(u)
                    S_set.add(u)
                    # If no swap found for this u, try next u? 
                    # Or just revert and try a different strategy?
                    # Let's just revert and try next u.
                    continue

            if not swap_made:
                # Stuck completely? Restart
                return best_S, best_S_set, best_size

        return best_S, best_S_set, best_size

    start_time = time.time()
    deadline = start_time + 600 # 10 min, safe margin
    
    best_overall_S = []
    best_overall_size = 0
    
    # Initial greedy
    S, S_set = greedy_init(random_order=False)
    best_overall_S = list(S)
    best_overall_size = len(S)
    
    # Local search on initial
    S, S_set, size = local_search(S, S_set, max_steps=500)
    if size > best_overall_size:
        best_overall_size = size
        best_overall_S = list(S)
        
    # Multiple restarts
    while time.time() < deadline:
        S, S_set = greedy_init(random_order=True)
        
        # Quick check if this start is promising
        if len(S) < best_overall_size - 2:
            continue # Unlikely to beat best if start is much smaller
            
        S, S_set, size = local_search(S, S_set, max_steps=1000)
        
        if size > best_overall_size:
            best_overall_size = size
            best_overall_S = list(S)
            
        # Optional: If we hit a very high size, we might try to optimize further
        # but for now, restarts are good.

    return np.array(best_overall_S, dtype=np.int64)
