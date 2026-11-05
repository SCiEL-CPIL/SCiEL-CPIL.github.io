import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 40
    m = 401
    
    best_score = float('inf')
    best_A = None
    
    def get_edges(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    edges.append([i, j])
        return edges

    def compute_score(A):
        B = A @ A
        # Mask non-edges
        B_masked = np.where(A > 0, B, 0)
        return np.max(B_masked)

    # Random restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        # Start with a random set of 401 edges
        A = np.zeros((n, n), dtype=int)
        edges = []
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not A[u, v]:
                A[u, v] = 1
                A[v, u] = 1
                edges.append((u, v))
        
        current_score = compute_score(A)
        
        # Local search
        steps = 0
        max_steps = 5000
        while steps < max_steps and time.time() < deadline:
            steps += 1
            B = A @ A
            B_masked = np.where(A > 0, B, 0)
            
            # Find edges with max book size
            max_val = np.max(B_masked)
            if max_val <= 7: # Theoretical min is 7
                break
            
            # Pick a random edge with max book size
            candidates = np.argwhere(B_masked == max_val)
            # Filter for i < j to avoid duplicates
            valid_cands = [tuple(c) for c in candidates if c[0] < c[1]]
            if not valid_cands:
                break
            u, v = random.choice(valid_cands)
            
            # Try to move u to w or v to w
            # Pick a random w
            improved = False
            for _ in range(20): # Try 20 random w's
                if time.time() > deadline:
                    break
                w = random.randint(0, n-1)
                if w == u or w == v:
                    continue
                # Check if edge (u,w) or (v,w) can be formed
                # We are moving u to w: remove (u,v), add (u,w)
                # Condition: A[u,w] == 0
                if A[u, w] == 1:
                    continue
                
                # Perform move
                A[u, v] = 0
                A[v, u] = 0
                A[u, w] = 1
                A[w, u] = 1
                
                new_score = compute_score(A)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                elif new_score == current_score:
                    # Accept with some probability to allow exploration
                    # Or check secondary metric: sum of squares
                    # For now, accept with prob 0.5
                    if random.random() < 0.5:
                        improved = True
                        break
                    else:
                        # Revert
                        A[u, v] = 1
                        A[v, u] = 1
                        A[u, w] = 0
                        A[w, u] = 0
                
                # Revert if not improved
                A[u, v] = 1
                A[v, u] = 1
                A[u, w] = 0
                A[w, u] = 0
            
            if not improved:
                # Stuck, break local search and restart
                break
        
        if current_score < best_score:
            best_score = current_score
            best_A = A.copy()
        
        if best_score <= 7:
            break

    if best_A is None:
        # Fallback to seed
        A = np.zeros((n, n), dtype=int)
        for a in range(20):
            for b in range(20, 40):
                A[a, b] = 1
                A[b, a] = 1
        A[0, 1] = 1
        A[1, 0] = 1
        best_A = A

    return get_edges(best_A)

# EVOLVE-BLOCK-END
