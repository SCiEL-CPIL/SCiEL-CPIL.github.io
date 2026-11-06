import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    n = 40
    m = 401
    
    # Initial construction: K20,20 + 1 edge
    # Vertices 0-19 in A, 20-39 in B
    # This has many triangles. 
    # A better start might be a random graph G(n, p) with p ~ m/n^2 = 401/1600 ~ 0.25
    # But K20,20 is a standard starting point. 
    # Actually, let's start with a random graph to break symmetry, 
    # or better, a quasi-random graph like a modular graph?
    # Let's stick to K20,20 + 1 as it's deterministic and known, 
    # but we will use SA to move away from it quickly.
    
    # Initialize adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    edges = set()
    
    # Build K20,20
    for a in range(20):
        for b in range(20, 40):
            adj[a, b] = True
            adj[b, a] = True
            edges.add((a, b))
    # Add 1 edge within A
    adj[0, 1] = True
    adj[1, 0] = True
    edges.add((0, 1))
    
    # Precompute common neighbors matrix C
    # C[u, v] = number of common neighbors of u and v
    # This is adj @ adj
    # Note: For u != v, C[u,v] is the book size of edge (u,v) if (u,v) in E.
    # If (u,v) not in E, C[u,v] is the number of triangles that would be formed by adding (u,v).
    
    C = adj @ adj
    # C is integer matrix. 
    # For diagonal, C[u,u] = degree(u).
    
    # Objective: Sum of squares of book sizes for all edges in E.
    # Let B_e be book size of edge e.
    # Objective = sum_{e in E} B_e^2
    
    def compute_objective(adj, C):
        # We need sum of C[u,v]^2 for all (u,v) in E with u < v
        # We can mask the upper triangle of C^2 with the edge mask
        C_sq = C * C
        # Mask: 1 where edge exists (u < v), 0 otherwise
        mask = np.triu(np.ones((n, n), dtype=bool), 1) & adj
        return np.sum(C_sq[mask])
    
    current_obj = compute_objective(adj, C)
    
    # Track best solution
    best_obj = current_obj
    best_adj = adj.copy()
    best_edges = set(edges)
    
    # Simulated Annealing Parameters
    # We want to minimize max book size, but use sum of squares as guide.
    # Temperature schedule: T = T0 * alpha^step
    T0 = 100.0
    alpha = 0.99
    T = T0
    
    # We will perform steps until time limit.
    # Each step:
    # 1. Generate a candidate move (remove one edge, add one non-edge).
    # 2. Calculate delta in objective function.
    # 3. Accept if better or with probability exp(-delta/T).
    
    # To speed up, we can precompute potential gains.
    # However, dynamic updates of C are O(n^2) per move if done naively.
    # C = A @ A.
    # If we remove (u,v) and add (a,b):
    # A changes at (u,v), (v,u), (a,b), (b,a).
    # New C = (A + deltaA) @ (A + deltaA) = A@A + A@deltaA + deltaA@A + deltaA@deltaA
    # Since deltaA is sparse (4 non-zeros), we can update C efficiently.
    
    # Let's implement efficient C update.
    # Removing edge (u,v):
    # A[u,v] = 0, A[v,u] = 0
    # C_new = C - A[:,u] - A[:,v] - A[u,:] - A[v,:] + ... wait.
    # C[i,j] = sum_k A[i,k] A[k,j]
    # Change in A: A[u,v] -= 1, A[v,u] -= 1.
    # Delta C[i,j] = - (A[i,u] A[v,j] + A[i,v] A[u,j] + A[u,i] A[v,j] + A[v,i] A[u,j]) ... this is getting messy.
    
    # Simpler: Just recompute C = A @ A.
    # n=40. A @ A is 40x40x40 = 64,000 ops.
    # In NumPy, this is extremely fast (< 1ms).
    # If we do 10,000 steps, that's 10 seconds. Feasible.
    
    # Let's optimize the move selection.
    # Instead of random, we can pick the best move from a sample.
    
    # List of current edges and non-edges
    # We need fast access.
    # We'll just sample.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Pick a random edge to remove
        # To avoid O(N) list conversion, we can maintain a list and swap-remove.
        # But for simplicity, let's use the set and random.choice. 
        # random.choice on a set is O(N). N=400. 400 ops is fine.
        
        # To speed up, let's maintain a list of edges.
        # We'll rebuild the list if it gets too stale, or just use set.
        # Actually, let's just use the set. 400 items is small.
        
        e_remove = random.choice(list(edges))
        u, v = e_remove
        
        # Pick a random non-edge to add
        # Number of non-edges = 780 - 401 = 379.
        # We can sample until we find a non-edge.
        found = False
        for _ in range(20):
            a, b = random.sample(range(n), 2)
            if not adj[a, b]:
                found = True
                break
        if not found:
            continue
            
        # Calculate Delta Objective
        # Old Obj: sum_{e in E} C_e^2
        # New Graph: E' = E - {(u,v)} + {(a,b)}
        # C changes.
        
        # We need C' to calculate new book sizes.
        # Let's just compute C' = A' @ A'
        
        # Apply change to A
        adj[u, v] = False
        adj[v, u] = False
        adj[a, b] = True
        adj[b, a] = True
        
        C_new = adj @ adj
        
        # Calculate New Objective
        mask_new = np.triu(np.ones((n, n), dtype=bool), 1) & adj
        new_obj = np.sum(C_new[mask_new])
        
        delta = new_obj - current_obj
        
        # Check if this is a valid move (it is, by construction)
        # Acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            current_obj = new_obj
            C = C_new
            edges.discard(e_remove)
            edges.add((a, b))
            
            # Update best
            if current_obj < best_obj:
                best_obj = current_obj
                best_adj = adj.copy()
                best_edges = set(edges)
        else:
            # Revert
            adj[u, v] = True
            adj[v, u] = True
            adj[a, b] = False
            adj[b, a] = False
            # C is unchanged, current_obj unchanged
            
        # Cool down
        T *= alpha
        
        # Optional: If T is very low, we might get stuck.
        # We could restart if stuck for a long time, but let's keep it simple.
        
    # Return best found
    # Convert best_adj back to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                result.append([i, j])
                
    # Sanity check
    if len(result) != 401:
        # Fallback to initial
        result = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result
