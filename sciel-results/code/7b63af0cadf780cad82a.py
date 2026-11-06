import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 115.0  # Safety margin
    
    def is_valid_graph(A):
        # A is numpy array (n, n)
        C = A @ A
        # Check off-diagonal
        mask = ~np.eye(n, dtype=bool)
        return np.all(C[mask] <= 2)

    def repair(A):
        # Remove edges until valid
        C = A @ A
        changed = True
        while changed:
            changed = False
            mask = ~np.eye(n, dtype=bool)
            invalid = np.where(C[mask] > 2)
            if len(invalid[0]) == 0:
                break
            
            # Pick a random invalid pair (i, j)
            idx = random.randint(0, len(invalid[0]) - 1)
            i = invalid[0][idx]
            j = invalid[1][idx]
            
            # Find common neighbors
            common = np.where((A[i] == 1) & (A[j] == 1))[0]
            
            # Pick a random common neighbor k
            # Remove edge (i, k) or (j, k)
            # Let's remove (i, k)
            k = common[random.randint(0, len(common) - 1)]
            
            # Remove edge (i, k)
            A[i, k] = 0
            A[k, i] = 0
            
            # Update C incrementally? Or just recalculate?
            # For simplicity in repair, recalculate if few edges, or just do full matrix mult
            # n=20 is tiny. A @ A is very fast.
            C = A @ A
            changed = True
            
        return A

    def local_search(A, time_limit):
        C = A @ A
        start_time = time.time()
        
        while time.time() < time_limit:
            # 1. Try to add an edge
            added = False
            # Get all non-edges
            non_edges = np.where((A == 0) & (~np.eye(n, dtype=bool)))
            if len(non_edges[0]) > 0:
                # Shuffle to randomize
                idxs = np.arange(len(non_edges[0]))
                random.shuffle(idxs)
                
                for idx in idxs:
                    u = non_edges[0][idx]
                    v = non_edges[1][idx]
                    
                    # Check validity of adding (u, v)
                    # C[u, v] must be <= 2 (it won't change, but if it's >2 we are in invalid state? 
                    # We maintain valid state, so C[u,v] <= 2 is always true for non-edges? 
                    # Wait, C[u,v] is common neighbors. If u,v not connected, C[u,v] can be anything?
                    # No, if graph is valid, all pairs have C <= 2.
                    # So C[u,v] is already <= 2.
                    
                    # Check increments
                    # N(v) = neighbors of v
                    Nv = np.where(A[v] == 1)[0]
                    Nu = np.where(A[u] == 1)[0]
                    
                    # For all x in Nv, C[u, x] must be < 2
                    # C[u, Nv] < 2
                    if np.any(C[u, Nv] >= 2):
                        continue
                    # For all x in Nu, C[v, x] must be < 2
                    if np.any(C[v, Nu] >= 2):
                        continue
                    
                    # Valid addition
                    A[u, v] = 1
                    A[v, u] = 1
                    
                    # Update C
                    # C[u, v] unchanged
                    # C[u, x] += 1 for x in Nv
                    C[u, Nv] += 1
                    C[Nv, u] += 1
                    # C[v, x] += 1 for x in Nu
                    C[v, Nu] += 1
                    C[Nu, v] += 1
                    
                    added = True
                    break
            
            if added:
                continue
                
            # 2. Stuck. Try a swap (remove 1, add 1)
            # Pick a random edge to remove
            edges = np.where((A == 1) & (~np.eye(n, dtype=bool)))
            if len(edges[0]) == 0:
                break
                
            idx = random.randint(0, len(edges[0]) - 1)
            u = edges[0][idx]
            v = edges[1][idx]
            
            # Save state
            A_old = A.copy()
            C_old = C.copy()
            
            # Remove (u, v)
            A[u, v] = 0
            A[v, u] = 0
            
            # Update C
            Nv = np.where(A[v] == 1)[0] # Neighbors of v after removal? 
            # Wait, A is modified. 
            # Neighbors of v in A_old:
            Nv_old = np.where(A_old[v] == 1)[0]
            Nu_old = np.where(A_old[u] == 1)[0]
            
            # C[u, x] -= 1 for x in Nv_old \ {u}
            # C[v, x] -= 1 for x in Nu_old \ {v}
            
            mask_v = np.ones(n, dtype=bool)
            mask_v[u] = False
            mask_u = np.ones(n, dtype=bool)
            mask_u[v] = False
            
            C[u, Nv_old] -= 1
            C[Nv_old, u] -= 1
            C[v, Nu_old] -= 1
            C[Nu_old, v] -= 1
            
            # Try to add a new edge
            added = False
            non_edges = np.where((A == 0) & (~np.eye(n, dtype=bool)))
            if len(non_edges[0]) > 0:
                idxs = np.arange(len(non_edges[0]))
                random.shuffle(idxs)
                
                for idx in idxs:
                    a = non_edges[0][idx]
                    b = non_edges[1][idx]
                    
                    Na = np.where(A[a] == 1)[0]
                    Nb = np.where(A[b] == 1)[0]
                    
                    if np.any(C[a, Nb] >= 2):
                        continue
                    if np.any(C[b, Na] >= 2):
                        continue
                    
                    # Add (a, b)
                    A[a, b] = 1
                    A[b, a] = 1
                    
                    C[a, Nb] += 1
                    C[Nb, a] += 1
                    C[b, Na] += 1
                    C[Na, b] += 1
                    
                    added = True
                    break
            
            if added:
                # We removed 1 and added 1. Edge count is same.
                # But maybe it opens up more additions?
                # Let's accept this move to explore.
                # To avoid cycling, we could track history, but randomization helps.
                pass
            else:
                # Revert
                A = A_old
                C = C_old
                # If we can't even swap, we are likely in a local optimum or dead end.
                # Break to restart? Or just keep trying random swaps?
                # Let's break to allow restart in outer loop.
                break

        return A

    best_edges = 0
    best_graph = None
    
    while time.time() < deadline:
        # Initialize random graph
        p = random.uniform(0.3, 0.6)
        A = (np.random.rand(n, n) < p).astype(int)
        A = (A + A.T) // 2
        np.fill_diagonal(A, 0)
        
        # Repair
        A = repair(A)
        
        # Local Search
        # Give it a small time slice, e.g., 1s or 100ms
        # Or run until stuck?
        # Let's run until stuck, but with a max iteration cap to avoid infinite loops on bad starts
        time_limit = time.time() + 1.0 # 1 second per run
        A = local_search(A, time_limit)
        
        # Check validity and count
        if is_valid_graph(A):
            edges = np.sum(A) // 2
            if edges > best_edges:
                best_edges = edges
                best_graph = A.copy()
        
        # If we found a very high edge count, maybe we can tighten the deadline?
        # No, just keep going.
        
    # Convert best_graph to list of edges
    if best_graph is None:
        # Fallback: empty graph or simple valid graph
        return []
        
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_graph[i, j] == 1:
                edges.append([i, j])
                
    return edges
