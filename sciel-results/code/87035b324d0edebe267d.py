import numpy as np
import random

def construct():
    # Target n=47 first, then 46
    for n in [47, 46]:
        min_deg_required = n - 39
        
        # Try multiple random instances
        for trial in range(500):
            # Initialize G as random regular-ish
            # Degree approx n - 39
            # For n=46, d=7. For n=47, d=8.
            d = min_deg_required
            
            # Generate random d-regular graph
            # Simple method: random edges, then adjust
            # Or use a known algorithm for random regular graphs
            # For simplicity, start with empty and add edges until deg >= d
            # Then remove C4s
            
            adj = np.zeros((n, n), dtype=int)
            
            # Add random edges
            # To ensure connectivity and high degree, add many edges
            # p = d / n
            p = d / n * 1.5 # Slightly denser
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u, v] = 1
                        adj[v, u] = 1
            
            # Check min degree
            degs = adj.sum(axis=1)
            if degs.min() < min_deg_required:
                # Try to add edges to low-degree vertices
                for _ in range(100):
                    low_vertices = np.where(degs < min_deg_required)[0]
                    if len(low_vertices) == 0:
                        break
                    u = random.choice(low_vertices)
                    # Find non-neighbor with low degree
                    non_neighbors = np.where(adj[u] == 0)[0]
                    non_neighbors = non_neighbors[non_neighbors != u]
                    if len(non_neighbors) == 0:
                        continue
                    # Prefer non-neighbors with low degree
                    cand_degs = degs[non_neighbors]
                    # Sort by degree
                    sorted_cands = non_neighbors[np.argsort(cand_degs)]
                    v = sorted_cands[0]
                    # Check if adding (u,v) creates C4? 
                    # For now, just add, we will remove C4s later
                    adj[u, v] = 1
                    adj[v, u] = 1
                    degs[u] += 1
                    degs[v] += 1
            
            if degs.min() < min_deg_required:
                continue

            # Iterate to remove C4s
            success = True
            max_iter = 2000
            for _ in range(max_iter):
                # Find C4s using A^2
                A2 = adj @ adj
                # C4 exists if A2[u,v] >= 2 and A[u,v] == 0
                # We want to find one such pair
                c4_found = False
                # Vectorized check
                # Mask: A2 >= 2 and A == 0
                # Exclude diagonal
                np.fill_diagonal(A2, 0)
                mask = (A2 >= 2) & (adj == 0)
                
                if not mask.any():
                    break
                
                # Get indices of one C4 pair
                # We need to pick a specific C4 to remove an edge from
                # Get all pairs
                pairs = np.argwhere(mask)
                # Pick a random pair
                u, v = random.choice(pairs)
                
                # Find common neighbors
                common = np.where((adj[u] == 1) & (adj[v] == 1))[0]
                # common has length >= 2
                # Pick two common neighbors b, d
                # C4: u-b-v-d-u
                # Edges: (u,b), (b,v), (v,d), (d,u)
                # We need to remove one of these edges.
                # We must ensure degrees remain >= min_deg_required
                
                b = common[0]
                d = common[1]
                
                edges_to_try = [(u,b), (b,v), (v,d), (d,u)]
                removed = False
                for eu, ev in edges_to_try:
                    if degs[eu] > min_deg_required and degs[ev] > min_deg_required:
                        adj[eu, ev] = 0
                        adj[ev, eu] = 0
                        degs[eu] -= 1
                        degs[ev] -= 1
                        removed = True
                        c4_found = True
                        break
                if not removed:
                    # Cannot remove from this C4
                    # Try another C4?
                    # For simplicity, if we can't remove from the picked one, 
                    # we might be stuck if all C4s are "critical".
                    # But we can try another random C4 pair.
                    # However, if we loop, we might waste time.
                    # Let's just break and restart if we can't remove.
                    # Or try to find another C4.
                    # To keep it simple, if this specific C4 is stuck, 
                    # we try to pick another one in the next iteration.
                    # But if we pick the same one again, we loop.
                    # So we should mark it or pick another.
                    # For now, just continue to next iteration (which picks a new random C4).
                    pass

            # Final verification
            if mask.any(): # This mask is from the last iteration, might be stale
                 # Re-check
                 A2_final = adj @ adj
                 np.fill_diagonal(A2_final, 0)
                 mask_final = (A2_final >= 2) & (adj == 0)
                 if mask_final.any():
                     continue # Still has C4s
            
            if degs.min() >= min_deg_required:
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u, v] == 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    return {'n': 39, 'edges': []}
