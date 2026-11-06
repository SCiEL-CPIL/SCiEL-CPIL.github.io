import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Target n=46, min_deg=7
    # Try n=46 first. If fails, try 45, etc.
    
    # Hardcode Hoffman-Singleton Graph (50 vertices, 7-regular, C4-free)
    # Generated from known structure or hardcoded edges.
    # Since I can't easily generate it from scratch without a library, I will use a 
    # known property or a simpler high-degree C4-free graph if possible.
    # Actually, I can construct a C4-free graph using the Greedy method with the 
    # relaxed constraint (common neighbors <= 1) and hope it works for n=46.
    
    best_result = {'n': 39, 'edges': []}
    
    for n in [46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = n - 39
        if min_deg_needed <= 0:
            best_result = {'n': n, 'edges': []}
            continue
            
        success = False
        # Multiple random restarts
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 1000 + n)
            
            adj = [set() for _ in range(n)]
            edges = []
            
            # Greedy addition
            # Sort all potential edges by (deg[u] + deg[v]) to balance
            # But we need to re-evaluate degrees.
            
            # We will do passes.
            # In each pass, we look for the "best" edge to add.
            # Best edge: minimizes (deg[u] + deg[v]) among valid edges.
            
            for pass_num in range(n * 2): # Enough passes to converge
                best_edge = None
                best_score = float('inf')
                
                # To speed up, we can just iterate and pick the first valid one 
                # with low degree sum, or randomize.
                # Given n=46, n^2 is small.
                
                candidates = []
                for u in range(n):
                    for v in range(u+1, n):
                        if v in adj[u]:
                            continue
                        # Check if adding (u,v) is safe
                        # 1. |N(u) ∩ N(v)| <= 1
                        common_uv = len(adj[u] & adj[v])
                        if common_uv > 1:
                            continue
                            
                        # 2. Check if adding (u,v) causes any other pair to have >1 common neighbors
                        # Only pairs (u,z) and (v,z) are affected.
                        safe = True
                        for z in range(n):
                            if z == u or z == v:
                                continue
                            # Check pair (u, z)
                            if z in adj[u]:
                                # v is added to N(u). 
                                # If v is in N(z), then v becomes a common neighbor of u and z.
                                if v in adj[z]:
                                    # Count common neighbors of u and z AFTER adding (u,v)
                                    # Current common: adj[u] & adj[z]
                                    # New common: (adj[u] | {v}) & adj[z]
                                    # Since v in adj[z], v is in the intersection.
                                    # We need to check if the TOTAL count becomes > 1.
                                    # Current count: len(adj[u] & adj[z])
                                    # If v was NOT in adj[u] (it wasn't, we are adding it), 
                                    # then v is a NEW common neighbor.
                                    # So new count = old count + 1.
                                    # We need new count <= 1 => old count <= 0.
                                    # So if old count >= 1, it becomes >= 2. Unsafe.
                                    if len(adj[u] & adj[z]) >= 1:
                                        safe = False
                                        break
                            # Check pair (v, z)
                            if z in adj[v]:
                                if u in adj[z]:
                                    if len(adj[v] & adj[z]) >= 1:
                                        safe = False
                                        break
                        if not safe:
                            continue
                        
                        score = len(adj[u]) + len(adj[v])
                        if score < best_score:
                            best_score = score
                            best_edge = (u, v)
                
                if best_edge is None:
                    break
                    
                u, v = best_edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
                # Early check: if min degree reached, we can stop adding?
                # No, we want to maximize density to ensure min degree holds.
                # But if all degrees >= min_deg_needed, we can break.
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= min_deg_needed:
                    # We can stop adding edges to save time?
                    # Actually, adding more edges might break C4-free?
                    # No, we check safety.
                    # But we don't need more edges.
                    break
                    
            # Final check
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_needed:
                # Verify C4-free
                has_c4 = False
                for u in range(n):
                    for v in range(u+1, n):
                        if len(adj[u] & adj[v]) > 1:
                            has_c4 = True
                            break
                    if has_c4:
                        break
                if not has_c4:
                    success = True
                    best_result = {'n': n, 'edges': edges}
                    break
                    
        if success:
            break
            
    return best_result
