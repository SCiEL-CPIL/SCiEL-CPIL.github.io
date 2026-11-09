import numpy as np
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 120s kill

    q = 5
    # Enumerate all points of PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Generate symmetric 3x3 matrices over GF(5) that are non-degenerate
    # We will try a subset of forms to find the best core
    forms = []
    # Diagonal forms
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(1, 5):
                forms.append((a, b, c, 0, 0, 0))
    # Non-diagonal symmetric forms: M = [[a, d, e], [d, b, f], [e, f, c]]
    # To keep it fast, we'll try a few specific non-diagonal ones
    # But diagonal ones are known to work well for polarity graphs.
    # Let's stick to diagonal and maybe a couple of off-diagonal.
    
    random.seed(42)
    random.shuffle(forms)
    
    best_result = None
    best_total_c4 = float('inf')
    
    # Function to calculate C4s using a specific edge (u, v) in a given adjacency list
    def count_c4_using_edge(adj, u, v):
        # Count pairs (x, y) such that x in N(u)\{v}, y in N(v)\{u}, and (x, y) in E
        count = 0
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[v]:
                if y == u:
                    continue
                if y in adj[x]:
                    count += 1
        return count

    # Function to calculate total C4s
    def count_total_c4(adj):
        total = 0
        for u in range(31):
            for v in range(u+1, 31):
                # Number of common neighbors
                common = len(adj[u] & adj[v])
                total += common * (common - 1) // 2
        return total // 4

    for form in forms:
        if time.time() > deadline - 60:
            break
            
        a, b, c, d, e, f = form
        # Build adjacency matrix using the bilinear form
        # B(p,q) = a*x1*x2 + d*x1*y2 + e*x1*z2 + d*y1*x2 + b*y1*y2 + f*y1*z2 + e*z1*x2 + f*z1*y2 + c*z1*z2
        # Since M is symmetric, M[0][1]=M[1][0]=d, etc.
        
        edges_core = []
        for i in range(31):
            for j in range(i+1, 31):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (a*x1*x2 + d*x1*y2 + e*x1*z2 + 
                       d*y1*x2 + b*y1*y2 + f*y1*z2 + 
                       e*z1*x2 + f*z1*y2 + c*z1*z2) % 5
                if val == 0:
                    edges_core.append([i, j])
        
        if len(edges_core) != 90:
            continue
        
        # This is a valid C4-free core with 90 edges
        adj = [set() for _ in range(31)]
        for u, v in edges_core:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find best edge to add (minimizing new C4s)
        best_edge = None
        best_add = float('inf')
        
        for u in range(31):
            for v in range(u+1, 31):
                if v in adj[u]:
                    continue
                # Count C4s created by adding (u, v)
                # This is count_c4_using_edge(adj, u, v) but note that (u,v) is not in adj yet.
                # The formula counts paths u-x-y-v.
                add = 0
                for x in adj[u]:
                    for y in adj[v]:
                        if y in adj[x]:
                            add += 1
                if add < best_add:
                    best_add = add
                    best_edge = [u, v]
        
        if best_edge is None:
            continue
        
        # Add the best edge
        final_edges = edges_core + [best_edge]
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        
        current_c4 = count_total_c4(adj)
        
        # Local Search: Steepest Descent Swaps
        # We will try to improve by swapping one edge with one non-edge
        # Max iterations to avoid getting stuck
        max_swaps = 200
        for _ in range(max_swaps):
            if time.time() > deadline - 10:
                break
                
            best_swap = None
            best_delta = 0 # We only accept if delta < 0
            
            # Get list of current edges and non-edges
            edge_set = set(tuple(sorted(e)) for e in final_edges)
            all_pairs = [(i, j) for i in range(31) for j in range(i+1, 31)]
            
            # We will iterate through all possible swaps.
            # To speed up, we can pre-filter or just do it if it's fast enough.
            # 91 edges * 374 non-edges = ~34k swaps.
            # Each swap evaluation is O(deg(u)*deg(v) + deg(a)*deg(b)).
            # Degrees are around 6. So ~100 ops per swap. 34k * 100 = 3.4M ops.
            # This is fast in Python if optimized.
            
            for u, v in final_edges:
                # Remove (u, v) temporarily to evaluate adding (a, b)
                # We don't actually modify adj, we just simulate.
                # But for simplicity and correctness, we can calculate delta directly.
                
                # Delta = C4s_created_by_adding(a,b) - C4s_destroyed_by_removing(u,v)
                # C4s_destroyed_by_removing(u,v) is count_c4_using_edge(adj, u, v)
                destroyed = count_c4_using_edge(adj, u, v)
                
                # We need to find (a, b) that minimizes (created - destroyed)
                # i.e., minimizes created, since destroyed is fixed for this (u,v).
                # Wait, destroyed depends on (u,v).
                # So for each (u,v), we find the best (a,b) to add.
                
                best_add_edge = None
                min_created = float('inf')
                
                # Iterate through non-edges
                for a, b in all_pairs:
                    if (a, b) in edge_set or (b, a) in edge_set:
                        continue
                    # Check if (a, b) is the same as (u, v)
                    if {a, b} == {u, v}:
                        continue
                    
                    # Calculate C4s created by adding (a, b) to G \ {(u, v)}
                    # Let G' = G \ {(u, v)}.
                    # We need to count paths a-x-y-b in G'.
                    # x is in N_G'(a), y is in N_G'(b), and (x, y) in E(G').
                    
                    # N_G'(a) = N_G(a) \ {b} \ ({u} if a==u or a==v?)
                    # Actually, if a is u or v, its neighborhood changes.
                    
                    created = 0
                    for x in adj[a]:
                        # x must be in N_G'(a)
                        if x == b: continue
                        # If removing (u,v) affects N(a)?
                        # Only if a is u or v, or if x is u or v and (a, x) was (u, v) or (v, u).
                        # Since (u, v) is the only edge removed, (a, x) is removed only if {a, x} = {u, v}.
                        if {a, x} == {u, v}:
                            continue
                        
                        for y in adj[b]:
                            # y must be in N_G'(b)
                            if y == a: continue
                            if {b, y} == {u, v}:
                                continue
                            
                            # (x, y) must be in E(G')
                            # (x, y) is removed only if {x, y} = {u, v}.
                            if {x, y} == {u, v}:
                                continue
                            
                            if y in adj[x]:
                                created += 1
                                
                    if created < min_created:
                        min_created = created
                        best_add_edge = (a, b)
                        
                if best_add_edge is not None:
                    delta = min_created - destroyed
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (u, v, best_add_edge)
                        
            if best_swap is None:
                break
                
            # Apply the best swap
            u, v, (a, b) = best_swap
            # Remove (u, v)
            adj[u].remove(v)
            adj[v].remove(u)
            # Update final_edges
            final_edges.remove([u, v])
            final_edges.remove([v, u]) if [v, u] in final_edges else None
            # Add (a, b)
            adj[a].add(b)
            adj[b].add(a)
            final_edges.append([a, b])
            
            current_c4 += best_delta
            
            if current_c4 < best_total_c4:
                best_total_c4 = current_c4
                best_result = [list(e) for e in final_edges]
                
            # If we reached 0 C4s, we can stop
            if current_c4 == 0:
                break

    if best_result is None:
        # Fallback to greedy if no valid core found (should not happen)
        return _greedy_construct()
        
    return best_result

def _greedy_construct():
    """Fallback greedy construction."""
    random.seed(42)
    
    edges = []
    adj = [set() for _ in range(31)]
    
    remaining = []
    for i in range(31):
        for j in range(i+1, 31):
            remaining.append([i, j])
    
    random.shuffle(remaining)
    
    for _ in range(91):
        best_e = None
        best_add = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            add = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        add += 1
            if add < best_add:
                best_add = add
                best_e = e
            if add == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append(best_e)
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)
        
    while len(edges) < 91 and remaining:
        for e in remaining:
            u, v = e
            if v not in adj[u]:
                edges.append(e)
                adj[u].add(v)
                adj[v].add(u)
                remaining.remove(e)
                break
                
    return edges[:91]
