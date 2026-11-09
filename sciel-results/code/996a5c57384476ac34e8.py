import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_n = 0
    best_edges = []

    # Try n=47 first, then 46, then 45...
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        if min_deg_needed < 0:
            min_deg_needed = 0
            
        found = False
        
        # Number of restarts depends on time available
        # We will increase restarts if the algorithm is faster.
        max_restarts = 100
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 100 + target_n)
            
            # Use list of sets for neighbors for faster iteration in can_add
            N = [set() for _ in range(target_n)]
            deg = [0] * target_n
            
            # common[u][v] = |N(u) cap N(v)|
            # Using a 2D list of ints is fine for n=47
            common = [[0] * target_n for _ in range(target_n)]
            
            def can_add(u, v):
                # 1. u and v must not be connected
                if v in N[u]:
                    return False
                # 2. common[u][v] must be <= 1
                if common[u][v] > 1:
                    return False
                # 3. For all w in N(v), common[u][w] must be 0
                for w in N[v]:
                    if common[u][w] > 0:
                        return False
                # 4. For all w in N(u), common[v][w] must be 0
                for w in N[u]:
                    if common[v][w] > 0:
                        return False
                return True

            def add_edge(u, v):
                N[u].add(v)
                N[v].add(u)
                deg[u] += 1
                deg[v] += 1
                
                # Update common
                # common[x][y] increases if x in N(u) and y in N(v)? No.
                # Adding (u,v):
                # v becomes a common neighbor of u and any w in N(v).
                # So common[u][w] += 1 for w in N(v).
                # u becomes a common neighbor of v and any w in N(u).
                # So common[v][w] += 1 for w in N(u).
                
                for w in N[v]:
                    if w != u:
                        common[u][w] += 1
                        common[w][u] += 1
                        
                for w in N[u]:
                    if w != v:
                        common[v][w] += 1
                        common[w][v] += 1
                        
            def remove_edge(u, v):
                N[u].remove(v)
                N[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
                
                for w in N[v]:
                    if w != u:
                        common[u][w] -= 1
                        common[w][u] -= 1
                        
                for w in N[u]:
                    if w != v:
                        common[v][w] -= 1
                        common[w][v] -= 1

            success = True
            removals = 0
            max_removals = 20 # Limit backtracking/removals
            
            # Improved initialization: Start with a random subset of edges to get closer to min_deg_needed
            # This helps avoid getting stuck in local minima early on.
            # For n=47, min_deg=8. We can try to add ~8*n/2 edges randomly first, then repair.
            # However, random edges often create C4s. 
            # Let's stick to the greedy addition but maybe start with a few forced edges?
            # The current greedy approach is decent but slow.
            
            # Alternative: Use a "dense" start for low n, sparse for high n?
            # Let's try to speed up the search by being more aggressive with edge addition.
            
            for _ in range(2000): # Max steps
                if time.time() > deadline:
                    success = False
                    break
                    
                # Check if we are done
                if min(deg) >= min_deg_needed:
                    success = True
                    break

                low_vertices = [i for i in range(target_n) if deg[i] < min_deg_needed]
                if not low_vertices:
                    success = True
                    break
                    
                u = random.choice(low_vertices)
                
                # Find a valid v to connect to u
                candidates = []
                for v in range(target_n):
                    if v == u:
                        continue
                    if can_add(u, v):
                        candidates.append(v)
                        
                if not candidates:
                    # Stuck. Try to remove a random edge to unblock?
                    # Prioritize removing edges that are less critical (e.g., from high degree vertices)
                    # or just random.
                    if removals < max_removals:
                        # Pick a random edge
                        all_edges = []
                        for i in range(target_n):
                            for j in N[i]:
                                if i < j:
                                    all_edges.append((i, j))
                        if all_edges:
                            e_u, e_v = random.choice(all_edges)
                            remove_edge(e_u, e_v)
                            removals += 1
                            continue
                    else:
                        success = False
                        break
                            
                v = random.choice(candidates)
                add_edge(u, v)
                
            if success:
                # Final verification
                # Recalculate common to be sure
                # Check C4-free: for all u,v, common[u][v] <= 1
                c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if common[i][j] > 1:
                            c4_free = False
                            break
                    if not c4_free:
                        break
                
                if c4_free and min(deg) >= min_deg_needed:
                    final_edges = []
                    for i in range(target_n):
                        for j in N[i]:
                            if i < j:
                                final_edges.append([i, j])
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break # Break restart loop
        
        if best_n == 47:
            break
        if found:
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: n=39, empty graph is valid (min deg 0 >= 0)
        return {'n': 39, 'edges': []}
