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
        
        # Time-aware restarts
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        # Each restart takes roughly (target_n * min_deg_needed) / 1000 seconds
        est_restart_time = (target_n * max(min_deg_needed, 1)) / 2000.0
        max_restarts = max(1, int(remaining / est_restart_time))
        max_restarts = min(max_restarts, 200)  # Cap at 200
        
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
                # Save neighbors before modification
                N_u = list(N[u])
                N_v = list(N[v])
                
                N[u].add(v)
                N[v].add(u)
                deg[u] += 1
                deg[v] += 1
                
                # Update common:
                # For each w in N_v (old neighbors of v, not including u):
                #   common[u][w] += 1 because v is now a common neighbor of u and w
                # For each w in N_u (old neighbors of u, not including v):
                #   common[v][w] += 1 because u is now a common neighbor of v and w
                
                for w in N_v:
                    if w != u:
                        common[u][w] += 1
                        common[w][u] += 1
                        
                for w in N_u:
                    if w != v:
                        common[v][w] += 1
                        common[w][v] += 1
                        
            def remove_edge(u, v):
                # Save neighbors before modification
                N_u = list(N[u])
                N_v = list(N[v])
                
                N[u].remove(v)
                N[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
                
                # Reverse the common updates
                for w in N_v:
                    if w != u:
                        common[u][w] -= 1
                        common[w][u] -= 1
                        
                for w in N_u:
                    if w != v:
                        common[v][w] -= 1
                        common[w][v] -= 1

            success = True
            removals = 0
            max_removals = 50  # Allow more removals to escape local minima
            
            # Use time-aware step limit
            elapsed = time.time() - start_time
            remaining = deadline - elapsed
            # Estimate steps per second based on n
            steps_per_sec = 5000 / target_n  # Rough estimate
            max_steps = max(500, int(remaining * steps_per_sec * 0.8))
            
            for step in range(max_steps):
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
                    
                # Pick the most constrained low-degree vertex (fewest valid candidates)
                best_u = None
                best_candidate_count = float('inf')
                for candidate_u in low_vertices:
                    count = 0
                    for v in range(target_n):
                        if v == candidate_u:
                            continue
                        if can_add(candidate_u, v):
                            count += 1
                            if count >= best_candidate_count:
                                break
                    if count < best_candidate_count:
                        best_candidate_count = count
                        best_u = candidate_u
                        if count == 0:
                            break
                u = best_u if best_u is not None else random.choice(low_vertices)
                
                # Find valid candidates, preferring those with lower "constraint"
                candidates = []
                for v in range(target_n):
                    if v == u:
                        continue
                    if can_add(u, v):
                        # Score candidate: prefer v where common[u][v] is low
                        # and where v has room (deg[v] < min_deg_needed or not too high)
                        score = common[u][v] * 2 + (0 if deg[v] < min_deg_needed else 1)
                        candidates.append((score, v))
                        
                if not candidates:
                    # Stuck. Remove an edge that is "least critical"
                    # Prefer edges where both endpoints have degree > min_deg_needed
                    if removals < max_removals:
                        # Find edges where both endpoints are "safe" to lose
                        safe_edges = []
                        for i in range(target_n):
                            for j in N[i]:
                                if i < j:
                                    # Only consider if both have degree > min_deg_needed
                                    if deg[i] > min_deg_needed and deg[j] > min_deg_needed:
                                        safe_edges.append((i, j))
                        
                        if safe_edges:
                            # Pick one at random from safe edges
                            e_u, e_v = random.choice(safe_edges)
                            remove_edge(e_u, e_v)
                        else:
                            # No safe edges, pick any edge
                            all_edges = []
                            for i in range(target_n):
                                for j in N[i]:
                                    if i < j:
                                        all_edges.append((i, j))
                            if all_edges:
                                e_u, e_v = random.choice(all_edges)
                                remove_edge(e_u, e_v)
                            else:
                                success = False
                                break
                        removals += 1
                        continue
                    else:
                        success = False
                        break
                            
                # Pick the best candidate (lowest score)
                candidates.sort(key=lambda x: x[0])
                v = candidates[0][1]
                add_edge(u, v)
                
            if success:
                # Robust final verification: recompute common from scratch
                common_verify = [[0] * target_n for _ in range(target_n)]
                for i in range(target_n):
                    for j in N[i]:
                        common_verify[i][j] += 1
                        common_verify[j][i] += 1
                # Actually, common[i][j] should be |N(i) ∩ N(j)|
                # Let me recompute properly
                # Verify C4-free: for all pairs (u,v), |N(u) ∩ N(v)| <= 1
                c4_free = True
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        # Count common neighbors
                        common_count = 0
                        for w in N[u]:
                            if w in N[v]:
                                common_count += 1
                                if common_count > 1:
                                    break
                        if common_count > 1:
                            c4_free = False
                            break
                    if not c4_free:
                        break
                
                c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if common_verify[i][j] > 1:
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
