"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import math

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n):
    dist = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = [i]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
    return dist

def _get_diameter(dist):
    max_d = 0
    for i in range(len(dist)):
        for j in range(len(dist)):
            if dist[i][j] > max_d:
                max_d = dist[i][j]
    return max_d

def _is_triangle_free(adj, u, v, n):
    neighbors_u = set(adj[u])
    neighbors_v = set(adj[v])
    common = neighbors_u.intersection(neighbors_v)
    if common:
        return False
    return True

def _verify_solution(added_edges):
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Check for duplicate edges in added_edges
    seen_added = set()
    for u, v in added_edges:
        if (u, v) in base_edges or (v, u) in base_edges:
            return False
        if (u, v) in seen_added or (v, u) in seen_added:
            return False
        seen_added.add((u, v))
        seen_added.add((v, u))
        
        adj[u].append(v)
        adj[v].append(u)
        
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
                    
    # Check connected and diameter
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return False
                
    diam = _get_diameter(dist)
    if diam > 4:
        return False
        
    return True

def _construct_initial():
    """
    Construct a valid solution using a simple greedy approach.
    We prioritize edges that reduce the diameter the most.
    """
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    added_edges = []
    
    dist = _bfs_distances(adj, n)
    current_diam = _get_diameter(dist)
    
    max_iters = 100
    iter_count = 0
    
    while current_diam > 4 and iter_count < max_iters:
        iter_count += 1
        max_d = -1
        best_u, best_v = -1, -1
        
        # Find the pair with maximum distance
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
                    best_u, best_v = i, j
        
        if best_u == -1:
            break
            
        # Try to add the edge directly if it's valid
        if (best_u, best_v) not in base_edges and (best_v, best_u) not in base_edges:
            if _is_triangle_free(adj, best_u, best_v, n):
                adj[best_u].append(best_v)
                adj[best_v].append(best_u)
                base_edges.add((best_u, best_v))
                base_edges.add((best_v, best_u))
                added_edges.append([best_u, best_v])
                dist = _bfs_distances(adj, n)
                current_diam = _get_diameter(dist)
                continue
        
        # If direct edge is not possible, search for any valid edge that reduces diameter
        found = False
        for i in range(n):
            for j in range(i+1, n):
                if i == best_u and j == best_v:
                    continue
                if (i, j) in base_edges or (j, i) in base_edges:
                    continue
                if _is_triangle_free(adj, i, j, n):
                    # Test adding edge
                    adj[i].append(j)
                    adj[j].append(i)
                    temp_dist = _bfs_distances(adj, n)
                    temp_diam = _get_diameter(temp_dist)
                    adj[i].remove(j)
                    adj[j].remove(i)
                    
                    if temp_diam < current_diam:
                        adj[i].append(j)
                        adj[j].append(i)
                        base_edges.add((i, j))
                        base_edges.add((j, i))
                        added_edges.append([i, j])
                        dist = temp_dist
                        current_diam = temp_diam
                        found = True
                        break
            if found:
                break
                
        if not found:
            # If no edge reduces diameter, we might be stuck or need a different strategy
            # For now, break to avoid infinite loop
            break

    return added_edges

def _local_search(added_edges, deadline):
    """
    Iterated Local Search to minimize the number of edges.
    """
    n = P['n']
    current_solution = [e[:] for e in added_edges]
    
    if not _verify_solution(current_solution):
        return current_solution
        
    best_solution = [e[:] for e in current_solution]
    best_len = len(current_solution)
    
    # Precompute all possible non-base edges
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    all_possible_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_edges:
                all_possible_edges.append((u, v))
                
    # Shuffle for randomization in perturbations
    random.shuffle(all_possible_edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove any edge
        for i in range(len(current_solution)):
            if time.time() > deadline:
                return best_solution
            # Temporarily remove edge i
            test_solution = current_solution[:i] + current_solution[i+1:]
            if _verify_solution(test_solution):
                current_solution = test_solution
                improved = True
                if len(current_solution) < best_len:
                    best_len = len(current_solution)
                    best_solution = [e[:] for e in current_solution]
                break # Restart loop as indices changed
                
        # 2. If no deletion worked, try perturbation (add an edge, then try to delete)
        if not improved:
            # Pick a random edge to add
            if all_possible_edges:
                e_add = random.choice(all_possible_edges)
                u, v = e_add
                # Check if it's triangle-free in current graph
                adj = [[] for _ in range(n)]
                for uu, vv in P['edges']:
                    adj[uu].append(vv)
                    adj[vv].append(uu)
                for uu, vv in current_solution:
                    adj[uu].append(vv)
                    adj[vv].append(uu)
                    
                if _is_triangle_free(adj, u, v, n):
                    # Add it
                    new_solution = current_solution + [[u, v]]
                    # Now try to remove edges from this new solution
                    # We want to remove 2 edges to end up with 1 less? Or just 1 to stay same but different structure?
                    # Actually, standard ILS: Perturb, then Local Search.
                    # Local Search on new_solution might find a better minimum.
                    # Let's run a quick local search on new_solution
                    temp_sol = new_solution
                    temp_improved = True
                    while temp_improved and time.time() < deadline:
                        temp_improved = False
                        for i in range(len(temp_sol)):
                            if time.time() > deadline:
                                break
                            test_sol = temp_sol[:i] + temp_sol[i+1:]
                            if _verify_solution(test_sol):
                                temp_sol = test_sol
                                temp_improved = True
                                break
                        if time.time() > deadline:
                            break
                            
                    # Accept if better or equal (to escape local minima)
                    if len(temp_sol) < len(current_solution):
                        current_solution = temp_sol
                        if len(current_solution) < best_len:
                            best_len = len(current_solution)
                            best_solution = [e[:] for e in current_solution]
                        improved = True
                    elif random.random() < 0.1: # Small probability of accepting equal to diversify
                        current_solution = temp_sol
                        improved = True # Force loop to continue

    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    # 1. Generate initial solution
    added = _construct_initial()
    
    if not _verify_solution(added):
        # Fallback: if greedy fails, try a hardcoded valid set? 
        # The greedy should work. If not, return empty (invalid) or try to fix.
        # For robustness, let's try to add edges until valid if greedy fails.
        n = P['n']
        adj = [[] for _ in range(n)]
        base_edges = set()
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
            base_edges.add((u, v))
            base_edges.add((v, u))
            
        added = []
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) not in base_edges and _is_triangle_free(adj, u, v, n):
                    adj[u].append(v)
                    adj[v].append(u)
                    base_edges.add((u, v))
                    base_edges.add((v, u))
                    added.append([u, v])
        # This might be too many edges, but it's valid.
        # We will rely on local search to minimize.

    # 2. Local Search to minimize edges
    optimized = _local_search(added, deadline)
    
    # 3. Final check
    if _verify_solution(optimized):
        # Ensure types are correct
        res = []
        for u, v in optimized:
            res.append([int(u), int(v)])
        return res
    else:
        # If optimization broke it (shouldn't happen), return original
        if _verify_solution(added):
            res = []
            for u, v in added:
                res.append([int(u), int(v)])
            return res
        else:
            return []

# EVOLVE-BLOCK-END
