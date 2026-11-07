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
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
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
    # Check if adding (u, v) creates a triangle.
    # A triangle is created if there exists w such that (u,w) and (v,w) are in adj.
    # We can optimize by iterating over the smaller neighbor list.
    if len(adj[u]) > len(adj[v]):
        u, v = v, u
    for w in adj[u]:
        if w != v and w in adj[v]:
            return False
    return True

def _verify_solution(added_edges):
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # Check for duplicates with base or within added
    added_set = set()
    for u, v in added_edges:
        if (u, v) in base_edges_set or (v, u) in base_edges_set:
            return False
        if (u, v) in added_set or (v, u) in added_set:
            return False
        added_set.add((u, v))
        added_set.add((v, u))
        adj[u].append(v)
        adj[v].append(u)
        
    # Check triangle-free property
    # For each edge (u,v) in the final graph, check if any common neighbor exists.
    # Actually, simpler: for each u, check pairs in adj[u].
    for u in range(n):
        for i in range(len(adj[u])):
            for j in range(i+1, len(adj[u])):
                if adj[u][i] in adj[u][j]: # This is O(E^2) roughly, but n is small
                     # Wait, adj[u][j] is a neighbor of u. Is it also a neighbor of adj[u][i]?
                     # We need to check if adj[u][i] and adj[u][j] are connected.
                     # Let's use a set for faster lookup.
                     pass
    
    # More efficient triangle check:
    for u in range(n):
        neighbors = set(adj[u])
        for v in adj[u]:
            if v > u: # Check each edge once? No, just check if any common neighbor.
                common = neighbors.intersection(adj[v])
                if common:
                    return False

    # Check connectivity and diameter
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
    Construct a valid initial solution. 
    We use a greedy approach to add edges until diameter <= 4, 
    but we prioritize edges that reduce the maximum distance significantly 
    and are triangle-free.
    """
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    added_edges = []
    dist = _bfs_distances(adj, n)
    current_diam = _get_diameter(dist)
    
    max_iters = 100
    iter_count = 0
    
    while current_diam > 4 and iter_count < max_iters:
        iter_count += 1
        # Find the pair with max distance
        max_d = -1
        best_u, best_v = -1, -1
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
                    best_u, best_v = i, j
        
        if best_u == -1:
            break
            
        # Try to add (best_u, best_v) if valid
        if (best_u, best_v) not in base_edges_set and _is_triangle_free(adj, best_u, best_v, n):
            adj[best_u].append(best_v)
            adj[best_v].append(best_u)
            base_edges_set.add((best_u, best_v))
            base_edges_set.add((best_v, best_u))
            added_edges.append([best_u, best_v])
            dist = _bfs_distances(adj, n)
            current_diam = _get_diameter(dist)
        else:
            # If direct edge is not possible, find an alternative edge that reduces diameter
            # and is triangle-free.
            found = False
            # Sort candidate edges by potential impact? Or just iterate.
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) in base_edges_set:
                        continue
                    if not _is_triangle_free(adj, i, j, n):
                        continue
                    # Simulate adding
                    adj[i].append(j)
                    adj[j].append(i)
                    temp_dist = _bfs_distances(adj, n)
                    temp_diam = _get_diameter(temp_dist)
                    adj[i].remove(j)
                    adj[j].remove(i)
                    
                    if temp_diam < current_diam:
                        adj[i].append(j)
                        adj[j].append(i)
                        base_edges_set.add((i, j))
                        base_edges_set.add((j, i))
                        added_edges.append([i, j])
                        dist = temp_dist
                        current_diam = temp_diam
                        found = True
                        break
                if found:
                    break
            if not found:
                break

    return added_edges

def _local_search(added_edges, deadline):
    """
    Iterated Local Search to minimize the number of edges.
    1. Try to remove any edge.
    2. If stuck, perform a "shake": swap an edge with a random valid edge to escape local minima.
    """
    n = P['n']
    current_solution = [e[:] for e in added_edges]
    
    if not _verify_solution(current_solution):
        return current_solution
        
    improved = True
    no_improve_count = 0
    max_no_improve = 20 # Shake after this many failed attempts
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove any edge
        for i in range(len(current_solution)):
            if time.time() > deadline:
                return current_solution
            test_solution = current_solution[:i] + current_solution[i+1:]
            if _verify_solution(test_solution):
                current_solution = test_solution
                improved = True
                no_improve_count = 0
                break # Restart loop as indices changed
                
        if improved:
            continue
            
        # 2. If no deletion worked, try a swap (shake)
        no_improve_count += 1
        if no_improve_count >= max_no_improve:
            no_improve_count = 0
            # Shake: Pick a random edge to remove and a random valid edge to add
            if len(current_solution) > 0:
                # Pick random edge to remove
                idx_remove = random.randint(0, len(current_solution) - 1)
                e_remove = current_solution[idx_remove]
                
                # Find all possible edges that can be added (triangle-free, not in graph)
                # Build current graph state
                adj = [[] for _ in range(n)]
                base_edges_set = set()
                for u, v in P['edges']:
                    adj[u].append(v)
                    adj[v].append(u)
                    base_edges_set.add((u, v))
                    base_edges_set.add((v, u))
                for u, v in current_solution:
                    adj[u].append(v)
                    adj[v].append(u)
                    base_edges_set.add((u, v))
                    base_edges_set.add((v, u))
                
                # Remove the edge to be swapped out from the temporary check set?
                # Actually, we are swapping: remove e_remove, add e_add.
                # e_add must be triangle-free in the graph WITHOUT e_remove?
                # Or triangle-free in the current graph?
                # If we add e_add to current graph, it must be triangle-free.
                # But we are also removing e_remove.
                # Usually, we check if (current - e_remove + e_add) is valid.
                
                # Let's just pick a random candidate edge from all possible non-base pairs
                candidates = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (u, v) in base_edges_set:
                            continue
                        # Check triangle-free in current graph (conservative)
                        if _is_triangle_free(adj, u, v, n):
                            candidates.append([u, v])
                
                if candidates:
                    e_add = random.choice(candidates)
                    # Construct new solution
                    new_solution = [e[:] for e in current_solution if e != e_remove]
                    # Ensure e_add is not already in new_solution (it shouldn't be if not in base and we checked candidates from base_set which includes current)
                    # Wait, base_edges_set includes current_solution edges. So e_add is not in current_solution.
                    new_solution.append(e_add)
                    
                    if _verify_solution(new_solution):
                        current_solution = new_solution
                        # We didn't reduce count, but we changed the landscape.
                        # We don't set improved=True because count didn't drop, 
                        # but we reset no_improve_count to allow more shakes if needed?
                        # Actually, if we accept a same-size swap, we are still in the same "count" layer.
                        # We should probably only shake if we've been stuck for a while.
                        # Let's accept the shake and continue.
        
    return current_solution

def construct():
    start_time = time.time()
    deadline = start_time + 80 # Safety margin for 90s limit
    
    # 1. Generate initial solution
    added = _construct_initial()
    
    if not _verify_solution(added):
        # Fallback: try a few random restarts of initial construction?
        # Or just return empty if it fails (invalid). 
        # But _construct_initial is quite robust.
        # If it fails, we might need a different initial heuristic.
        # Let's try one more time with a different tie-breaking or randomization if needed.
        # For now, assume it works.
        pass

    # 2. Local Search to minimize edges
    optimized = _local_search(added, deadline)
    
    # 3. Final check
    if _verify_solution(optimized):
        res = []
        for u, v in optimized:
            res.append([int(u), int(v)])
        return res
    else:
        if _verify_solution(added):
            res = []
            for u, v in added:
                res.append([int(u), int(v)])
            return res
        else:
            return []

# EVOLVE-BLOCK-END
