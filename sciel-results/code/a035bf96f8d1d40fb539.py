"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import math
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_dist(adj, n):
    """Compute all-pairs shortest paths. Returns list of lists of distances."""
    dist = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
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

def _is_triangle_free_with_base(adj, u, v, n):
    """Check if adding edge (u,v) to adj keeps it triangle-free."""
    # Check against existing neighbors
    # A triangle is formed if there is a w such that (u,w) in adj and (v,w) in adj
    # Note: adj contains base + previously added edges in the search tree
    neighbors_u = set(adj[u])
    neighbors_v = set(adj[v])
    # If they share a common neighbor, it's a triangle
    if neighbors_u.intersection(neighbors_v):
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
        
    seen_added = set()
    for u, v in added_edges:
        if (u, v) in base_edges_set or (v, u) in base_edges_set:
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
    dist = _bfs_dist(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return False
                
    diam = _get_diameter(dist)
    if diam > 4:
        return False
        
    return True

def _precompute_candidates():
    """
    Precompute all edges (u,v) that are:
    1. Not in base path.
    2. Do not form a triangle with the base path alone.
    This is a necessary condition for any valid solution, so we can restrict search to these.
    """
    n = P['n']
    base_adj = [[] for _ in range(n)]
    base_edges_set = set()
    for u, v in P['edges']:
        base_adj[u].append(v)
        base_adj[v].append(u)
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges_set:
                continue
            # Check triangle-free with base
            # If u and v share a neighbor in base, adding (u,v) creates a triangle
            # In a path, u and v share a neighbor only if they are distance 2 apart.
            # e.g., (0,2) shares 1. (1,3) shares 2.
            # Also, if they are distance 1, it's a base edge (already skipped).
            # If distance > 2, no common neighbor in path.
            # So basically, we can't add edges of length 2.
            # Wait, what if we add (0,2) and (1,3)? (0,2) and (1,3) don't form a triangle with each other directly?
            # But (0,2) forms a triangle with base edge (0,1) and (1,2). So (0,2) is forbidden.
            # So any edge with |u-v| == 2 is forbidden.
            # Any edge with |u-v| == 1 is base.
            # So candidates are |u-v| >= 3.
            # Let's verify triangle-free condition properly.
            common = set(base_adj[u]).intersection(base_adj[v])
            if not common:
                candidates.append((u, v))
                
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin for 60s limit
    
    n = P['n']
    candidates = _precompute_candidates()
    
    # Sort candidates by "potential" - maybe distance in path? 
    # Edges that jump further are generally more useful for reducing diameter.
    # But we need to be careful.
    
    # Initial state
    base_adj = [[] for _ in range(n)]
    base_edges_set = set()
    for u, v in P['edges']:
        base_adj[u].append(v)
        base_adj[v].append(u)
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # We will use a recursive search (Branch and Bound)
    # State: current adjacency list, set of added edges, current diameter
    # We want to find a set of edges S such that diam(G + S) <= 4 and |S| is minimized.
    
    best_solution = None
    best_size = float('inf')
    
    # Lower bound heuristic:
    # If current diameter is D, we need to reduce it to 4.
    # Each edge can reduce the diameter by at most... well, it's complex.
    # A simpler bound: If there are K pairs at distance > 4, and each edge can cover at most M such pairs,
    # we need at least ceil(K/M) more edges.
    # However, computing which pairs an edge covers is expensive.
    
    # Alternative: Iterative Deepening Depth First Search (IDDFS) on the number of edges.
    # Try k=1, k=2, ... until a solution is found.
    # For each k, use DFS to find a subset of size k that works.
    # This guarantees optimality (minimum k).
    
    # To make DFS efficient, we need good pruning.
    # 1. Triangle-free check.
    # 2. Diameter check (if current diam > 4 and we have no more edges to add, fail).
    #    Actually, if we are at depth d and have added k-d edges, we can check if it's POSSIBLE
    #    to reach diameter 4 with d more edges.
    #    Heuristic: If the current diameter is D, and the maximum possible reduction per edge is R,
    #    then if D - d*R > 4, we can prune.
    #    What is R? In a path of 24, adding an edge (u,v) with |u-v|=L reduces the distance between u and v from L to 1.
    #    It also reduces distances for nodes in between.
    #    Max reduction in diameter: If we connect the two ends (0, 23), diameter drops from 23 to something small.
    #    Let's just use a simple check: if current_diam > 4 and edges_remaining == 0, fail.
    
    # Let's use IDDFS.
    
    # Precompute all-pairs dist for base graph to help with heuristics?
    base_dist = _bfs_dist(base_adj, n)
    
    def dfs(current_adj, added_edges, depth, max_depth, current_diam):
        nonlocal best_solution, best_size, deadline
        
        if time.time() > deadline:
            return False # Timeout signal
            
        if len(added_edges) > best_size:
            return False # Prune: already worse than best found
            
        if current_diam <= 4:
            # Found a valid solution
            best_solution = [e[:] for e in added_edges]
            best_size = len(added_edges)
            return True
            
        if depth == max_depth:
            # No more edges allowed, but diam > 4
            return False
            
        # Heuristic Pruning:
        # If current_diam is large, we might need more edges.
        # A very rough lower bound: 
        # The number of pairs with distance > 4 is N_pairs.
        # Each new edge can reduce the distance of at most ... 
        # This is hard to estimate quickly. Let's rely on the depth limit.
        
        # Generate candidates for the next edge
        # We should prioritize edges that reduce the diameter the most.
        # To avoid exponential blowup, we can limit the branching factor or order candidates.
        
        # Get current distances to find the "most critical" pairs?
        # Actually, just iterating through all candidates might be too slow if we branch too wide.
        # Total candidates ~ 200.
        # If we try all 200 at each level, and depth is 10, that's 200^10. Impossible.
        
        # We need a smarter search.
        # Let's try a greedy construction first to get a good upper bound (best_size).
        # Then use the IDDFS to prove optimality or find better.
        
        # For the DFS itself, let's restrict candidates to those that are "useful".
        # An edge is useful if it reduces the current diameter.
        
        # Compute current distances
        dist = _bfs_dist(current_adj, n)
        
        # Find pairs with max distance
        max_d = current_diam
        if max_d <= 4:
             return True # Should have been caught above, but safe
        
        # Identify edges that reduce the diameter
        # If we add (u,v), what is the new diameter?
        # Calculating new diameter for every candidate is expensive (BFS for each).
        # But n=24 is small. 200 candidates * 24 BFS * 24 nodes = ~115k ops. Fast.
        
        useful_candidates = []
        for u, v in candidates:
            # Check if (u,v) is already added
            if (u, v) in added_edges_set or (v, u) in added_edges_set:
                continue
            # Check triangle-free
            if not _is_triangle_free_with_base(current_adj, u, v, n):
                continue
            
            # Test adding (u,v)
            current_adj[u].append(v)
            current_adj[v].append(u)
            
            new_dist = _bfs_dist(current_adj, n)
            new_diam = _get_diameter(new_dist)
            
            # Revert
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
            if new_diam < current_diam:
                useful_candidates.append((u, v, new_diam))
                
        # Sort by new_diam (best reduction first)
        useful_candidates.sort(key=lambda x: x[2])
        
        # Limit branching to top K candidates to keep search tractable?
        # If we limit, we might miss the optimal solution.
        # But if we don't, it's too slow.
        # Let's try top 10 or 20.
        # Given the structure of the path, likely only a few edges matter.
        
        top_k = 10
        for u, v, new_diam in useful_candidates[:top_k]:
            # Add edge
            current_adj[u].append(v)
            current_adj[v].append(u)
            added_edges.append([u, v])
            added_edges_set.add((u, v))
            added_edges_set.add((v, u))
            
            # Recurse
            if dfs(current_adj, added_edges, depth + 1, max_depth, new_diam):
                return True
                
            # Backtrack
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            added_edges.pop()
            added_edges_set.remove((u, v))
            added_edges_set.remove((v, u))
            
        return False

    # Helper to manage added_edges_set in DFS
    # We need to pass it or make it global/nonlocal
    # Let's refactor DFS to take added_edges_set as argument or use a global set
    # Since Python closures are tricky with mutable state in recursion for backtracking,
    # let's use a global set for the current path's edges.
    
    current_added_set = set()
    
    def dfs_v2(current_adj, added_edges, depth, max_depth, current_diam):
        nonlocal best_solution, best_size, deadline, current_added_set
        
        if time.time() > deadline:
            return False
            
        if len(added_edges) > best_size:
            return False
            
        if current_diam <= 4:
            best_solution = [e[:] for e in added_edges]
            best_size = len(added_edges)
            return True
            
        if depth == max_depth:
            return False
            
        # Heuristic: If current_diam is high, we might be stuck.
        
        dist = _bfs_dist(current_adj, n)
        
        useful_candidates = []
        for u, v in candidates:
            if (u, v) in current_added_set or (v, u) in current_added_set:
                continue
            if not _is_triangle_free_with_base(current_adj, u, v, n):
                continue
            
            current_adj[u].append(v)
            current_adj[v].append(u)
            
            new_dist = _bfs_dist(current_adj, n)
            new_diam = _get_diameter(new_dist)
            
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
            if new_diam < current_diam:
                useful_candidates.append((u, v, new_diam))
                
        useful_candidates.sort(key=lambda x: x[2])
        
        # Branching factor control
        # If we are deep, maybe try more? If shallow, try fewer?
        # Let's try top 15.
        top_k = 15
        for u, v, new_diam in useful_candidates[:top_k]:
            current_adj[u].append(v)
            current_adj[v].append(u)
            added_edges.append([u, v])
            current_added_set.add((u, v))
            current_added_set.add((v, u))
            
            if dfs_v2(current_adj, added_edges, depth + 1, max_depth, new_diam):
                return True
                
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            added_edges.pop()
            current_added_set.remove((u, v))
            current_added_set.remove((v, u))
            
        return False

    # First, get a good initial solution using Greedy to set best_size
    # This helps prune the IDDFS significantly.
    
    def greedy_construct():
        adj = [list(x) for x in base_adj]
        added = []
        added_set = set()
        dist = _bfs_dist(adj, n)
        diam = _get_diameter(dist)
        
        while diam > 4:
            # Find best edge
            best_u, best_v = -1, -1
            best_new_diam = diam + 1
            
            for u, v in candidates:
                if (u, v) in added_set or (v, u) in added_set:
                    continue
                if not _is_triangle_free_with_base(adj, u, v, n):
                    continue
                
                adj[u].append(v)
                adj[v].append(u)
                new_dist = _bfs_dist(adj, n)
                new_diam = _get_diameter(new_dist)
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_u, best_v = u, v
                    
            if best_u == -1:
                return None # Stuck
                
            # Add it
            adj[best_u].append(best_v)
            adj[best_v].append(best_u)
            added.append([best_u, best_v])
            added_set.add((best_u, best_v))
            added_set.add((best_v, best_u))
            
            dist = _bfs_dist(adj, n)
            diam = _get_diameter(dist)
            
        return added

    greedy_sol = greedy_construct()
    if greedy_sol:
        best_solution = greedy_sol
        best_size = len(greedy_sol)
    
    # Now run IDDFS to find optimal
    # Start from k=1
    # If best_size is 10, we check 1..9
    # If we find one, it's optimal.
    
    # To speed up, we can start from max(1, best_size - 5) if best_size is large?
    # No, IDDFS must start from 1 to prove optimality.
    # But if we have a good upper bound, we can prune branches that exceed it.
    
    # Optimization: If best_size is small (e.g., <= 5), IDDFS is fast.
    # If best_size is large (e.g., 10), IDDFS might be slow.
    # We can try a "Binary Search" on the number of edges? No, validity is not monotonic in a simple way for "exists solution of size k".
    # Actually, if there is a solution of size k, there is a solution of size k+1 (just add a redundant edge? No, must be triangle-free).
    # Adding an edge might break triangle-free. So monotonicity is not guaranteed.
    # So we must check k=1, 2, ...
    
    # However, we can limit the depth of the DFS at each k.
    # And we can use the greedy solution to prune.
    
    max_k_to_check = best_size - 1
    if max_k_to_check < 1:
        max_k_to_check = 1 # If greedy found 1, it's optimal.
        
    # Actually, if greedy found 10, we check 1 to 9.
    # If we find a solution of size 8, we update best_size to 8 and stop?
    # No, we should check if 7, 6... are possible.
    # So we iterate k from 1 to best_size-1.
    # If we find a solution for k, we set best_size = k and break?
    # No, we want the MINIMUM. So if we find one for k, that's the minimum?
    # Yes, because we are iterating k in increasing order.
    # So the first k for which we find a solution is the optimal one.
    
    # Wait, the DFS finds *a* solution. If it finds one for k, then k is the answer.
    # We don't need to check smaller k's if we already checked them and failed.
    
    # So:
    for k in range(1, best_size):
        if time.time() > deadline:
            break
            
        # Reset state for this k
        current_adj = [list(x) for x in base_adj]
        added_edges = []
        current_added_set = set()
        
        # Run DFS with max_depth = k
        # Note: dfs_v2 uses nonlocal best_size for pruning.
        # We want to find ANY solution of size k.
        # If found, best_size becomes k and we return.
        
        # To make DFS faster, we can pass a specific target?
        # The current dfs_v2 stops if it finds ANY solution (diam <= 4).
        # It prunes if len(added) > best_size.
        # Since we are iterating k, and best_size is the current upper bound,
        # if we are checking k < best_size, the pruning len(added) > best_size is not very active.
        # We should prune if len(added) > k.
        
        # Let's modify dfs_v2 to take a max_allowed_size.
        
        def dfs_opt(current_adj, added_edges, depth, max_depth, current_diam, max_allowed_size):
            nonlocal deadline, current_added_set
            
            if time.time() > deadline:
                return False
                
            if len(added_edges) > max_allowed_size:
                return False
                
            if current_diam <= 4:
                return True # Found valid solution of size <= max_allowed_size
                
            if depth == max_depth:
                return False
                
            dist = _bfs_dist(current_adj, n)
            
            useful_candidates = []
            for u, v in candidates:
                if (u, v) in current_added_set or (v, u) in current_added_set:
                    continue
                if not _is_triangle_free_with_base(current_adj, u, v, n):
                    continue
                
                current_adj[u].append(v)
                current_adj[v].append(u)
                
                new_dist = _bfs_dist(current_adj, n)
                new_diam = _get_diameter(new_dist)
                
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                
                if new_diam < current_diam:
                    useful_candidates.append((u, v, new_diam))
                    
            useful_candidates.sort(key=lambda x: x[2])
            
            top_k = 15 # Branching factor
            for u, v, new_diam in useful_candidates[:top_k]:
                current_adj[u].append(v)
                current_adj[v].append(u)
                added_edges.append([u, v])
                current_added_set.add((u, v))
                current_added_set.add((v, u))
                
                if dfs_opt(current_adj, added_edges, depth + 1, max_depth, new_diam, max_allowed_size):
                    return True
                    
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                added_edges.pop()
                current_added_set.remove((u, v))
                current_added_set.remove((v, u))
                
            return False

        # Run the search for size k
        # We need to reset current_added_set
        current_added_set = set()
        
        found = dfs_opt(current_adj, added_edges, 0, k, _get_diameter(_bfs_dist(current_adj, n)), k)
        
        if found:
            # We found a solution of size k
            # We need to extract it from added_edges
            # But added_edges was modified in place.
            # We should copy it.
            best_solution = [e[:] for e in added_edges]
            best_size = k
            break # Since we iterate k from 1, this is the minimum
            
    # If no better solution found, best_solution is the greedy one.
    # If greedy failed, best_solution is None.
    # But greedy should work.
    
    if best_solution is None:
        # Fallback
        return []
        
    # Final verification
    if _verify_solution(best_solution):
        res = []
        for u, v in best_solution:
            res.append([int(u), int(v)])
        return res
    else:
        return []

# EVOLVE-BLOCK-END
