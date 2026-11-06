import time
import random
from math import gcd
from collections import deque

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85 # 5s safety margin

    n = limit
    # Nodes are 1..512. Indices 0..511 in arrays.
    # We will work on G' (nodes 2..512) to exploit the structure.
    # Node i in G' corresponds to number i+1 (if we shift) or just keep original indices 1..511 (0..510).
    # Let's use original indices 1..511 for the graph G'.
    
    # Build adjacency list for G' (nodes 2 to 512)
    # adj[v] is a set of neighbors u in G' such that lcm(v, u) <= 512
    adj = [set() for _ in range(n)] # index 0 unused for G', or we can map 2..512 to 0..510
    
    # Let's map number x (2..512) to index x-2 (0..510) for compactness in G'
    # But for clarity, let's just use the number itself as index in a dict or list of size 513.
    # List of size 513 is fine.
    adj_full = [set() for _ in range(n + 1)]
    
    for u in range(2, n + 1):
        for v in range(u + 1, n + 1):
            g = gcd(u, v)
            lcm_val = (u * v) // g
            if lcm_val <= limit:
                adj_full[u].add(v)
                adj_full[v].add(u)
                
    # Find connected components in G' (nodes 2..512)
    visited = [False] * (n + 1)
    components = []
    
    for u in range(2, n + 1):
        if not visited[u]:
            # BFS
            comp = []
            queue = deque([u])
            visited[u] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for neighbor in adj_full[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
            
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_clique_size = 0
    best_clique_nodes = []
    
    # We need to find the max clique in each component.
    # Since components are independent, the max clique of G' is the max of the max cliques of its components.
    
    for comp in components:
        if time.time() > deadline:
            break
            
        # If the component is small, we can do exact search.
        # If it's large, we might need heuristics or a faster exact solver.
        # Let's try Bron-Kerbosch with pivot for all, but with a time check.
        
        # Map component nodes to 0..k-1 for the solver
        comp_nodes = comp
        k = len(comp_nodes)
        if k == 0:
            continue
            
        # Build local adjacency matrix/list
        # local_adj[i] is set of indices j in 0..k-1
        index_map = {node: i for i, node in enumerate(comp_nodes)}
        local_adj = [set() for _ in range(k)]
        
        for i, u in enumerate(comp_nodes):
            for v in adj_full[u]:
                if v in index_map:
                    local_adj[i].add(index_map[v])
                    
        # Run Max Clique on local_adj
        # Using a simple recursive Bron-Kerbosch with pivot
        # To avoid deep recursion limits or overhead, we use a non-recursive or optimized recursive approach.
        # Python recursion limit is usually 1000. K might be up to ~300.
        # We should be careful.
        
        def max_clique_bronekerbosch(adj_list):
            # Returns the size and nodes of the max clique
            n_local = len(adj_list)
            if n_local == 0:
                return 0, []
            
            best_size = 0
            best_set = []
            
            # Convert to list of lists for faster access? Sets are fine.
            # Pivot selection: choose v in P with max degree in P
            def expand(R, P, X):
                nonlocal best_size, best_set
                
                if not P and not X:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_set = list(R)
                    return
                
                if not P:
                    return
                    
                # Pivot selection
                # Choose v in P U X with max degree in P
                # For simplicity, pick first element of P if X is empty, else max degree in P U X
                # Standard: pick v in P U X
                candidates = P.union(X)
                if not candidates:
                    return
                    
                # Find v with max degree in P
                max_deg = -1
                pivot = None
                for v in candidates:
                    deg = len(adj_list[v].intersection(P))
                    if deg > max_deg:
                        max_deg = deg
                        pivot = v
                        
                # Recurse on P \ N(pivot)
                # If pivot is in P, P \ N(pivot) excludes pivot's neighbors.
                # If pivot is in X, P \ N(pivot) excludes pivot's neighbors.
                
                # Optimization: If P is small, iterate directly.
                # We iterate over nodes in P that are NOT neighbors of pivot.
                # If pivot is in P, we also skip pivot itself?
                # Standard Bron-Kerbosch:
                # for v in P \ N(u)
                
                nodes_to_process = set()
                for v in P:
                    if v != pivot and v not in adj_list[pivot]:
                        nodes_to_process.add(v)
                
                # If pivot is not in P (i.e., in X), then P \ N(pivot) is just P minus neighbors.
                # If pivot is in P, we remove pivot and its neighbors.
                # The logic "v != pivot and v not in adj_list[pivot]" works for both if we consider adj_list[pivot] doesn't contain pivot.
                
                for v in nodes_to_process:
                    if time.time() > deadline:
                        return
                    
                    new_R = R | {v}
                    new_P = P.intersection(adj_list[v])
                    new_X = X.intersection(adj_list[v])
                    
                    expand(new_R, new_P, new_X)
                    
                    P.remove(v)
                    X.add(v)
                    
            expand(set(), set(range(n_local)), set())
            return best_size, best_set

        # To handle large components safely, we check size.
        # If k > 60, exact search might be slow.
        # But let's try. If it times out, we fall back.
        
        # For very large components, we can use a heuristic first to get a lower bound,
        # then use that to prune the exact search?
        # Or just run a greedy heuristic to get a good initial best.
        
        # Greedy heuristic for initial bound
        greedy_set = set()
        # Sort by degree
        degs = [len(local_adj[i]) for i in range(k)]
        order = sorted(range(k), key=lambda x: -degs[x])
        for v in order:
            # Check if v is compatible with all in greedy_set
            ok = True
            for u in greedy_set:
                if u not in local_adj[v]:
                    ok = False
                    break
            if ok:
                greedy_set.add(v)
                
        # Pass greedy_set as initial best?
        # The expand function above doesn't take an initial best.
        # Let's modify expand to take initial best or just run it.
        # If the component is huge, this will hang.
        
        # Let's add a timeout mechanism or size check.
        if k > 100:
            # Use a faster heuristic or limited search
            # For now, let's just use the greedy result as the best for this component
            # and hope the exact search on smaller components or the greedy is good enough.
            # Actually, let's try to run a limited depth search or just the greedy.
            # Given the previous score, the max clique is likely < 50.
            # If the component is large but sparse, Max Clique is hard.
            # But LCM graphs are structured.
            
            # Let's try a simple local search on this component to improve the greedy bound.
            current_set = greedy_set
            improved = True
            while improved and time.time() < deadline:
                improved = False
                nodes_in_set = list(current_set)
                for i in range(len(nodes_in_set)):
                    u = nodes_in_set[i]
                    temp_set = current_set - {u}
                    # Find candidates
                    candidates = []
                    for v in range(k):
                        if v in current_set:
                            continue
                        # Check compatibility with temp_set
                        ok = True
                        for w in temp_set:
                            if w not in local_adj[v]:
                                ok = False
                                break
                        if ok:
                            candidates.append(v)
                            
                    if len(candidates) >= 2:
                        # Check pairs
                        for a_idx in range(len(candidates)):
                            v1 = candidates[a_idx]
                            for b_idx in range(a_idx + 1, len(candidates)):
                                v2 = candidates[b_idx]
                                if v2 in local_adj[v1]:
                                    current_set = temp_set | {v1, v2}
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                        
            if len(current_set) > best_clique_size:
                best_clique_size = len(current_set)
                best_clique_nodes = [comp_nodes[i] for i in current_set]
                
        else:
            # Exact search
            size, nodes_idx = max_clique_bronekerbosch(local_adj)
            if size > best_clique_size:
                best_clique_size = size
                best_clique_nodes = [comp_nodes[i] for i in nodes_idx]

    # The final set is best_clique_nodes plus 1
    final_set = set(best_clique_nodes)
    final_set.add(1)
    
    # Verify
    # If verification fails, fallback to [1]
    is_valid = True
    final_list = list(final_set)
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            a = final_list[i]
            b = final_list[j]
            g = gcd(a, b)
            if (a * b) // g > limit:
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        return [1]
        
    return sorted(final_set)

construct()
