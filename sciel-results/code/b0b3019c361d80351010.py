import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    max_time = 110  # Safety margin for 120s limit
    
    # 1. Generate Baseline: Paley Graph on 17 vertices
    # This is a known K4-free graph with alpha=6? 
    # Wait, Paley(17) has alpha = ? 
    # R(4,6) > 17 is known. Paley(17) is K4-free. 
    # Independence number of Paley(17):
    # It is known that Paley(17) has independence number 6? 
    # If alpha(Paley(17)) >= 6, it's invalid.
    # Let's verify or use a known valid seed.
    # Actually, the seed program in the prompt uses Paley(17) and gets score 1.0.
    # The prompt says "A valid graph on n vertices certifies R(4,6) > n".
    # If Paley(17) is valid, then alpha <= 5.
    # Let's assume the seed is valid as per the prompt context (score 1.0).
    
    def make_paley(n):
        if n % 4 != 1:
            return None
        squares = {x * x % n for x in range(1, n)}
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in squares:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    base_graph = make_paley(17)
    if base_graph is None:
        return {'n': 1, 'edges': []}
        
    best_n = 17
    best_edges = base_graph['edges']
    
    # Helper functions for validation and manipulation
    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_k4(adj, n):
        # Check for K4 in O(n^4) worst case, but n is small (<=35)
        # Optimization: Check triangles in neighborhood of each node?
        # If we add edges incrementally, we can check locally.
        # For full check:
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    common = adj[i] & adj[j]
                    for k in common:
                        if k > i:
                            common2 = (adj[i] & adj[j] & adj[k])
                            for l in common2:
                                if l > i:
                                    return True
        return False

    def has_indep_6(adj, n):
        # Check for independent set of size 6
        # This is NP-hard, but n<=35 is small.
        # Use branching or simple backtracking.
        # We can use a simple recursive search with pruning.
        def search(idx, current_set, remaining_needed):
            if len(current_set) == 6:
                return True
            if idx >= n:
                return False
            # Pruning: not enough vertices left
            if (n - idx) < (6 - len(current_set)):
                return False
            
            # Option 1: Include idx
            # Check if idx is independent of all in current_set
            is_indep = True
            for v in current_set:
                if idx in adj[v]:
                    is_indep = False
                    break
            
            if is_indep:
                if search(idx + 1, current_set + [idx], remaining_needed - 1):
                    return True
            
            # Option 2: Exclude idx
            if search(idx + 1, current_set, remaining_needed):
                return True
            
            return False
        
        return search(0, [], 6)

    def is_valid(adj, n):
        if has_k4(adj, n):
            return False
        if has_indep_6(adj, n):
            return False
        return True

    # Attempt to grow the graph
    # We will try to add vertices one by one.
    # For each new vertex, we try random edge configurations.
    
    current_n = 17
    current_edges = list(best_edges)
    current_adj = build_adj(current_n, current_edges)
    
    # We need to try to extend to n=18, 19, ... up to 35
    # Strategy: Randomized Greedy with Restarts
    # We keep the best graph found.
    
    for target_n in range(18, 36):
        if time.time() - start_time > max_time:
            break
            
        # Try to extend from best known valid graph of size target_n-1
        # Actually, we should try to extend from the *current* best graph of that size.
        # But if we get stuck at 18, we can't get 19.
        # So we need a loop that tries to find *any* valid graph of size target_n.
        
        found = False
        attempts = 0
        max_attempts = 2000 # Adjust based on time
        
        while not found and time.time() - start_time < max_time and attempts < max_attempts:
            attempts += 1
            
            # Start from a random perturbation of a known good graph?
            # Or start from scratch? Starting from Paley(17) is good.
            # Let's start from Paley(17) and try to add (target_n - 17) vertices.
            
            work_n = 17
            work_edges = list(base_graph['edges'])
            work_adj = build_adj(work_n, work_edges)
            
            success_for_this_attempt = True
            
            for k in range(17, target_n):
                if time.time() - start_time > max_time:
                    success_for_this_attempt = False
                    break
                
                # Add vertex k
                # Determine edges to vertices 0..k-1
                # Constraints:
                # 1. No K4: The neighbors of k, say S, must form a triangle-free subgraph?
                #    No, if a,b,c in S and edges (a,b),(b,c),(c,a) exist, then {k,a,b,c} is K4.
                #    So S must be an independent set in the line graph? 
                #    Simpler: For any triangle (a,b,c) in the existing graph, 
                #    k cannot be connected to all three.
                
                # 2. No Alpha 6: The non-neighbors of k, say U, must not contain an independent set of size 5.
                #    i.e., alpha(G[U]) <= 4.
                
                # Let's try to find a valid edge set for vertex k.
                # We can model this as a SAT problem or just random search.
                # Random search:
                # Try random subsets of edges to previous vertices.
                
                valid_found_for_k = False
                local_attempts = 100
                
                for _ in range(local_attempts):
                    if time.time() - start_time > max_time:
                        valid_found_for_k = False
                        success_for_this_attempt = False
                        break
                        
                    # Randomly decide edges to 0..k-1
                    # Bias towards existing density? Paley(17) density is ~50%.
                    # Let's just flip coin.
                    new_edges = []
                    potential_neighs = set()
                    
                    for j in range(k):
                        if random.random() < 0.5:
                            new_edges.append([k, j])
                            potential_neighs.add(j)
                    
                    # Check K4 constraint:
                    # For any triangle (a,b,c) in G[0..k-1], k cannot connect to all.
                    # We need to pre-compute triangles or check on the fly.
                    # Checking on the fly is expensive.
                    # Let's check if potential_neighs contains a triangle from the existing graph.
                    
                    k4_violation = False
                    # Get triangles in existing graph
                    # This is O(k^3) per attempt. k is up to 35. 35^3 ~ 40k. 100 attempts * 40k * 17 steps = 68M ops. Might be slow in Python.
                    # Optimization: Pre-compute list of triangles for each step?
                    # Or just check: For each pair (a,b) in potential_neighs, if (a,b) is edge, 
                    # check if there is c in potential_neighs such that (a,c) and (b,c) are edges.
                    
                    # Let's do the efficient check:
                    # If potential_neighs contains a triangle, invalid.
                    # Iterate over pairs in potential_neighs.
                    neigh_list = list(potential_neighs)
                    for i in range(len(neigh_list)):
                        a = neigh_list[i]
                        for j in range(i + 1, len(neigh_list)):
                            b = neigh_list[j]
                            if b in work_adj[a]: # Edge (a,b) exists
                                # Check if there is c in potential_neighs connected to both a and b
                                common_ab = work_adj[a] & work_adj[b]
                                # Intersect with potential_neighs
                                if common_ab & potential_neighs:
                                    k4_violation = True
                                    break
                        if k4_violation:
                            break
                    
                    if k4_violation:
                        continue
                    
                    # Check Alpha constraint:
                    # Non-neighbors U = {0..k-1} \ potential_neighs
                    # Check if alpha(G[U]) <= 4.
                    # We can use a quick check:
                    # If U has size < 5, alpha <= |U| <= 4, so OK.
                    if len(set(range(k)) - potential_neighs) < 5:
                        alpha_ok = True
                    else:
                        # Need to check if G[U] has independent set of size 5.
                        # Build adj for U? Or just check on the fly.
                        # Since we only need to know if there IS one, we can run a small backtracking search on U.
                        # Max size of U is k-1.
                        u_list = list(set(range(k)) - potential_neighs)
                        
                        def has_indep_5_in_set(node_set, adj):
                            # Backtracking for indep set of size 5
                            def search(idx, current):
                                if len(current) == 5:
                                    return True
                                if idx >= len(node_set):
                                    return False
                                if (len(node_set) - idx) < (5 - len(current)):
                                    return False
                                
                                v = node_set[idx]
                                # Try include v
                                if all(v not in adj[u] for u in current):
                                    if search(idx + 1, current + [v]):
                                        return True
                                # Try exclude v
                                if search(idx + 1, current):
                                    return True
                                return False
                            
                            return search(0, [])
                        
                        alpha_ok = not has_indep_5_in_set(u_list, work_adj)
                    
                    if alpha_ok:
                        # Valid configuration for vertex k
                        # Update work graph
                        for e in new_edges:
                            work_edges.append(e)
                            a, b = e
                            work_adj[a].add(b)
                            work_adj[b].add(a)
                        valid_found_for_k = True
                        break
                
                if not valid_found_for_k:
                    success_for_this_attempt = False
                    break
                
            if success_for_this_attempt:
                # We successfully constructed a valid graph of size target_n
                best_n = target_n
                best_edges = work_edges
                found = True

        if not found:
            # Could not reach target_n
            # We can break, or try harder with more attempts?
            # Given time limits, break is safer to ensure we return a valid result.
            # However, maybe we can try to improve the current best_n by random walks?
            # For now, break.
            break

    # Final validation of the returned graph
    final_adj = build_adj(best_n, best_edges)
    if is_valid(final_adj, best_n):
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback to base
        return base_graph

# EVOLVE-BLOCK-END
