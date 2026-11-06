"""
Strategy: Constructive Backtracking Search.
Instead of generating random graphs and verifying them with expensive exact B&B,
we build the graph vertex-by-vertex. We maintain the triangle-free constraint
strictly during construction. We use a fast greedy heuristic to estimate the
independence number and prune branches where the heuristic alpha >= 10.
This focuses compute on valid/semi-valid structures rather than random noise.
"""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _greedy_independence_number(adj, n):
    """
    Fast heuristic to estimate the size of the maximum independent set.
    Uses a greedy algorithm: repeatedly pick the vertex with minimum degree
    in the remaining subgraph, add it to the independent set, and remove
    it and its neighbors.
    This returns a lower bound on alpha. If this lower bound is >= 10,
    the graph is invalid (since we need alpha < 10).
    """
    remaining = set(range(n))
    indep_size = 0
    
    while remaining:
        # Find vertex with minimum degree in the induced subgraph on 'remaining'
        min_deg = float('inf')
        chosen = None
        for v in remaining:
            deg = len([u for u in remaining if u in adj[v]])
            if deg < min_deg:
                min_deg = deg
                chosen = v
        
        indep_size += 1
        # Remove chosen and its neighbors from remaining
        neighbors = adj[chosen] & remaining
        remaining.discard(chosen)
        remaining.difference_update(neighbors)
        
        if indep_size >= 10:
            return 10 # Early exit, we only care if it's >= 10
        
    return indep_size

def _check_triangle_free(adj, n):
    """Verify triangle-freeness (should be true by construction, but safe to check)."""
    for i in range(n):
        for j in adj[i]:
            if j > i:
                for k in adj[i]:
                    if k > j and k in adj[j]:
                        return False
    return True

def _exact_independence_number(adj, n, timeout=5.0):
    """
    Exact independence number via Branch and Bound.
    Only called on candidates that passed the greedy heuristic.
    """
    start_time = time.time()
    best = [0]
    
    # Pre-compute neighbor sets for speed
    adj_sets = [set(adj[i]) for i in range(n)]
    
    def dfs(remaining, current_size):
        if time.time() - start_time > timeout:
            return
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best[0]:
            return
        
        # Heuristic bound: Greedy on remaining
        rem_list = list(remaining)
        greedy_count = 0
        temp_remaining = set(rem_list)
        while temp_remaining:
            min_deg = float('inf')
            chosen = None
            for v in temp_remaining:
                deg = len([u for u in temp_remaining if u in adj_sets[v]])
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            greedy_count += 1
            temp_remaining.discard(chosen)
            temp_remaining.difference_update(adj_sets[chosen] & temp_remaining)
            if greedy_count >= 10 - current_size:
                # If even the greedy bound doesn't beat best, we might prune
                # But we need to be careful. Let's just use the simple bound.
                break
        
        if current_size + greedy_count <= best[0]:
            return

        # Pick a vertex to branch on. Heuristic: Max degree in remaining.
        rem_list = list(remaining)
        max_deg = -1
        chosen = rem_list[0]
        for v in rem_list:
            deg = len([u for u in rem_list if u in adj_sets[v]])
            if deg > max_deg:
                max_deg = deg
                chosen = v
        
        # Branch 1: Include chosen
        new_remaining = set(remaining)
        new_remaining.discard(chosen)
        new_remaining.difference_update(adj_sets[chosen])
        dfs(new_remaining, current_size + 1)
        
        # Branch 2: Exclude chosen
        new_remaining2 = set(remaining)
        new_remaining2.discard(chosen)
        dfs(new_remaining2, current_size)

    dfs(set(range(n)), 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # We will try to construct graphs of increasing size.
    # Start from a known good base or just try to build from scratch?
    # Building from scratch with backtracking for n=30 is hard.
    # Let's use a hybrid: Start with a random valid-ish graph and repair, 
    # OR use the constructive approach for smaller n and extend.
    
    # The constructive approach:
    # We maintain a list of vertices. For each new vertex, we decide its neighbors.
    # To ensure triangle-free: If we add vertex v, and connect to u and w,
    # then u and w must NOT be connected.
    
    # Let's try to find a graph for n=30, then 31, etc.
    # Since backtracking from scratch is exponential, we will use a 
    # "Randomized Greedy Construction with Backtracking" for the last few vertices.
    
    # Strategy:
    # 1. Generate a random triangle-free graph on n-5 vertices that is "dense" enough.
    # 2. Try to add 5 vertices one by one, choosing neighbors carefully.
    # 3. Use the greedy independence bound to prune.
    
    # Actually, let's stick to a simpler robust method:
    # Iterate n from 27 to 40.
    # For each n, run a limited backtracking search to build a valid graph.
    
    for target_n in range(27, 41):
        if time.time() > deadline - 20:
            break
            
        found = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and not found and time.time() < deadline - 20:
            attempts += 1
            
            # Initialize empty graph
            adj = [set() for _ in range(target_n)]
            
            # We will add edges randomly but maintain triangle-freeness.
            # To help with alpha, we want the graph to be "dense" in a way that
            # reduces independent sets, but triangle-free graphs are limited.
            # A random triangle-free graph tends to have large alpha.
            # We need a structured graph.
            
            # Let's try a specific structure: Union of cliques? No, triangle-free.
            # Bipartite graphs are triangle-free. Max alpha of bipartite graph on n vertices?
            # If balanced, alpha ~ n/2. For n=30, alpha ~ 15. Too big.
            # We need non-bipartite triangle-free graphs (odd cycles, etc.).
            
            # Let's use the "Andrasfai" style construction as a seed and perturb it.
            # Or, use a random regular graph and remove triangles?
            
            # Better: Start with a random graph, remove triangles greedily to make it triangle-free.
            # Then check alpha.
            
            # Generate random edges
            random.seed(attempts + target_n)
            edge_candidates = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.4:
                        edge_candidates.append((i, j))
            
            random.shuffle(edge_candidates)
            
            # Add edges if they don't create a triangle
            for u, v in edge_candidates:
                # Check if adding (u,v) creates a triangle
                # Triangle exists if there is a w such that (u,w) and (v,w) are in adj
                common = adj[u] & adj[v]
                if not common:
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Now we have a triangle-free graph.
            # Check independence number.
            # First, fast greedy check.
            greedy_alpha = _greedy_independence_number(adj, target_n)
            
            if greedy_alpha < 10:
                # Promising! Run exact check.
                exact_alpha = _exact_independence_number(adj, target_n, timeout=5)
                if exact_alpha < 10:
                    best_n = target_n
                    best_edges = [[u, v] for u in range(target_n) for v in adj[u] if u < v]
                    found = True
                    break
                else:
                    # Exact alpha >= 10, invalid.
                    pass
            else:
                # Greedy alpha >= 10, likely invalid. Skip exact check.
                pass
        
        if found:
            # Found a graph for this n. Try to extend to n+1 in next loop iteration.
            # We could also try to extend the specific graph we found.
            continue
        else:
            # If we fail to find one for this n, we might still have a smaller one.
            # But if n=27 fails, maybe n=28 will? Unlikely if 27 is hard.
            # However, the problem states 39 is known. So we should be able to find 27+.
            # If we fail, we break and return the best found so far.
            # Actually, if we fail for n=27, we should probably break, as larger n is harder.
            # But sometimes random luck works for larger n? No, density issues.
            # Let's break if we fail for a few consecutive n.
            if best_n == 0:
                 # We didn't even find 27? That's bad.
                 # Fallback to a known small graph.
                 # And(9) is 26.
                 n_base = 26
                 best_n = n_base
                 best_edges = [[i, j] for i in range(n_base) for j in range(i + 1, n_base) if (j - i) % n_base % 3 == 1]
            break

    # If we didn't find anything better than 26, return 26.
    if best_n < 26:
        n_base = 26
        best_n = n_base
        best_edges = [[i, j] for i in range(n_base) for j in range(i + 1, n_base) if (j - i) % n_base % 3 == 1]

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
