"""
Ramsey graph search: maximize n such that there exists an (n,5,5)-free graph.
Uses local search with simulated annealing on the number of K5 and independent-5
configurations, starting from the Paley graph on 37 vertices.
"""
import time
import random
import numpy as np

def _paley_graph(q=37):
    residues = {(x * x) % q for x in range(1, q)}
    adj = np.zeros((q, q), dtype=bool)
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i, j] = adj[j, i] = True
    return adj

def _count_k5_in_subgraph(adj_sub, k=5):
    """Count cliques of size k in a small subgraph given as adjacency matrix."""
    n = adj_sub.shape[0]
    if n < k:
        return 0
    # For small n, brute-force over combinations is fine, but let's be smart.
    # Count K5 by checking all 5-subsets.
    from itertools import combinations
    count = 0
    for combo in combinations(range(n), k):
        ok = True
        for i in range(k):
            for j in range(i + 1, k):
                if not adj_sub[combo[i], combo[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            count += 1
    return count

def _count_k5_fast(adj, k=5):
    """
    Count number of K5 cliques in the whole graph using adjacency matrix.
    For n <= 42, we can use a recursive backtracking or simply combinations
    with early pruning. Combinations(42,5) = 850668 which is too slow in Python.
    
    Better: For each 5-tuple, check. But that's still slow.
    
    Alternative: Use the fact that a K5 is determined by its vertices. 
    We can iterate over all 5-combinations. For n=42, that's ~850k checks.
    Each check is 10 edge lookups. ~8.5M operations. In Python this might take
    a few seconds. Let's try to optimize.
    
    Actually, let's use a different approach: count K5 by building from K4s.
    Or, use numpy to vectorize some parts.
    
    For simplicity and correctness, let's use combinations but with early exit.
    """
    from itertools import combinations
    n = adj.shape[0]
    count = 0
    # Precompute adjacency as a set of frozensets for quick lookup? No, matrix is fine.
    for combo in combinations(range(n), k):
        c = list(combo)
        # Check all pairs
        ok = True
        for i in range(k):
            ci = c[i]
            for j in range(i + 1, k):
                cj = c[j]
                if not adj[ci, cj]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            count += 1
    return count

def _count_independent_k5(adj, k=5):
    """Count independent sets of size k, i.e., K5 in the complement."""
    n = adj.shape[0]
    comp = ~adj
    # Ensure diagonal is False in comp too (irrelevant for k>=2)
    np.fill_diagonal(comp, False)
    return _count_k5_fast(comp, k)

def _count_triangles_in_subgraph(adj_sub):
    """Count triangles in a subgraph given by boolean adjacency matrix."""
    n = adj_sub.shape[0]
    if n < 3:
        return 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_sub[i, j]:
                # Count common neighbors of i and j that are > j (to avoid double counting)
                common = np.logical_and(adj_sub[i, :], adj_sub[j, :])
                # Only count vertices > j to count each triangle once
                common[:j + 1] = False
                count += np.sum(common)
    return count

def _delta_k5_when_flipping(adj, u, v):
    """
    Compute the change in number of K5s if we flip edge (u,v).
    If adding edge: delta = + (number of K4s in common neighborhood of u,v)
    If removing edge: delta = - (number of K4s in common neighborhood of u,v)
    
    Wait, that's not quite right. A K5 containing edge (u,v) needs 3 more vertices
    that form a triangle and are all adjacent to both u and v.
    So the number of K5s containing (u,v) = number of triangles in the common neighborhood.
    
    So if we add edge (u,v), we add triangles_count K5s.
    If we remove edge (u,v), we remove triangles_count K5s.
    """
    common = np.logical_and(adj[u, :], adj[v, :])
    common[u] = False
    common[v] = False
    # Get the subgraph induced by common neighbors
    indices = np.where(common)[0]
    if len(indices) < 3:
        return 0
    sub_adj = adj[np.ix_(indices, indices)]
    return _count_triangles_in_subgraph(sub_adj)

def _delta_ind5_when_flipping(adj, u, v):
    """
    Similar for independent sets. Flipping (u,v) affects independent sets containing {u,v}.
    An independent set of size 5 containing u,v needs 3 more vertices that are
    mutually non-adjacent and non-adjacent to both u and v.
    So we need to count independent triangles (independent 3-sets) in the
    common non-neighborhood of u and v.
    """
    # Common non-neighbors: vertices w != u,v such that not adj[u,w] and not adj[v,w]
    common_non = np.logical_and(~adj[u, :], ~adj[v, :])
    common_non[u] = False
    common_non[v] = False
    indices = np.where(common_non)[0]
    if len(indices) < 3:
        return 0
    # In the complement graph, we need to count cliques (triangles) in the induced subgraph
    comp = ~adj
    np.fill_diagonal(comp, False)
    sub_adj = comp[np.ix_(indices, indices)]
    return _count_triangles_in_subgraph(sub_adj)

def _total_cost(adj):
    """Total number of K5 + independent 5-sets."""
    k5 = _count_k5_fast(adj, 5)
    ind5 = _count_independent_k5(adj, 5)
    return k5 + ind5

def _is_valid(adj):
    return _total_cost(adj) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Start from Paley graph
    q = 37
    adj = _paley_graph(q)
    
    # Verify seed is valid
    assert _is_valid(adj), "Paley graph on 37 vertices should be (5,5)-free"
    
    best_n = q
    best_adj = adj.copy()
    
    # Try to extend to larger n
    # Strategy: Try to find a valid graph on n vertices for n = 42, 41, ..., 38
    # For each n, run local search for a limited time.
    
    def _try_find_valid_graph(target_n, time_budget):
        """
        Try to find a valid graph on target_n vertices.
        Starts from a random graph or extends the current best.
        """
        # Initialize: take the best_adj (37 vertices) and add (target_n - 37) new vertices
        # Connect new vertices randomly to old ones.
        adj_local = np.zeros((target_n, target_n), dtype=bool)
        if target_n > q:
            adj_local[:q, :q] = best_adj
            # Add edges from new vertices to old ones randomly
            for i in range(q, target_n):
                for j in range(q):
                    if random.random() < 0.5:
                        adj_local[i, j] = adj_local[j, i] = True
            # Add edges among new vertices randomly
            for i in range(q, target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_local[i, j] = adj_local[j, i] = True
        else:
            adj_local = best_adj[:target_n, :target_n].copy()
        
        # Compute initial cost
        current_cost = _total_cost(adj_local)
        
        if current_cost == 0:
            return adj_local
        
        # Local search with simulated annealing
        n = target_n
        temperature = 1.0
        cooling = 0.9999
        steps_per_temp = 50
        
        for step in range(1000000):
            if time.time() > start_time + time_budget:
                break
            
            # Pick a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Compute deltas
            # If edge is currently present, flipping removes it
            # If edge is currently absent, flipping adds it
            is_present = adj_local[u, v]
            
            delta_k5 = _delta_k5_when_flipping(adj_local, u, v)
            delta_ind5 = _delta_ind5_when_flipping(adj_local, u, v)
            
            # If adding edge: cost change = +delta_k5 - delta_ind5
            # If removing edge: cost change = -delta_k5 + delta_ind5
            if is_present:
                delta_cost = -delta_k5 + delta_ind5
            else:
                delta_cost = delta_k5 - delta_ind5
            
            # Accept with Metropolis criterion
            if delta_cost <= 0:
                accept = True
            else:
                # Simulated annealing
                if temperature > 1e-6:
                    prob = np.exp(-delta_cost / temperature)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                adj_local[u, v] = not is_present
                adj_local[v, u] = not is_present
                current_cost += delta_cost
                if current_cost < 0:
                    current_cost = 0  # Shouldn't happen, but clamp
                if current_cost == 0:
                    return adj_local
            
            # Cool down
            if step % steps_per_temp == 0:
                temperature *= cooling
        
        return None
    
    # Try from n=42 down to n=38
    for target_n in [42, 41, 40, 39, 38]:
        remaining = deadline - time.time()
        if remaining < 30:
            break
        time_budget = min(remaining * 0.5, 50)  # Give up to 50s per n
        result = _try_find_valid_graph(target_n, time_budget)
        if result is not None:
            # Verify
            if _is_valid(result):
                best_n = target_n
                best_adj = result
                break
    
    # If we found a valid graph, return it
    if best_n > q:
        n = best_n
        adj_final = best_adj
    else:
        n = q
        adj_final = adj
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_final[i, j]:
                edges.append((i, j))
    
    return n, edges
