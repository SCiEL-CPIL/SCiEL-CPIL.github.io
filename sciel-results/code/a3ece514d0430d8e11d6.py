"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def priority(adjacency):
    """
    Compute vertex priorities for a fixed greedy decoder.
    
    Strategy: Use a combination of structural features that help the greedy
    decoder find larger independent sets. The key insight is that vertices
    with low degree and low 'coreness' (in the k-core sense) should be selected
    first, as they are less likely to block many other vertices.
    
    We use a multi-step approach:
    1. Compute degree
    2. Compute a refined score based on local density and k-core peeling
    3. Return negative scores so that lower scores (higher priority) are picked first
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # Convert to numpy if not already
    adj = np.asarray(adjacency, dtype=bool)
    
    # Deadline guard: we have 120s total, but priority() is called per graph.
    # The evaluator kills after 150s wall-clock for the whole run.
    # We need to be fast. Let's aim for < 5s per graph.
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    # 1. Degree
    degree = adj.sum(axis=1).astype(float)
    
    # 2. K-core peeling to get core numbers
    # Core number of a vertex is the largest k such that the vertex belongs to a k-core.
    # We can compute this efficiently using a peeling process.
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    deg = degree.copy()
    
    # Use a bucket-based approach for k-core
    # Max degree is at most n-1
    max_deg = int(deg.max()) if n > 0 else 0
    if max_deg == 0:
        core = np.zeros(n, dtype=int)
    else:
        # Initialize buckets
        buckets = [[] for _ in range(max_deg + 1)]
        for i in range(n):
            buckets[int(deg[i])].append(i)
        
        # Peeling process
        k = 0
        while k <= max_deg:
            # Process all vertices in bucket k
            while buckets[k]:
                v = buckets[k].pop()
                if not remaining[v]:
                    continue
                remaining[v] = False
                core[v] = k
                # Update neighbors
                neighbors = np.where(adj[v] & remaining)[0]
                for u in neighbors:
                    if deg[u] > k:
                        # Move u from its current bucket to a lower one
                        old_deg = int(deg[u])
                        buckets[old_deg].remove(u)
                        deg[u] -= 1
                        new_deg = int(deg[u])
                        if new_deg >= k:
                            buckets[new_deg].append(u)
                        else:
                            # This shouldn't happen in standard peeling if we process in order
                            # but let's be safe
                            buckets[k].append(u)
            k += 1
        
        # Any remaining vertices have core number >= max_deg
        # Actually, the peeling process above should assign core numbers correctly
        # Vertices that were never peeled have core = their final degree in the remaining subgraph
        # But our loop goes up to max_deg, so all should be assigned
        # Let's verify: if a vertex has degree d, it can have core at most d.
        # The loop goes to max_deg, so all vertices should be processed.
    
    # 3. Local density: number of edges among neighbors
    # For each vertex v, count edges between its neighbors
    # This is O(n * avg_degree^2) which might be slow for dense graphs
    # Let's use a faster approximation or skip if too slow
    
    # Check time
    if time.time() > deadline - 5.0:
        # Not enough time for expensive computations
        # Use a simpler score
        score = -degree - 0.1 * core.astype(float)
        return score
    
    # Compute local density more efficiently
    # local_density[v] = number of edges in the subgraph induced by N(v)
    # This equals (sum of degrees of neighbors - degree[v]) / 2
    # But we need to be careful: the sum of degrees of neighbors includes edges to v
    # So: sum_deg_neighbors = adj[v] @ degree
    # Edges in N(v) = (sum_deg_neighbors - degree[v]) / 2
    # Wait, this counts edges from neighbors to v as well.
    # Let's think: sum_{u in N(v)} deg(u) = sum_{u in N(v)} (1 + number of neighbors of u other than v)
    # = degree[v] + 2 * (number of edges in N(v))
    # So: edges_in_N(v) = (sum_{u in N(v)} deg(u) - degree[v]) / 2
    
    sum_deg_neighbors = adj @ degree  # For each v, sum of degrees of its neighbors
    edges_in_neighborhood = (sum_deg_neighbors - degree) / 2.0
    
    # 4. Combine features
    # We want to prioritize vertices with:
    # - Low degree
    # - Low core number
    # - Low local density (fewer edges among neighbors means less 'cliquey')
    
    # Normalize features to [0, 1] range
    if n > 1:
        deg_norm = degree / degree.max() if degree.max() > 0 else np.zeros(n)
        core_norm = core.astype(float) / core.max() if core.max() > 0 else np.zeros(n)
        ld_norm = edges_in_neighborhood / edges_in_neighborhood.max() if edges_in_neighborhood.max() > 0 else np.zeros(n)
    else:
        deg_norm = np.zeros(n)
        core_norm = np.zeros(n)
        ld_norm = np.zeros(n)
    
    # Weighted combination
    # Lower values are better for greedy selection
    # We want to minimize: w1*deg + w2*core + w3*local_density
    # But local density might be less important
    score = 1.0 * deg_norm + 0.5 * core_norm + 0.2 * ld_norm
    
    # Return negative score so that greedy decoder picks lowest score first
    return -score

# EVOLVE-BLOCK-END
