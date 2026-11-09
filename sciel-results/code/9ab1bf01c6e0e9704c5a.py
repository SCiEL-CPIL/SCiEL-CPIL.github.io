"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a priority score to each vertex for the greedy MIS decoder.
    
    The greedy decoder picks vertices in descending score order, skipping
    neighbours of already-picked vertices. We want to maximize the number
    of vertices picked (independent set size).
    
    Strategy: Combine multiple structural features:
    1. Degree (lower degree is better - fewer conflicts)
    2. Neighborhood sparsity (how connected a vertex's neighbors are to each other)
       - If a vertex's neighbors are densely connected, picking this vertex 
         blocks many potential candidates. If sparse, it blocks fewer.
    3. Effective degree / coreness - how central a vertex is in the graph
    
    The key insight: when we pick a vertex, we remove it AND all its neighbors.
    So we want to pick vertices whose removal eliminates the fewest "valuable" 
    future candidates. A vertex with low degree is good. A vertex whose neighbors
    are all high-degree (and thus would block many others anyway) is also good
    because removing them doesn't lose much.
    
    We use a composite score that balances:
    - Low degree (primary)
    - Low "opportunity cost" of removing neighbors (neighbors that are also low-degree
      and have low neighborhood density are more valuable to keep)
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Degree of each vertex
    degree = adj.sum(axis=1).astype(float)
    
    # For each vertex, compute how "expensive" it is to remove its neighbors.
    # If a neighbor u of v has low degree and its neighbors are not very connected,
    # then u is a "good" candidate that we'd lose by picking v.
    
    # Neighborhood density for each vertex: fraction of edges among its neighbors
    # For vertex i, look at the subgraph induced by neighbors of i
    # n_i = degree[i], edges among neighbors = 0.5 * adj[neighbors_i, :][neighbors_i, :].sum()
    
    # Efficient computation using matrix multiplication
    # For vertex i, the number of edges among its neighbors is:
    # sum_{j in N(i)} sum_{k in N(i), k != j} adj[j,k] / 2
    # = 0.5 * (adj @ adj)[i, i] - 0.5 * degree[i]  ... wait, let me think again.
    
    # (adj @ adj)[i, j] = number of common neighbors of i and j
    # For i != j: (adj@adj)[i,j] = |N(i) ∩ N(j)|
    # For i == j: (adj@adj)[i,i] = degree[i]
    
    # Number of edges among neighbors of i:
    # For each neighbor j of i, count edges from j to other neighbors of i.
    # = sum_{j in N(i)} sum_{k in N(i), k != j} adj[j,k]
    # = sum_{j in N(i)} (sum_{k in N(i)} adj[j,k] - adj[j,j])
    # Since adj[j,j] = 0 (no self-loops):
    # = sum_{j in N(i)} sum_{k in N(i)} adj[j,k]
    # = (adj @ adj)[i, :] restricted... hmm, not quite.
    
    # Actually: let S_i = N(i). Then edges among S_i = 0.5 * sum_{j in S_i} sum_{k in S_i} adj[j,k]
    # = 0.5 * (A[S_i, S_i]).sum()
    
    # We can compute this as: for each i, sum over j in N(i) of (A[j, :][N(i)]).sum()
    # = for each i, sum over j in N(i) of sum over k in N(i) of A[j,k]
    # = (A @ A)[i, :] ... no.
    
    # Let me use: edges_in_N(i) = 0.5 * sum_{j=1}^n A[i,j] * (A[j, :] @ A[i, :])
    # Because A[j,:] @ A[i,:] = sum_k A[j,k]*A[i,k] = |N(j) ∩ N(i)|
    # And sum_{j in N(i)} |N(j) ∩ N(i)| = sum_{j in N(i)} sum_{k in N(i)} A[j,k] = 2 * edges_in_N(i)
    
    # So: 2 * edges_in_N(i) = sum_j A[i,j] * (A @ A)[i, j]... wait no.
    # sum_j A[i,j] * (A[j,:] @ A[i,:]) = sum_j A[i,j] * sum_k A[j,k] A[i,k]
    # = sum_{j,k} A[i,j] A[j,k] A[i,k]
    # This counts triangles containing i, times... hmm.
    
    # Let me just do it more directly but efficiently.
    # edges_among_neighbors[i] = 0.5 * sum_{j in N(i)} sum_{k in N(i)} A[j,k]
    
    # Using matrix ops: Let D = diag(A.sum(axis=1)). 
    # For each i: sum_{j in N(i)} sum_{k in N(i)} A[j,k] 
    # = sum_j A[i,j] * (sum_k A[i,k] * A[j,k])
    # = sum_j A[i,j] * ((A @ A.T)[i, j])  ... since A is symmetric, A @ A.T = A @ A
    # Wait: (A @ A)[i, j] = sum_k A[i,k] A[k,j]. Since A is symmetric, A[k,j]=A[j,k].
    # So (A @ A)[i, j] = sum_k A[i,k] A[j,k] = |N(i) ∩ N(j)|
    
    # So: sum_j A[i,j] * (A @ A)[i, j] = sum_{j in N(i)} |N(i) ∩ N(j)|
    # = sum_{j in N(i)} sum_{k in N(i)} A[j,k]  (since k in N(i) means A[i,k]=1, and A[j,k]=1 means k in N(j))
    # = 2 * edges_among_N(i)
    
    A2 = adj @ adj  # (n x n), A2[i,j] = |N(i) ∩ N(j)|
    
    # 2 * edges_among_N(i) = sum_j A[i,j] * A2[i,j]
    two_edges_among_N = (adj * A2).sum(axis=1)
    edges_among_N = two_edges_among_N / 2.0
    
    # Neighborhood density: edges_among_N[i] / (degree[i] * (degree[i]-1) / 2) for degree >= 2
    # This tells us how "cliquish" the neighborhood is.
    
    deg_safe = np.maximum(degree, 1.0)
    # For degree 0 or 1, neighborhood density is 0
    max_possible_edges = degree * (degree - 1) / 2.0
    max_possible_edges_safe = np.where(max_possible_edges > 0, max_possible_edges, 1.0)
    neigh_density = np.where(max_possible_edges > 0, edges_among_N / max_possible_edges_safe, 0.0)
    
    # Now compute a "cost" of picking each vertex.
    # When we pick vertex i, we remove i and all its neighbors.
    # The "value" we lose from removing neighbors is related to how good those neighbors are.
    # A neighbor j is "good" if it has low degree and low neighborhood density.
    
    # Let's define a "vertex quality" for each vertex:
    # quality[j] = -degree[j] - alpha * neigh_density[j] * degree[j]
    # Actually, let's think about what makes a vertex good to keep:
    # - Low degree: it conflicts with fewer vertices
    # - Its neighbors are densely connected: if we DON'T pick it, its neighbors might pick each other? No...
    
    # Alternative approach: The cost of picking i is the number of "regrettable" removals.
    # A neighbor j is regrettable if j would have been picked in a greedy process on its own.
    # Approximate: j is more valuable if it has lower degree.
    
    # Cost of picking i = sum_{j in N(i)} (1 / (1 + degree[j]))
    # This gives more weight to removing low-degree neighbors.
    
    # Also consider: if neighbors of i are densely connected, they'd block each other anyway,
    # so removing them is less costly.
    
    # Let's compute: for each i, the weighted sum of "value" of its neighbors,
    # where value of neighbor j = 1/(1+degree[j]) * (1 - neigh_density[j])
    # The (1 - neigh_density[j]) factor: if j's neighborhood is dense, j's neighbors would 
    # conflict with each other, so j is less "efficient" in the independent set.
    
    neighbor_value = 1.0 / (1.0 + degree) * (1.0 - neigh_density)
    
    # Cost of picking i = sum of neighbor_value over neighbors of i
    cost = adj @ neighbor_value
    
    # Primary signal: degree (lower is better)
    # Secondary: cost of removal (lower is better)
    # Tertiary: neighborhood density (lower density = neighbors are less interconnected, 
    #           meaning they'd be more "independent" candidates, so removing them is worse)
    
    # Normalize each component
    deg_min, deg_max = degree.min(), degree.max()
    deg_range = deg_max - deg_min if deg_max > deg_min else 1.0
    deg_norm = (degree - deg_min) / deg_range  # 0 = lowest degree, 1 = highest
    
    cost_min, cost_max = cost.min(), cost.max()
    cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
    cost_norm = (cost - cost_min) / cost_range  # 0 = lowest cost, 1 = highest
    
    # Score: we want high score for low degree and low cost
    # score = -w1 * deg_norm - w2 * cost_norm
    # Let's weight degree more heavily
    
    w1 = 1.0
    w2 = 0.5
    
    score = -w1 * deg_norm - w2 * cost_norm
    
    # Add a small tiebreaker based on vertex index for determinism
    # (not strictly needed but helps with consistent behavior)
    
    return score.astype(float)

# EVOLVE-BLOCK-END
