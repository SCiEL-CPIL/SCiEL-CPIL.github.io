"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy independent set selection.
    Returns a score per vertex; higher score = selected earlier.
    Uses conflict radius (edges among neighbors), degree, spectral centrality,
    and neighborhood features. Vertices with lower combined centrality/conflict
    are preferred (assigned higher priority).
    """
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Convert to float for computations
    A = adjacency.astype(float)

    # Degree
    degree = A.sum(axis=1)

    # Conflict radius: number of edges among neighbors of each vertex
    # For vertex i, conflict_radius[i] = number of edges in the subgraph induced by N(i)
    # This equals (A @ A @ degree) / 2 for each vertex? No.
    # Actually, for vertex i, the number of edges among its neighbors is:
    # sum_{j in N(i)} (degree[j] - 1) / 2 ... no that's not right either.
    # Let's compute it properly: for vertex i, look at neighbors N(i), count edges between them.
    # This is (1/2) * sum_{j in N(i)} sum_{k in N(i), k != j} A[j,k]
    # = (1/2) * (sum over j in N(i) of (A @ degree)[j] - degree[i])
    # Because sum_{k in N(i)} A[j,k] for fixed j is the number of neighbors of j that are also neighbors of i.
    
    # More carefully: for vertex i, edges among N(i) = (1/2) * sum_{j in N(i)} (number of neighbors of j that are in N(i))
    # The number of neighbors of j that are in N(i) = sum_{k in N(i)} A[j,k] (for j != i, and j in N(i))
    # = (A @ A)[j, i] ... hmm this is getting complicated.
    
    # Simpler approach: conflict_radius[i] = (A @ (A * degree[:, None]))[i] / 2? No.
    # Let's use: for each vertex i, conflict_radius[i] = 0.5 * (sum_{j: A[i,j]=1} (sum_{k: A[i,k]=1, k!=j} A[j,k]))
    # = 0.5 * (sum_{j in N(i)} ((A @ A)[j, i] - 1)) ... no
    # Actually (A @ A)[j, i] = number of common neighbors of j and i.
    # If j is a neighbor of i, then common neighbors of j and i are exactly the neighbors of j that are also neighbors of i (excluding i itself, but A[j,i]=1 so it's included).
    # So sum_{k in N(i)} A[j,k] includes k=i (since A[j,i]=1) and other common neighbors.
    # So sum_{k in N(i), k != i} A[j,k] = (A @ A)[j, i] - 1
    # Then conflict_radius[i] = 0.5 * sum_{j in N(i)} ((A @ A)[j, i] - 1)
    # = 0.5 * (sum_{j in N(i)} (A @ A)[j, i] - degree[i])
    # = 0.5 * ((A @ (A @ A.T))[i] ... wait, let me think again.
    
    # sum_{j in N(i)} (A @ A)[j, i] = sum_{j} A[i,j] * (A @ A)[j, i] = (A @ (A @ A))[i, i]? No.
    # (A @ (A @ A))[i, i] = sum_j A[i,j] * (A @ A)[j, i] ... yes that's what we want!
    # Wait, (A @ A)[j, i] = sum_k A[j,k] * A[k,i]. So sum_j A[i,j] * sum_k A[j,k] * A[k,i].
    # This counts triangles containing edge (i, ?) ... hmm.
    
    # Let me just compute it directly with matrix operations:
    # conflict_radius[i] = 0.5 * (sum_{j in N(i)} (number of neighbors of j that are also neighbors of i, excluding i))
    # number of neighbors of j that are in N(i) \ {i} = (A @ A)[j, i] - 1 (since A[j,i]=1 for j in N(i))
    # So conflict_radius[i] = 0.5 * (sum_{j in N(i)} ((A @ A)[j, i] - 1))
    # = 0.5 * (sum_{j} A[i,j] * (A @ A)[j, i] - degree[i])
    
    AA = A @ A  # (A@A)[j,i] = number of length-2 paths from j to i = common neighbors
    # sum_j A[i,j] * (A@A)[j,i] = (A @ AA)[i, i]? Let's check: (A @ AA)[i, i] = sum_j A[i,j] * AA[j, i]. Yes!
    sum_common = np.diag(A @ AA)
    conflict_radius = 0.5 * (sum_common - degree)
    
    # Ensure non-negative
    conflict_radius = np.maximum(conflict_radius, 0)

    # Power iteration for principal eigenvector (spectral centrality)
    max_iter = 200
    v = np.ones(n) / np.sqrt(n)
    v = v / (np.linalg.norm(v) + 1e-12)

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new = v_new / norm
        if np.linalg.norm(v_new - v) < 1e-10:
            v = v_new
            break
        v = v_new

    eigenvector = v

    # Neighborhood expansion: sum of degrees of neighbors
    nbr_degree_sum = A @ degree

    # Outward expansion: average degree of neighbors
    outward_exp = np.zeros(n)
    nonzero_deg = degree > 0
    outward_exp[nonzero_deg] = nbr_degree_sum[nonzero_deg] / degree[nonzero_deg]

    # Number of neighbors that are high-degree (above median)
    median_deg = np.median(degree)
    high_deg_mask = degree > median_deg
    high_nbr_count = A @ high_deg_mask.astype(float)

    # Normalize features
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    norm_degree = normalize(degree)
    norm_eigenvector = normalize(eigenvector)
    norm_conflict = normalize(conflict_radius)
    norm_nbr_deg_sum = normalize(nbr_degree_sum)
    norm_outward = normalize(outward_exp)
    norm_high_nbr = normalize(high_nbr_count)

    # Weighted combination - emphasize conflict radius and degree
    w1 = 0.30  # degree (lower degree preferred)
    w2 = 0.20  # spectral centrality (lower preferred)
    w3 = 0.25  # conflict radius (lower preferred) - NEW FEATURE
    w4 = 0.10  # neighbor degree sum
    w5 = 0.08  # outward expansion
    w6 = 0.07  # high-degree neighbor count

    centrality = (
        w1 * norm_degree +
        w2 * norm_eigenvector +
        w3 * norm_conflict +
        w4 * norm_nbr_deg_sum +
        w5 * norm_outward +
        w6 * norm_high_nbr
    )

    # Priority: higher score = selected first
    priority_scores = 1.0 - centrality

    return priority_scores
# EVOLVE-BLOCK-END
