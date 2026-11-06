"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Degrees
    deg = A.sum(axis=1)
    
    # Build an auxiliary weighted graph where edge weights capture
    # the marginal conflict between adjacent vertices.
    # For adjacent vertices u and v, the "conflict weight" is proportional
    # to how much picking one blocks the other from being in the independent set.
    # We use a spectral relaxation approach: find the principal eigenvector
    # of a matrix that encodes these conflicts, then use it as the priority score.
    
    # Construct conflict matrix: for each edge (u,v), the weight represents
    # how costly it is to have both u and v in the candidate set.
    # A natural choice: weight(u,v) = 1 / (deg[u] + deg[v] + 1)
    # This downweights edges where both endpoints are high-degree (since
    # high-degree vertices are less likely to be in the MIS anyway).
    
    # Actually, let's use a different approach: we want vertices that are
    # "isolated" in the sense that their neighbors are highly connected among
    # themselves (clique-like neighborhoods). Such vertices block fewer
    # independent-set candidates.
    
    # Compute the Laplacian of the graph
    D = np.diag(deg)
    L = D - A
    
    # For spectral relaxation, we look at the smallest non-trivial eigenvector
    # of the Laplacian, which captures the "bottleneck" structure.
    # Alternatively, we can use the Fiedler vector approach.
    
    # Let's try a power iteration approach on a modified adjacency matrix.
    # We define a "cooperation" matrix where entries are higher for vertices
    # that are less conflicting.
    
    # Simpler structural approach: use the eigenvector of the adjacency matrix
    # corresponding to the largest eigenvalue (principal eigenvector).
    # This captures the "centrality" in a spectral sense.
    
    # Power iteration for principal eigenvector of A
    # Initialize with random vector
    rng = np.random.default_rng(42)
    x = rng.random(n)
    x /= np.linalg.norm(x)
    
    # Number of power iterations - keep it moderate for speed
    max_iter = 50
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new
    
    # The principal eigenvector gives a spectral centrality score.
    # Vertices with high spectral centrality are "more central" in the graph.
    # For independent set, we might want to pick less central vertices first,
    # or we might want to pick central vertices first (to block their neighbors).
    
    # Let's combine spectral centrality with degree-based heuristics.
    # Low degree + high spectral centrality might be a good combination:
    # pick vertices that are somewhat central (so they block many neighbors)
    # but not too high degree (so they don't block too many potential MIS members).
    
    # Normalize the eigenvector
    x_norm = x / (np.abs(x).max() + 1e-12)
    
    # Also compute a "neighbor independence" score:
    # For each vertex v, compute the fraction of its neighbors that are
    # mutually exclusive (i.e., adjacent to each other).
    # If neighbors are densely connected, picking v blocks fewer MIS candidates.
    
    # Compute A^2 to get 2-step connections
    A2 = A @ A
    
    # For each vertex v, count edges among its neighbors:
    # edges_in_N(v) = 0.5 * sum_{u in N(v)} sum_{w in N(v), w != u} A[u,w]
    # = 0.5 * sum_{u in N(v)} (A2[v,u] - A[v,u])  [since A2[v,u] counts paths v->u->w, 
    #   and we want w in N(v), so A[v,w]=1, meaning w is a neighbor of v]
    # Actually, A2[v,u] for u in N(v) counts the number of common neighbors of v and u.
    # The number of edges within N(v) is:
    # 0.5 * sum_{u in N(v)} (number of neighbors of u that are also in N(v))
    # = 0.5 * sum_{u in N(v)} (A2[v,u] - A[v,u])  [subtract the direct edge v-u]
    
    # For each v, sum over neighbors u of (A2[v,u] - A[v,u])
    # This is: sum_u A[v,u] * (A2[v,u] - A[v,u])
    # = sum_u A[v,u]*A2[v,u] - sum_u A[v,u]^2
    # = (A * A2).sum(axis=1) - deg  [since A is 0/1, A^2 = A]
    
    edges_in_N = (A * A2).sum(axis=1) - deg
    
    # Normalize by degree to get average neighbor connectivity
    safe_deg = np.maximum(deg, 1.0)
    neighbor_connectivity = edges_in_N / safe_deg
    
    # Degree of neighbors
    neighbor_deg_sum = A @ deg
    mean_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Combine scores:
    # 1. -deg: prefer low degree
    # 2. + spectral_centrality: prefer spectrally central vertices (they block many)
    # 3. + neighbor_connectivity: prefer vertices whose neighbors are densely connected
    #    (picking such a vertex blocks fewer potential MIS members)
    # 4. + mean_neighbor_deg: prefer vertices adjacent to high-degree vertices
    
    # Normalize each component to [0, 1] range for better combination
    deg_min, deg_max = deg.min(), deg.max()
    deg_range = deg_max - deg_min + 1e-12
    deg_norm = (deg - deg_min) / deg_range
    
    spec_min, spec_max = x_norm.min(), x_norm.max()
    spec_range = spec_max - spec_min + 1e-12
    spec_norm = (x_norm - spec_min) / spec_range
    
    nc_min, nc_max = neighbor_connectivity.min(), neighbor_connectivity.max()
    nc_range = nc_max - nc_min + 1e-12
    nc_norm = (neighbor_connectivity - nc_min) / nc_range
    
    md_min, md_max = mean_neighbor_deg.min(), mean_neighbor_deg.max()
    md_range = md_max - md_min + 1e-12
    md_norm = (mean_neighbor_deg - md_min) / md_range
    
    # Weighted combination
    score = -0.8 * deg_norm + 0.5 * spec_norm + 0.4 * nc_norm + 0.3 * md_norm
    
    # Check deadline
    if time.time() > deadline:
        return score.astype(float)
    
    return score.astype(float)
# EVOLVE-BLOCK-END
