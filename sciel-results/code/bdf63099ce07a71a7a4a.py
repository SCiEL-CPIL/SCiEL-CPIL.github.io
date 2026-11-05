# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing.
    
    Parameters:
    adjacency: bool matrix (n x n) where adjacency[i][j] is True if edge exists
    
    Returns:
    labels: array of 0/1 labels for each vertex
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # We'll work with the adjacency matrix. For an undirected graph, we assume
    # the matrix is symmetric. If not, we'll symmetrize it to be safe.
    # Actually, let's just use it as is, but ensure we handle the diagonal (no self-loops)
    np.fill_diagonal(adj, 0)
    
    # Initialize spins randomly
    np.random.seed(42)  # For reproducibility in evaluation
    spins = np.random.randint(0, 2, size=n).astype(np.int8)
    
    # Precompute degree for efficiency
    # The change in cut size when flipping vertex v:
    # If v is in set S (spin=1), flipping it to T (spin=0) changes cut by:
    #   edges from v to T (now become crossing) - edges from v to S (now become non-crossing)
    # Let's define: for vertex v, let same_v = number of neighbors with same spin
    #                let diff_v = number of neighbors with different spin
    # If we flip v, the new cut size change is:
    #   delta = diff_v - same_v = (total_degree_v - same_v) - same_v = total_degree_v - 2*same_v
    # But we need to be careful: when we flip v, the spins of neighbors don't change,
    # so we can compute the gain from flipping v as:
    #   gain = (number of neighbors with opposite spin) - (number of neighbors with same spin)
    #        = (deg_v - same_v) - same_v = deg_v - 2 * same_v
    
    deg = adj.sum(axis=1)
    
    # Helper to compute current cut size
    def cut_size(spins):
        # Cut size = number of edges (i,j) where spins[i] != spins[j]
        # = sum over i<j of (1 if spins[i]!=spins[j] else 0)
        # Efficient: for each i, count neighbors j>i with different spin
        # Or: 0.5 * sum_i sum_j adj[i][j] * (1 - spins[i]*spins[j] * 2 + 1) ... 
        # Simpler: create a matrix where M[i][j] = 1 if spins[i] != spins[j]
        # cut = 0.5 * (adj * (1 - (2*spins-1)[:,None] * (2*spins-1)[None,:])).sum()
        # But that's expensive. Let's do:
        # For each vertex i, the number of edges from i to vertices with different spin
        # sum that over all i, divide by 2
        s = 2 * spins - 1  # convert to -1/1
        # Edge (i,j) contributes if s[i] != s[j], i.e., s[i]*s[j] = -1
        # So contribution = (1 - s[i]*s[j]) / 2
        # Total cut = 0.5 * sum_{i,j} adj[i][j] * (1 - s[i]*s[j]) / 2 ... wait
        # Actually: number of crossing edges = 0.5 * sum_i sum_j adj[i][j] * [spins[i] != spins[j]]
        # [spins[i] != spins[j]] = 1 - spins[i]*spins[j] if spins are 0/1? No.
        # If spins are 0/1: [spins[i] != spins[j]] = 1 - spins[i]*(1-spins[j]) - (1-spins[i])*spins[j]
        # = 1 - spins[i] + spins[i]*spins[j] - spins[j] + spins[i]*spins[j]
        # = 1 - spins[i] - spins[j] + 2*spins[i]*spins[j]
        # Let's just do it directly for clarity, but it's O(n^2)
        # For n=100, this is fine.
        diff = (spins[:, None] != spins[None, :]).astype(np.int8)
        return int((adj * diff).sum() // 2)
    
    current_cut = cut_size(spins)
    best_spins = spins.copy()
    best_cut = current_cut
    
    # Simulated Annealing parameters
    # Temperature schedule: geometric cooling
    T_start = 10.0
    T_end = 0.1
    # Number of iterations: need to be careful with time limit (120s)
    # For n=100, each iteration is O(n) if we compute gain efficiently
    # Let's do multiple sweeps at each temperature
    num_temps = 500
    sweep_len = n  # one sweep = n vertex flips attempted
    
    # Cooling rate
    if num_temps > 1:
        cooling_rate = (T_end / T_start) ** (1.0 / (num_temps - 1))
    else:
        cooling_rate = 1.0
    
    T = T_start
    
    # Precompute row sums for efficiency
    # For vertex v, same_v = adj[v] @ spins (dot product)
    # But we need to be careful: adj[v] is a row, spins is a column
    # same_v = np.dot(adj[v], spins) gives number of neighbors with spin=1
    # If spins[v] == 1: same = same_v, diff = deg[v] - same_v
    # If spins[v] == 0: same = deg[v] - same_v, diff = same_v
    
    for _ in range(num_temps):
        # One sweep: try to flip each vertex once
        for v in range(n):
            # Compute gain from flipping vertex v
            # same_v = number of neighbors with same spin as v
            # If spins[v] == 1: same_v = adj[v] @ spins
            # If spins[v] == 0: same_v = deg[v] - (adj[v] @ spins)
            neighbor_sum = np.dot(adj[v], spins)
            if spins[v] == 1:
                same_v = neighbor_sum
            else:
                same_v = deg[v] - neighbor_sum
            
            diff_v = deg[v] - same_v
            gain = diff_v - same_v  # change in cut if we flip v
            
            if gain >= 0:
                # Always accept improving moves
                spins[v] = 1 - spins[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_spins = spins.copy()
            else:
                # Degrading move: accept with probability exp(gain/T)
                if T > 1e-10:
                    prob = np.exp(gain / T)
                    if np.random.random() < prob:
                        spins[v] = 1 - spins[v]
                        current_cut += gain
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_spins = spins.copy()
        
        # Cool down
        T *= cooling_rate
    
    return best_spins
# EVOLVE-BLOCK-END
