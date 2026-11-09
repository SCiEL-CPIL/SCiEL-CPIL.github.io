"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    """
    Solve Maximum Cut using spectral embedding, randomized hyperplane projections, 
    and fast local search with vectorized gain calculations.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    # Assuming adjacency is a list of lists or a matrix of 0/1
    adj = np.array(adjacency, dtype=np.int8)
    
    # Precompute degree and neighbor sums for fast gain calculation
    # degree[v] = sum(adj[v])
    degree = adj.sum(axis=1)
    
    def get_cut_size(labels):
        """
        Calculate the number of edges crossing the cut.
        labels: array of 0/1 of length n
        """
        # Convert labels to +1/-1 for easier math? Or just use diff.
        # Cut size = sum_{i<j} A[i][j] * (1 if labels[i] != labels[j] else 0)
        # Efficient way:
        # Let x be vector of labels (0/1).
        # We want sum A * (x_i != x_j).
        # Note that x_i != x_j is 1 if x_i xor x_j.
        # Alternatively, sum A * (x_i + x_j - 2*x_i*x_j) is not quite right for 0/1.
        # Let's stick to the robust method:
        # Create a matrix M where M[i][j] = 1 if labels[i] != labels[j]
        # This is O(n^2). For n=100, it's 10,000 ops. Very fast.
        diff = (labels[:, None] != labels[None, :]).astype(np.int8)
        return int(np.sum(adj * diff) // 2)

    def local_search(labels, max_rounds=50):
        """
        Fast local search.
        Gain of flipping vertex v:
        If v is in set S, flipping it moves it to T.
        Edges from v to S become cut edges.
        Edges from v to T become non-cut edges.
        Gain = (edges to T) - (edges to S)
        """
        labels = labels.copy()
        best_cut = get_cut_size(labels)
        
        # Precompute current partition sums for fast updates?
        # Actually, for n=100, we can just compute gain for all vertices at once in each pass.
        
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Calculate gain for flipping each vertex
            # gain[v] = sum_{u in T} A[v][u] - sum_{u in S} A[v][u]
            # where S = {u | labels[u] == labels[v]} and T = {u | labels[u] != labels[v]}
            
            # Vectorized gain calculation:
            # Let x be labels (0/1).
            # For vertex v, neighbors in same partition: adj[v] dot x (if labels[v]=1) or adj[v] dot (1-x) (if labels[v]=0)?
            # Let's use a simpler vectorized approach.
            # Let x be the label vector.
            # Let A be adjacency.
            # For vertex v, sum of neighbors with label 1: (A @ x)[v]
            # For vertex v, sum of neighbors with label 0: (A @ (1-x))[v] = degree[v] - (A @ x)[v]
            
            sum_neighbors_1 = adj @ labels  # Vector of length n
            sum_neighbors_0 = degree - sum_neighbors_1
            
            # If labels[v] == 1:
            # Current contribution to cut from v's edges: edges to 0s = sum_neighbors_0[v]
            # If flipped to 0: contribution becomes edges to 1s = sum_neighbors_1[v]
            # Gain = sum_neighbors_1[v] - sum_neighbors_0[v]
            
            # If labels[v] == 0:
            # Current contribution: edges to 1s = sum_neighbors_1[v]
            # If flipped to 1: contribution becomes edges to 0s = sum_neighbors_0[v]
            # Gain = sum_neighbors_0[v] - sum_neighbors_1[v]
            
            gains = np.where(labels == 1, sum_neighbors_1 - sum_neighbors_0, sum_neighbors_0 - sum_neighbors_1)
            
            # Find vertices with positive gain
            # To avoid simultaneous flip issues, we can flip all positive gains?
            # No, standard local search flips one by one or all if independent.
            # Flipping all positive gain vertices at once is safe if we re-evaluate?
            # Actually, if we flip all, the gains change.
            # Let's just flip all vertices where gain > 0 in one go, then recalculate.
            # This is "parallel" local search. It converges faster.
            
            flip_mask = gains > 0
            if np.any(flip_mask):
                labels[flip_mask] = 1 - labels[flip_mask]
                # Update best_cut efficiently?
                # It's easier to just recalculate cut size if we did a batch flip.
                # Or we can sum the gains? No, gains interact.
                # Recalculating cut size is O(n^2) = 10k ops. Very cheap.
                best_cut = get_cut_size(labels)
                improved = True
            
            if not improved:
                break
        
        return labels, best_cut

    def simulated_annealing(labels, max_iters=2000):
        """
        Simulated annealing with fast gain checks.
        """
        labels = labels.copy()
        current_cut = get_cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        T = 10.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            if T < 0.01:
                break
            
            # Pick a random vertex
            v = random.randint(0, n - 1)
            
            # Calculate gain for flipping v
            # Using the precomputed logic from local search but for a single vertex
            # sum_neighbors_1[v] and sum_neighbors_0[v]
            # We need current sums. Let's maintain them or compute on the fly.
            # Computing on the fly for one vertex is O(degree(v)).
            
            # Let's compute sum of neighbors with label 1 and 0 for v
            # This is O(n) if we loop, but we can use dot product?
            # adj[v] is a row.
            # sum_1 = np.dot(adj[v], labels)
            # sum_0 = degree[v] - sum_1
            
            sum_1 = np.dot(adj[v], labels)
            sum_0 = degree[v] - sum_1
            
            if labels[v] == 1:
                gain = sum_1 - sum_0
            else:
                gain = sum_0 - sum_1
            
            delta = gain
            
            if delta > 0 or random.random() < np.exp(delta / T):
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Revert if not accepted (though we didn't change it yet in this branch logic)
                # Wait, if we don't accept, we don't flip.
                pass
            
            T *= cooling
        
        return best_labels, best_cut

    # Step 1: Spectral embedding
    degree_float = degree.astype(float)
    D = np.diag(degree_float)
    L = D - adj.astype(float)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eig(L)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
    
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    fiedler = eigenvectors[:, 1] if n > 1 else eigenvectors[:, 0]
    labels_fiedler = (fiedler > 0).astype(int)
    
    # Step 2: Local search from Fiedler
    best_labels, best_cut = local_search(labels_fiedler, max_rounds=50)
    
    # Step 3: Random hyperplane projections
    k = min(5, n - 1)
    V = eigenvectors[:, 1:k+1]
    
    num_projections = 20
    for _ in range(num_projections):
        if time.time() > deadline - 20:
            break
        
        u = np.random.randn(k)
        u /= np.linalg.norm(u)
        projections = V @ u
        labels_proj = (projections > 0).astype(int)
        
        labels_l, cut_l = local_search(labels_proj, max_rounds=50)
        if cut_l > best_cut:
            best_cut = cut_l
            best_labels = labels_l.copy()
    
    # Step 4: Simulated Annealing
    if time.time() < deadline - 15:
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=2000)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()
    
    # Step 5: Final Polish
    if time.time() < deadline - 5:
        labels_final, cut_final = local_search(best_labels, max_rounds=50)
        if cut_final > best_cut:
            best_cut = cut_final
            best_labels = labels_final.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
