import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=np.int8)
    # Ensure symmetry (just in case, though task says unweighted graphs)
    # adj = (adj + adj.T) // 2
    # Zero diagonal
    np.fill_diagonal(adj, 0)
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_C = -1
    best_x = None
    
    # Helper to compute cut size efficiently
    # C = 0.5 * (sum(A) - x^T A x) is not quite right for 0/1 or -1/1 directly without care.
    # Let's stick to the S-vector method for updates, but we need initial C.
    # For x in {-1, 1}:
    # Cut size C = 0.25 * sum_{i,j} (1 - x_i x_j) A_{ij}
    #             = 0.25 * (sum_{i,j} A_{ij} - sum_{i,j} x_i x_j A_{ij})
    #             = 0.25 * (2*E - x^T A x)
    # Where E is total edges (sum of upper triangle) or 0.5*sum(A).
    # Let S_total = sum(A) = 2E.
    # C = 0.25 * (S_total - x^T A x)
    
    S_total = np.sum(adj)
    
    def get_cut(x):
        # x is -1/1
        xAx = x @ (adj @ x)
        return 0.25 * (S_total - xAx)
    
    # Spectral Initialization
    # Use the largest eigenvector of A (or A - lambda I) to get a good start.
    # For Max Cut, the spectral bisection uses the second smallest eigenvector of Laplacian L = D - A.
    # Or simply the largest eigenvector of A if we consider it as a correlation matrix?
    # Actually, for Max Cut, we want to separate nodes.
    # The Laplacian L = D - A. The Fiedler vector (2nd smallest eigenvalue of L) gives a good partition.
    # Let's use scipy.sparse.linalg.eigsh for speed if n is large, but n=100 is small.
    # numpy.linalg.eig is fine.
    
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    
    try:
        # Get eigenvalues and vectors
        # We want the eigenvector corresponding to the second smallest eigenvalue of L.
        # Since L is symmetric, use eigh.
        vals, vecs = np.linalg.eigh(L)
        # vals are sorted ascending.
        # vecs[:, 0] is for smallest (usually 0 for connected graph).
        # vecs[:, 1] is Fiedler vector.
        fiedler = vecs[:, 1]
        # Initialize x based on sign of fiedler
        x = np.sign(fiedler)
        # Handle zeros
        x[x == 0] = 1
    except:
        # Fallback to random if spectral fails
        rng = np.random.default_rng(42)
        x = rng.choice([-1, 1], size=n)
    
    # Local Search (Greedy 1-flip)
    # We can speed this up by maintaining S = A x.
    # Gain of flipping i:
    # If x_i flips to -x_i:
    # New C = Old C + x_i * S_i
    # S_i = sum_j A_ij x_j
    # If we flip x_i, S_j for all j != i changes by -2 * A_ij * x_i (old x_i)
    # Wait, S is the vector of weighted sums.
    # Let's verify gain formula.
    # C = 0.25 * (S_total - x^T A x)
    # Let x' be x with x_i flipped.
    # x'^T A x' = x^T A x - 2 * x_i * (A x)_i
    # So C' = 0.25 * (S_total - (x^T A x - 2 x_i S_i)) = C + 0.5 * x_i S_i
    # Wait, earlier I used C += gain where gain = x_i S_i.
    # Let's re-derive carefully.
    # C = sum_{i<j, x_i != x_j} 1
    # Let's use the S-vector update logic which is standard.
    # If we flip vertex i:
    # The change in cut size is:
    # For each neighbor j:
    #   If x_i == x_j, edge was NOT in cut. After flip, x_i != x_j, edge IS in cut. Gain +1.
    #   If x_i != x_j, edge WAS in cut. After flip, x_i == x_j, edge NOT in cut. Gain -1.
    # So Gain = (# neighbors with same label) - (# neighbors with diff label)
    # S_i = sum_j A_ij x_j = (# same) * 1 + (# diff) * (-1)  [assuming x_i=1]
    # If x_i = 1:
    #   Same: x_j=1. Diff: x_j=-1.
    #   S_i = N_same - N_diff
    #   Gain = N_same - N_diff = S_i
    # If x_i = -1:
    #   Same: x_j=-1. Diff: x_j=1.
    #   S_i = N_same * (-1) + N_diff * (1) = N_diff - N_same
    #   Gain = N_same - N_diff = -S_i
    # So Gain = x_i * S_i.
    # And C_new = C_old + x_i * S_i.
    
    # Let's implement a fast greedy local search with restarts.
    
    rng = np.random.default_rng(42)
    
    def local_search_1flip(x, S=None):
        # x is -1/1 array
        if S is None:
            S = adj @ x
        C = get_cut(x)
        
        improved = True
        while improved:
            improved = False
            gains = x * S
            i = np.argmax(gains)
            max_gain = gains[i]
            
            if max_gain > 0:
                v = x[i]
                x[i] = -v
                C += max_gain
                S -= 2 * v * adj[:, i]
                improved = True
        
        return x, C, S
    
    def local_search_2swap(x, S=None):
        if S is None:
            S = adj @ x
        C = get_cut(x)
        
        improved = True
        while improved:
            improved = False
            gains_1d = x * S
            # For 2-swap of i and j:
            # Gain = gain_i + gain_j - 2 * A[i,j] * (1 if x[i]==x[j] else -1)
            # where gain_i = x[i]*S[i], gain_j = x[j]*S[j]
            # If x[i]==x[j]: flipping both doesn't change their relative relation, but changes relations with others
            # Actually, let's derive properly.
            # When we flip both i and j:
            # For edges (i,k) where k not in {i,j}: gain = x[i]*A[i,k]*x[k] if x[i] flips... 
            # Let me use the standard formula:
            # Delta C for flipping i alone: x[i]*S[i]
            # Delta C for flipping j alone: x[j]*S[j]  
            # Delta C for flipping both: x[i]*S[i] + x[j]*S[j] - 2*A[i,j]*x[i]*x[j]
            # Wait, need to be careful. Let's think about it differently.
            # 
            # Cut size C = sum_{i<j} A[i,j] * (1 - x[i]*x[j])/2
            # 
            # When we flip both x[i] and x[j]:
            # - Edge (i,j): term changes from A[i,j]*(1-x[i]*x[j])/2 to A[i,j]*(1-(-x[i])*(-x[j]))/2 = same!
            #   So edge (i,j) contribution doesn't change.
            # - For edge (i,k), k not in {i,j}: 
            #   Old: A[i,k]*(1-x[i]*x[k])/2
            #   New: A[i,k]*(1-(-x[i])*x[k])/2 = A[i,k]*(1+x[i]*x[k])/2
            #   Change: A[i,k]*x[i]*x[k]
            # - For edge (j,k), k not in {i,j}:
            #   Old: A[j,k]*(1-x[j]*x[k])/2
            #   New: A[j,k]*(1-(-x[j])*x[k])/2 = A[j,k]*(1+x[j]*x[k])/2
            #   Change: A[j,k]*x[j]*x[k]
            # 
            # Total change = x[i]*sum_{k!=i,j} A[i,k]*x[k] + x[j]*sum_{k!=i,j} A[j,k]*x[k]
            #            = x[i]*(S[i] - A[i,j]*x[j]) + x[j]*(S[j] - A[i,j]*x[i])
            #            = x[i]*S[i] - x[i]*A[i,j]*x[j] + x[j]*S[j] - x[j]*A[i,j]*x[i]
            #            = x[i]*S[i] + x[j]*S[j] - 2*A[i,j]*x[i]*x[j]
            #
            # So the pair gain is: gains_1d[i] + gains_1d[j] - 2*adj[i,j]*x[i]*x[j]
            
            # Compute all pair gains efficiently
            # G[i,j] = gains_1d[i] + gains_1d[j] - 2*adj[i,j]*x[i]*x[j]
            G = gains_1d[:, None] + gains_1d[None, :]
            # Subtract 2*adj[i,j]*x[i]*x[j]
            xx = x[:, None] * x[None, :]
            G -= 2 * adj * xx
            np.fill_diagonal(G, -1)
            
            max_idx = np.unravel_index(np.argmax(G), G.shape)
            i, j = max_idx
            max_pair_gain = G[i, j]
            
            if max_pair_gain > 0:
                vi = x[i]
                vj = x[j]
                x[i] = -vi
                x[j] = -vj
                C += max_pair_gain
                S -= 2 * vi * adj[:, i]
                S -= 2 * vj * adj[:, j]
                improved = True
        
        return x, C, S
    
    def full_local_search(x):
        x, C, S = local_search_1flip(x)
        x, C, S = local_search_2swap(x, S)
        x, C, S = local_search_1flip(x, S)
        # Additional polish: alternate 1-flip and 2-swap a few more times
        for _ in range(3):
            x, C, S = local_search_2swap(x, S)
            x, C, S = local_search_1flip(x, S)
        return x, C
    
    # First pass: Spectral
    x_init = x.copy()
    x_ls, C_ls = full_local_search(x_init)
    if C_ls > best_C:
        best_C = C_ls
        best_x = x_ls.copy()
    
    # Multi-start with random and perturbed initializations
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if i % 3 == 0:
            x_curr = rng.choice([-1, 1], size=n)
        elif i % 3 == 1:
            x_curr = best_x.copy()
            num_flips = max(1, n // 10)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        else:
            x_curr = best_x.copy()
            num_flips = max(1, n // 20)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        
        x_ls, C_ls = full_local_search(x_curr)
        
        if C_ls > best_C:
            best_C = C_ls
            best_x = x_ls.copy()
    
    # Final polish: run more iterations on the best solution found
    if best_x is not None and time.time() < deadline - 2:
        for _ in range(5):
            if time.time() > deadline - 1:
                break
            x_polish, C_polish, _ = local_search_1flip(best_x.copy())
            x_polish, C_polish, S_polish = local_search_2swap(x_polish, None)
            x_polish, C_polish, _ = local_search_1flip(x_polish)
            # Run a few more alternations
            for _ in range(5):
                x_polish, C_polish, S_polish = local_search_2swap(x_polish, S_polish)
                x_polish, C_polish, _ = local_search_1flip(x_polish)
            if C_polish > best_C:
                best_C = C_polish
                best_x = x_polish
    
    if best_x is None:
        best_x = rng.choice([-1, 1], size=n)
        
    labels = (best_x + 1) // 2
    return labels.astype(int)
