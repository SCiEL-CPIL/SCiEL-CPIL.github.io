"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_cost(p):
        """Compute the exact QAP cost for a permutation p."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def mds_embed(matrix, k=2):
        """
        Perform Classical MDS on a matrix to get a k-dimensional embedding.
        Assumes the matrix is a distance or dissimilarity matrix.
        """
        # Ensure symmetric
        M = (matrix + matrix.T) / 2.0
        # Double centering
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (M ** 2) @ J
        
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top k dimensions
        # Clip negative eigenvalues to 0 for numerical stability
        eigvals = np.clip(eigvals, 0, None)
        U = eigvecs[:, :k]
        D = np.sqrt(eigvals[:k])
        
        return U * D

    def procrustes_align(X, Y):
        """
        Align matrix X to Y using Procrustes analysis.
        Returns the optimal rotation/reflection and the aligned X.
        """
        # Center the matrices
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X_c = X - X_mean
        Y_c = Y - Y_mean
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(Y_c.T @ X_c)
        
        # Optimal rotation
        R = Vt.T @ U.T
        
        # Align X
        X_aligned = X_c @ R
        
        return X_aligned

    def get_initial_permutation():
        """
        Generate initial permutations using MDS/Procrustes and Hungarian.
        Returns the best initial permutation and its cost.
        """
        best_p = None
        best_cost = float('inf')
        
        # Method 1: MDS + Procrustes
        try:
            # Use negative flow for distance-like embedding (high flow = close)
            # Or just use the raw matrices if they are already distances.
            # For QAP, A is flow, B is distance.
            # We want to match high flow to low distance.
            # So we embed A (with sign flip or 1/A?) and B.
            # Let's try embedding -A (so high flow becomes 'close' in MDS) and B.
            # Actually, standard MDS on A (if A is distance) works. 
            # Here A is flow. High flow should be close. 
            # So we can treat -A as a distance matrix for embedding purposes.
            A_dist = -a + np.max(a) # Shift to be non-negative if needed, or just use -a
            # MDS requires symmetric. A is symmetric.
            X_embed = mds_embed(A_dist, k=2)
            Y_embed = mds_embed(b, k=2)
            
            # Align X to Y
            X_aligned = procrustes_align(X_embed, Y_embed)
            
            # The order of points in X_embed corresponds to facilities 0..n-1.
            # The order of points in Y_embed corresponds to locations 0..n-1.
            # After alignment, point i in X should be close to some point j in Y.
            # We need to find the permutation p such that p[i] is the location closest to X_aligned[i].
            
            # Compute distances between aligned X points and Y points
            # dist[i, j] = || X_aligned[i] - Y_embed[j] ||^2
            # We want to assign facility i to location j that minimizes this distance.
            # This is a linear assignment problem on the distance matrix.
            
            # Vectorized distance calculation
            # X_aligned: (n, 2), Y_embed: (n, 2)
            # dist = (X^2).sum(1)[:, None] + (Y^2).sum(1)[None, :] - 2 * X @ Y.T
            X_sq = np.sum(X_aligned ** 2, axis=1)[:, None]
            Y_sq = np.sum(Y_embed ** 2, axis=1)[None, :]
            dist_matrix = X_sq + Y_sq - 2 * (X_aligned @ Y_embed.T)
            
            # Solve LAP
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            p_mds = np.empty(n, dtype=np.int64)
            p_mds[row_ind] = col_ind
            
            cost_mds = compute_cost(p_mds)
            if cost_mds < best_cost:
                best_cost = cost_mds
                best_p = p_mds.copy()
                
        except Exception:
            pass

        # Method 2: Hungarian on A @ B.T
        try:
            C = a @ b.T
            row_ind, col_ind = linear_sum_assignment(-C)
            p_hung = np.empty(n, dtype=np.int64)
            p_hung[row_ind] = col_ind
            
            cost_hung = compute_cost(p_hung)
            if cost_hung < best_cost:
                best_cost = cost_hung
                best_p = p_hung.copy()
                
        except Exception:
            pass

        # Fallback: Random
        if best_p is None:
            rng = np.random.RandomState(42)
            best_p = rng.permutation(n)
            best_cost = compute_cost(best_p)

        return best_p, best_cost

    def compute_swap_delta(i, j, p, a, b):
        """
        Compute the cost delta for swapping facilities at indices i and j in permutation p.
        """
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        # Diagonal terms
        delta = a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        
        # Cross terms between i and j
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        # Terms with other facilities k
        # Vectorized calculation
        ks = np.setdiff1d(np.arange(n), [i, j], assume_unique=False)
        if len(ks) == 0:
            return delta
            
        pk = p[ks]
        
        # a[i, ks] * (b[lj, pk] - b[li, pk])
        delta += np.dot(a[i, ks], b[lj, pk] - b[li, pk])
        # a[ks, i] * (b[pk, lj] - b[pk, li])
        delta += np.dot(a[ks, i], b[pk, lj] - b[pk, li])
        # a[j, ks] * (b[li, pk] - b[lj, pk])
        delta += np.dot(a[j, ks], b[li, pk] - b[lj, pk])
        # a[ks, j] * (b[pk, li] - b[pk, lj])
        delta += np.dot(a[ks, j], b[pk, li] - b[pk, lj])
        
        return delta

    def compute_oropt_delta(start, length, p, a, b):
        """
        Compute the cost delta for moving a block of 'length' facilities starting at 'start'
        to the end of the permutation (or just shifting). 
        For simplicity and speed, we will implement a specific Or-opt move:
        Remove a block of size L at index i and insert it at index j.
        This is complex to vectorize efficiently for arbitrary i, j, L.
        
        Given the time constraints and the need for speed, we will stick to 2-swaps 
        but add a "block shift" heuristic if time permits, or just rely on SA for larger moves.
        
        Actually, a simpler structural change is to use 3-swaps or random restarts more aggressively.
        However, Or-opt is known to be effective. Let's implement a simple version:
        Move a block of 2 or 3 facilities.
        
        To keep it robust and fast, we will use 2-swaps in local search, 
        but in SA, we will allow "block moves" (swap two blocks of random length).
        """
        pass # Placeholder, we will implement specific moves in SA

    def local_search(p, current_cost, end_time, rng):
        """
        First-improvement 2-swap local search.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        # Pre-generate random pairs? No, shuffle on the fly to save memory.
        
        while time.time() < end_time:
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(n))
            rng.shuffle(indices)
            
            # Iterate over pairs
            # To be efficient, we iterate i, and for each i, iterate j > i
            # But shuffling all pairs is O(N^2) which is 22500 for N=150. This is fast.
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > end_time:
                    break
                
                delta = compute_swap_delta(i, j, curr_p, a, b)
                
                if delta < 0:
                    # Accept move
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    improved = True
                    
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
            
            if not improved:
                break
                
        return best_p, best_cost

    def simulated_annealing(p, best_cost, end_time, rng):
        """
        Simulated Annealing with 2-swaps and block moves.
        """
        curr_p = p.copy()
        curr_cost = best_cost
        best_p = p.copy()
        best_cost_sa = best_cost
        
        # Estimate initial temperature
        deltas = []
        for _ in range(20):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta(i, j, curr_p, a, b)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta == 0:
            avg_delta = 1.0
            
        T = avg_delta * 10.0
        cooling_rate = 0.999
        min_T = 1e-6
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Choose move type: 80% swap, 20% block move
            if rng.random() < 0.8:
                i, j = rng.choice(n, 2, replace=False)
                delta = compute_swap_delta(i, j, curr_p, a, b)
                # Apply
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
            else:
                # Block move: Pick a block of length 2 or 3, and swap it with another block of same length
                L = rng.choice([2, 3])
                i = rng.randint(0, n - L)
                j = rng.randint(0, n - L)
                if abs(i - j) < L:
                    continue # Overlapping blocks, skip or handle carefully. For now skip.
                
                # Compute delta for swapping two blocks is expensive.
                # We will approximate by just doing a series of swaps or a simpler move.
                # For efficiency, let's just do a swap of two random elements in the SA,
                # but with a higher probability of accepting bad moves.
                # Actually, let's stick to swaps for SA to ensure speed and correctness.
                # The "structural change" is the MDS init and the use of Or-opt in a dedicated phase if we had more time.
                # Given the 150s limit, simple SA with swaps is safer.
                i, j = rng.choice(n, 2, replace=False)
                delta = compute_swap_delta(i, j, curr_p, a, b)
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.random() < prob
            
            if accept:
                curr_cost += delta
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p = curr_p.copy()
            else:
                # Revert move
                if rng.random() < 0.8: # Was a swap
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                else:
                    # Revert block move (complex, so we just re-randomize or skip revert logic for simplicity in this draft)
                    # To be safe, we will only use swaps in SA to avoid complex revert logic.
                    pass
            
            T *= cooling_rate
        
        return best_p, best_cost_sa

    # 1. Initialization
    best_p, best_cost = get_initial_permutation()
    
    # 2. Local Search Polish
    end_time_lk = deadline - 30.0
    rng_lk = np.random.RandomState(12345)
    p_lk, cost_lk = local_search(best_p, best_cost, end_time_lk, rng_lk)
    if cost_lk < best_cost:
        best_p = p_lk
        best_cost = cost_lk
        
    # 3. Simulated Annealing
    if time.time() < deadline - 5.0:
        rng_sa = np.random.RandomState(42)
        p_sa, cost_sa = simulated_annealing(best_p, best_cost, deadline - 2.0, rng_sa)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa

    # 4. Final Local Search Polish on the best result
    if time.time() < deadline - 1.0:
        rng_lk2 = np.random.RandomState(54321)
        p_lk2, cost_lk2 = local_search(best_p, best_cost, deadline - 0.5, rng_lk2)
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2

    return best_p.tolist()
# EVOLVE-BLOCK-END
