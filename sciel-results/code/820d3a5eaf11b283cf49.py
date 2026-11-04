"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

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
        Generate initial permutations using MDS/Procrustes and Hungarian linearization.
        Returns the best initial permutation and its cost.
        """
        best_p = None
        best_cost = float('inf')
        
        # Method 1: MDS + Procrustes
        try:
            A_dist = -a + np.max(a)
            X_embed = mds_embed(A_dist, k=2)
            Y_embed = mds_embed(b, k=2)
            X_aligned = procrustes_align(X_embed, Y_embed)
            
            X_sq = np.sum(X_aligned ** 2, axis=1)[:, None]
            Y_sq = np.sum(Y_embed ** 2, axis=1)[None, :]
            dist_matrix = X_sq + Y_sq - 2 * (X_aligned @ Y_embed.T)
            
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            p_mds = np.empty(n, dtype=np.int64)
            p_mds[row_ind] = col_ind
            
            cost_mds = compute_cost(p_mds)
            if cost_mds < best_cost:
                best_cost = cost_mds
                best_p = p_mds.copy()
                
        except Exception:
            pass

        # Method 2: Hungarian on linearized QAP objective
        # C[i,j] = sum_k a[i,k] * b[j,k]
        # We want to minimize sum_i C[i, p[i]], so solve LAP on C
        try:
            C = a @ b.T  # C[i,j] = sum_k a[i,k]*b[j,k]
            row_ind, col_ind = linear_sum_assignment(C)
            p_hung = np.empty(n, dtype=np.int64)
            p_hung[row_ind] = col_ind
            
            cost_hung = compute_cost(p_hung)
            if cost_hung < best_cost:
                best_cost = cost_hung
                best_p = p_hung.copy()
                
        except Exception:
            pass

        # Method 3: Hungarian on column-wise linearization
        # C2[i,j] = sum_k a[k,i] * b[k,j]
        try:
            C2 = a.T @ b  # C2[i,j] = sum_k a[k,i]*b[k,j]
            row_ind, col_ind = linear_sum_assignment(C2)
            p_hung2 = np.empty(n, dtype=np.int64)
            p_hung2[row_ind] = col_ind
            
            cost_hung2 = compute_cost(p_hung2)
            if cost_hung2 < best_cost:
                best_cost = cost_hung2
                best_p = p_hung2.copy()
                
        except Exception:
            pass

        # Fallback: Random
        if best_p is None:
            rng = np.random.RandomState(42)
            best_p = rng.permutation(n)
            best_cost = compute_cost(best_p)

        return best_p, best_cost

    # Precompute for fast swap deltas
    # For swap of positions i,j with locations li=p[i], lj=p[j]:
    # delta = sum_k [a[i,k]*(b[lj,pk]-b[li,pk]) + a[k,i]*(b[pk,lj]-b[pk,li])
    #         + a[j,k]*(b[li,pk]-b[lj,pk]) + a[k,j]*(b[pk,li]-b[pk,lj])]
    # We can factor: for each facility k (k!=i,j), pk=p[k]:
    #   a[i,k]*(b[lj,pk]-b[li,pk]) + a[k,i]*(b[pk,lj]-b[pk,li])
    #   + a[j,k]*(b[li,pk]-b[lj,pk]) + a[k,j]*(b[pk,li]-b[pk,lj])
    # = (a[i,k]-a[j,k])*(b[lj,pk]-b[li,pk]) + (a[k,i]-a[k,j])*(b[pk,lj]-b[pk,li])
    # This is still O(n) per swap but with fewer ops.
    
    def compute_swap_delta_fast(i, j, p):
        """
        Fast cost delta for swapping facilities at positions i and j.
        Uses precomputed a, b arrays.
        """
        li = p[i]
        lj = p[j]
        
        # Create mask excluding i and j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        pk = p[ks]
        
        # Vectorized: (a[i,ks]-a[j,ks])*(b[lj,pk]-b[li,pk]) + (a[ks,i]-a[ks,j])*(b[pk,lj]-b[pk,li])
        d1 = (a[i, ks] - a[j, ks]) * (b[lj, pk] - b[li, pk])
        d2 = (a[ks, i] - a[ks, j]) * (b[pk, lj] - b[pk, li])
        
        delta = np.sum(d1) + np.sum(d2)
        
        # Diagonal and cross terms
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
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
        First-improvement 2-swap local search with fast delta computation.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        # Pre-generate all pairs once for reuse
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() < end_time:
            improved = False
            rng.shuffle(all_pairs)
            
            for i, j in all_pairs:
                if time.time() > end_time:
                    break
                
                delta = compute_swap_delta_fast(i, j, curr_p)
                
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
        Simulated Annealing with 2-swaps using fast delta.
        """
        curr_p = p.copy()
        curr_cost = best_cost
        best_p = p.copy()
        best_cost_sa = best_cost
        
        # Estimate initial temperature
        deltas = []
        for _ in range(20):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta_fast(i, j, curr_p)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta == 0:
            avg_delta = 1.0
            
        T = avg_delta * 10.0
        cooling_rate = 0.9995
        min_T = 1e-4
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            i, j = rng.choice(n, 2, replace=False)
            delta = compute_swap_delta_fast(i, j, curr_p)
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
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
            
            T *= cooling_rate
        
        return best_p, best_cost_sa

    # 1. Initialization
    best_p, best_cost = get_initial_permutation()
    
    # 2. Local Search Polish (generous time budget since delta is faster)
    end_time_lk = deadline - 25.0
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
