"""QAP solver for tho150: MDS/Procrustes Initialization + Simulated Annealing Polish."""
import numpy as np
import time
from scipy.spatial.distance import squareform


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64, copy=False)
    b = b.astype(np.int64, copy=False)

    def compute_cost(p_arr):
        """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
        bp = b[p_arr]  # bp[i,j] = b[p[i], p[j]]
        return int(np.sum(a * bp))

    def mds_procrustes_init():
        """
        Initialize permutation using MDS on the distance matrix B and Procrustes alignment with A.
        """
        # MDS on B
        # B is a distance matrix (presumably).
        # If B is not a metric distance matrix, we might need to be careful.
        # QAPLIB instances typically provide distance matrices.
        # Let's assume b is a distance matrix.
        
        # Standard MDS:
        # 1. Double centering
        # J = I - (1/n) * 11^T
        # M = -0.5 * J @ B^2 @ J
        # 2. Eigen decomposition
        # Take top n-1 or n-2 eigenvectors
        # 3. Coordinates
        
        # To be safe and fast, we use scipy's mds if available, or manual implementation.
        # Manual implementation is often faster for small n and avoids overhead.
        
        # Double centering
        J = np.eye(n) - np.ones((n, n)) / n
        B2 = b ** 2
        M = -0.5 * (J @ B2 @ J)
        
        # Eigen decomposition
        # We only need the top few eigenvectors
        # np.linalg.eigh is slow for large n, but n=150 is small.
        eigvals, eigvecs = np.linalg.eigh(M)
        
        # Sort by eigenvalues descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Use top 2 or 3 dimensions for Procrustes
        # 2D is usually sufficient for QAP, 3D might be slightly better
        k = min(3, n-1)
        X = eigvecs[:, :k] * np.sqrt(eigvals[:k])
        
        # Procrustes alignment
        # We want to align X (locations) with A (facilities)
        # We treat A as a "distance" matrix too? Or use it to define facility positions?
        # A is a flow matrix. We can compute MDS on A as well to get facility coordinates.
        
        # Let's compute MDS on A
        M_a = -0.5 * (J @ (a ** 2) @ J)
        eigvals_a, eigvecs_a = np.linalg.eigh(M_a)
        idx_a = np.argsort(eigvals_a)[::-1]
        eigvals_a = eigvals_a[idx_a]
        eigvecs_a = eigvecs_a[:, idx_a]
        
        Y = eigvecs_a[:, :k] * np.sqrt(eigvals_a[:k])
        
        # Procrustes: Find rotation R such that Y @ R approx X
        # Minimize || Y @ R - X ||_F
        # R = V U^T from SVD of Y^T X
        
        # SVD of Y^T X
        U, s, Vt = np.linalg.svd(Y.T @ X)
        R = U @ Vt
        
        # Rotate Y
        Y_rot = Y @ R.T
        
        # Assign facilities to locations by matching rows
        # We need a permutation p such that facility i is at location p[i]
        # We can use Hungarian algorithm on the distance between Y_rot and X
        # Or simply sort by coordinates?
        # Better: Use scipy.optimize.linear_sum_assignment
        
        from scipy.optimize import linear_sum_assignment
        
        # Cost matrix for assignment: distance between facility coords and location coords
        # D[i, j] = || Y_rot[i] - X[j] ||^2
        # Y_rot shape (n, k), X shape (n, k)
        # D = ||Y_rot||^2 + ||X||^2 - 2 * Y_rot @ X^T
        
        Y_rot_sq = np.sum(Y_rot ** 2, axis=1)[:, None]
        X_sq = np.sum(X ** 2, axis=1)[None, :]
        D = Y_rot_sq + X_sq - 2 * (Y_rot @ X.T)
        
        row_ind, col_ind = linear_sum_assignment(D)
        
        # p[facility] = location
        # row_ind is facilities, col_ind is locations
        p = np.zeros(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    def sa_polish(p_arr, cur_cost, time_limit):
        """
        Simulated Annealing polish with 2-swap moves.
        """
        # Precompute row sums of B for faster delta?
        # Delta for swapping i and j:
        # Let p be the current permutation.
        # Swap p[i] and p[j]. Let u = p[i], v = p[j].
        # New cost = Old cost - (terms involving u, v with old positions) + (terms involving u, v with new positions)
        
        # Terms involving i at u and j at v:
        # Old: a[i,i]b[u,u] + a[j,j]b[v,v] + a[i,j]b[u,v] + a[j,i]b[v,u]
        #      + sum_{k!=i,j} (a[i,k]b[u,p[k]] + a[k,i]b[p[k],u] + a[j,k]b[v,p[k]] + a[k,j]b[p[k],v])
        
        # New (i at v, j at u):
        # New: a[i,i]b[v,v] + a[j,j]b[u,u] + a[i,j]b[v,u] + a[j,i]b[u,v]
        #      + sum_{k!=i,j} (a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] + a[j,k]b[u,p[k]] + a[k,j]b[p[k],u])
        
        # Delta = New - Old
        # Delta = a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u])
        #        + sum_{k!=i,j} [ a[i,k](b[v,p[k]]-b[u,p[k]]) + a[k,i](b[p[k],v]-b[p[k],u])
        #                       + a[j,k](b[u,p[k]]-b[v,p[k]]) + a[k,j](b[p[k],u]-b[p[k],v]) ]
        
        # Let's precompute some vectors to speed this up.
        # For a fixed pair (i, j) with locations u, v:
        # diff_b_row[k] = b[v, p[k]] - b[u, p[k]]
        # diff_b_col[k] = b[p[k], v] - b[p[k], u]
        
        # sum_k a[i,k] * diff_b_row[k]
        # sum_k a[k,i] * diff_b_col[k]
        # sum_k a[j,k] * (-diff_b_row[k])  # since b[u]-b[v] = -(b[v]-b[u])
        # sum_k a[k,j] * (-diff_b_col[k])
        
        # So sum part = dot(a[i], diff_b_row) + dot(a[:,i], diff_b_col) - dot(a[j], diff_b_row) - dot(a[:,j], diff_b_col)
        #             = dot(a[i]-a[j], diff_b_row) + dot(a[:,i]-a[:,j], diff_b_col)
        
        # We must exclude k=i and k=j from the sum.
        # In the dot products, k=i and k=j are included. We need to subtract their contributions.
        
        # For k=i:
        # diff_b_row[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
        # diff_b_col[i] = b[p[i], v] - b[p[i], u] = b[u, v] - b[u, u]
        # Term_i = (a[i,i]-a[j,i]) * (b[v,u]-b[u,u]) + (a[i,i]-a[i,j]) * (b[u,v]-b[u,u])
        
        # For k=j:
        # diff_b_row[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
        # diff_b_col[j] = b[p[j], v] - b[p[j], u] = b[v, v] - b[v, u]
        # Term_j = (a[i,j]-a[j,j]) * (b[v,v]-b[u,v]) + (a[j,i]-a[j,j]) * (b[v,v]-b[v,u])
        
        # Total Delta = (Self terms) + (Sum part) - Term_i - Term_j
        
        # Self terms:
        # a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u])
        
        # This is O(n) per swap evaluation if we precompute diff vectors?
        # No, diff_b_row depends on p[k] for all k.
        # So we can't precompute diff_b_row for all k globally.
        # But for a specific swap (i,j), we can compute diff_b_row and diff_b_col in O(n).
        # Then the dot products are O(n).
        # So each swap evaluation is O(n).
        # Total O(n^3) per full pass of SA?
        # n=150, n^3 = 3.3M ops. Very fast.
        
        # Let's implement the O(n) delta calculation.
        
        T = 1000.0 # Initial temperature
        cooling = 0.9995
        min_T = 0.5
        p = p_arr.copy()
        cost = cur_cost
        
        # Precompute a_rows and a_cols for faster access?
        # a is int64, so dot product is exact.
        
        while time.time() < time_limit:
            if T < min_T:
                break
            
            # Pick random i, j
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            
            u = p[i]
            v = p[j]
            
            # Compute Delta
            # diff_b_row[k] = b[v, p[k]] - b[u, p[k]]
            # diff_b_col[k] = b[p[k], v] - b[p[k], u]
            
            # We can compute these vectors efficiently
            # b[v, :] - b[u, :] is constant vector for rows? No, p[k] varies.
            # b[v, p] is a vector of length n.
            # b[u, p] is a vector of length n.
            diff_b_row = b[v, p] - b[u, p]
            
            # b[:, v] - b[:, u] is constant vector for cols? No, p[k] varies.
            # b[p, v] is a vector of length n.
            # b[p, u] is a vector of length n.
            diff_b_col = b[p, v] - b[p, u]
            
            # Sum part
            sum_part = np.dot(a[i] - a[j], diff_b_row) + np.dot(a[:, i] - a[:, j], diff_b_col)
            
            # Subtract k=i and k=j terms
            # Term_i
            t_i_row = b[v, u] - b[u, u]
            t_i_col = b[u, v] - b[u, u]
            term_i = (a[i, i] - a[j, i]) * t_i_row + (a[i, i] - a[i, j]) * t_i_col
            
            # Term_j
            t_j_row = b[v, v] - b[u, v]
            t_j_col = b[v, v] - b[v, u]
            term_j = (a[i, j] - a[j, j]) * t_j_row + (a[j, i] - a[j, j]) * t_j_col
            
            # Self terms
            self_term = (a[i, i] * (b[v, v] - b[u, u]) + 
                         a[j, j] * (b[u, u] - b[v, v]) + 
                         a[i, j] * (b[v, u] - b[u, v]) + 
                         a[j, i] * (b[u, v] - b[v, u]))
            
            delta = self_term + sum_part - term_i - term_j
            
            # Acceptance
            if delta < 0:
                p[i], p[j] = v, u
                cost += delta
            else:
                # Metropolis criterion
                # exp(-delta / T)
                # Use log for stability? delta is int, T is float.
                # If delta is large, exp is 0.
                if np.random.rand() < np.exp(-delta / T):
                    p[i], p[j] = v, u
                    cost += delta
            
            T *= cooling
            
        return p, cost

    # --- Main Algorithm ---
    
    # Phase 1: MDS/Procrustes Initialization
    p = mds_procrustes_init()
    cur_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = cur_cost
    
    # Phase 2: SA Polish
    p, cur_cost = sa_polish(p, cur_cost, deadline - 20)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_p = p.copy()
    
    # Phase 3: Restart with Perturbation
    if time.time() < deadline - 10:
        p = best_p.copy()
        # Perturb
        for _ in range(10):
            i, j = np.random.randint(0, n, 2)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        cur_cost = compute_cost(p)
        p, cur_cost = sa_polish(p, cur_cost, deadline - 5)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()
            
    return best_p.tolist()
