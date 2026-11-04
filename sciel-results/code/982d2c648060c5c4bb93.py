"""Matrix multiplication 4,4,7 - rank minimization via structured decomposition."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    deadline = time.time() + 140  # Safety margin before 150s kill

    # Build target tensor: T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    a_idx_list = []
    b_idx_list = []
    c_idx_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx_list.append(i * m + j)
                b_idx_list.append(j * p + k)
                c_idx_list.append(k * n + i)
    
    a_idx = np.array(a_idx_list)
    b_idx = np.array(b_idx_list)
    c_idx = np.array(c_idx_list)
    n_entries = len(a_idx)  # 112
    
    # Precompute masks
    mask_U = [np.where(a_idx == a)[0] for a in range(nm)]
    mask_V = [np.where(b_idx == b)[0] for b in range(mp)]
    mask_W = [np.where(c_idx == c)[0] for c in range(pn)]

    def check_validity_full(U, V, W, tol=1e-6):
        """Full verification of the decomposition."""
        # Check non-zero entries
        for idx in range(n_entries):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        
        # Check zero entries
        nz_set = set(zip(a_idx.tolist(), b_idx.tolist(), c_idx.tolist()))
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in nz_set:
                        continue
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > tol:
                        return False
        return True

    def check_validity_fast(U, V, W, n_zero_samples=500, tol=1e-6):
        """Fast check: verify all non-zero entries and sample zero entries."""
        for idx in range(n_entries):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        
        nz_set = set(zip(a_idx.tolist(), b_idx.tolist(), c_idx.tolist()))
        rng = np.random.RandomState(42)
        for _ in range(n_zero_samples):
            a = rng.randint(0, nm)
            b = rng.randint(0, mp)
            c = rng.randint(0, pn)
            if (a, b, c) in nz_set:
                continue
            val = np.dot(U[a], V[b] * W[c])
            if abs(val) > tol:
                return False
        return True

    def als_iteration(U, V, W, rank, max_iter=200, tol=1e-10):
        """Run ALS iterations to minimize reconstruction error."""
        T_flat = np.ones(n_entries)
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Update U
            U_new = np.zeros((nm, rank))
            for a in range(nm):
                mask = mask_U[a]
                if len(mask) == 0:
                    U_new[a] = U[a]
                    continue
                B_m = b_idx[mask]
                C_m = c_idx[mask]
                # M_a[r, k] = V[B_m[k], r] * W[C_m[k], r]
                M_a = (V[B_m].T * W[C_m].T)  # (rank, n_a)
                try:
                    if M_a.shape[1] >= rank:
                        U_new[a], _, _, _ = np.linalg.lstsq(M_a.T, T_flat[mask], rcond=None)
                    else:
                        U_new[a] = np.linalg.pinv(M_a.T) @ T_flat[mask]
                except:
                    U_new[a] = U[a]
            U = U_new
            
            # Update V
            V_new = np.zeros((mp, rank))
            for b in range(mp):
                mask = mask_V[b]
                if len(mask) == 0:
                    V_new[b] = V[b]
                    continue
                A_m = a_idx[mask]
                C_m = c_idx[mask]
                M_b = (U[A_m].T * W[C_m].T)  # (rank, n_b)
                try:
                    if M_b.shape[1] >= rank:
                        V_new[b], _, _, _ = np.linalg.lstsq(M_b.T, T_flat[mask], rcond=None)
                    else:
                        V_new[b] = np.linalg.pinv(M_b.T) @ T_flat[mask]
                except:
                    V_new[b] = V[b]
            V = V_new
            
            # Update W
            W_new = np.zeros((pn, rank))
            for c in range(pn):
                mask = mask_W[c]
                if len(mask) == 0:
                    W_new[c] = W[c]
                    continue
                A_m = a_idx[mask]
                B_m = b_idx[mask]
                M_c = (U[A_m].T * V[B_m].T)  # (rank, n_c)
                try:
                    if M_c.shape[1] >= rank:
                        W_new[c], _, _, _ = np.linalg.lstsq(M_c.T, T_flat[mask], rcond=None)
                    else:
                        W_new[c] = np.linalg.pinv(M_c.T) @ T_flat[mask]
                except:
                    W_new[c] = W[c]
            W = W_new
            
            # Early stopping
            if it % 10 == 0:
                err = 0.0
                for idx in range(n_entries):
                    a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
                    val = np.dot(U[a], V[b] * W[c])
                    err += (val - 1.0)**2
                if err < tol:
                    break
        
        return U, V, W

    def build_strassen_schoolbook_hybrid():
        """
        Build a hybrid decomposition using Strassen for the 4x4x4 subproblem
        and schoolbook for the remaining parts.
        
        For 4x4 * 4x7:
        - Split the 7 columns into 4 + 3
        - Use Strassen (rank 7) for the 4x4 * 4x4 part
        - Use schoolbook (rank 12) for the 4x4 * 4x3 part
        - Total: 7 + 12 = 19 per row-block? No, need to think more carefully.
        
        Actually, let's use a different approach:
        - The full problem is (4x4) * (4x7) = 4x7
        - We can decompose this as:
          - 4x4 * 4x4 (first 4 columns): use Strassen rank 7
          - 4x4 * 4x3 (last 3 columns): use schoolbook rank 12
        - But Strassen for 4x4x4 has rank 7? No, standard Strassen is 2x2x2 rank 7.
        - For 4x4x4, we can tile 2x2 blocks: 4 blocks of 2x2x2, each rank 7 -> total 28
        - Or use a better algorithm. Let's use the known rank-28 algorithm for 4x4x4.
        
        Let me use a simpler approach: just use schoolbook with some randomization
        and then run ALS to reduce the rank.
        """
        pass  # Not using this, will use ALS from scratch

    # Start with schoolbook as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find lower-rank decompositions using ALS
    # Start from rank 100 down to 90
    for target_rank in range(100, 89, -1):
        if time.time() > deadline - 25:
            break
        
        found = False
        n_restarts = 5
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            # Initialize
            if restart == 0:
                # Warm start from schoolbook (take first target_rank columns)
                U = U_school[:, :target_rank].copy()
                V = V_school[:, :target_rank].copy()
                W = W_school[:, :target_rank].copy()
                # Add small noise to break symmetry
                U += np.random.randn(nm, target_rank) * 0.1
                V += np.random.randn(mp, target_rank) * 0.1
                W += np.random.randn(pn, target_rank) * 0.1
            else:
                # Random initialization
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            
            # Run ALS
            U, V, W = als_iteration(U, V, W, target_rank, max_iter=300)
            
            # Quick check
            if check_validity_fast(U, V, W):
                # Full check
                if check_validity_full(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Continue to try lower ranks, but we have a valid one
            continue
        else:
            # If we fail at this rank, it might still be possible at lower ranks
            # but unlikely. Stop after 3 consecutive failures.
            if target_rank <= 97:  # Failed at 97, 98, 99
                break

    # Final polish: try to verify and return
    if check_validity_full(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return schoolbook as fallback
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
