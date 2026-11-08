import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    """Rank 17 decomposition for 2x3x3 matrix multiplication."""
    n, m, p = 2, 3, 3
    R = 17
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Part 2: schoolbook A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Part 3: schoolbook A[0:2,0:3] * B[0:3,2] -> C[0:2,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    n, m, p = 2, 3, 3
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank16_linear_search(deadline):
    """
    Search for rank-16 decomposition by trying all subsets of 16 from the 18 
    'natural' rank-1 terms (each corresponding to one tensor component).
    
    The 18 natural rank-1 terms are: for each (i,j,k), term = e_i ⊗ e_j ⊗ e_k
    which gives U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    
    We check if any 16 of these can be combined (with coefficients) to reproduce 
    the full tensor. This is a linear algebra problem: can we find coefficients 
    such that the sum of 16 rank-1 tensors equals the all-ones tensor?
    """
    n, m, p = 2, 3, 3
    # The 18 natural rank-1 terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_t = np.zeros(n*m)
                V_t = np.zeros(m*p)
                W_t = np.zeros(p*n)
                U_t[i*m + j] = 1.0
                V_t[j*p + k] = 1.0
                W_t[k*n + i] = 1.0
                terms.append((U_t, V_t, W_t))
    
    # Try all subsets of 16 terms out of 18
    # There are C(18,16) = 153 subsets
    for excluded in combinations(range(18), 2):
        if time.time() > deadline:
            break
        included = [t for t in range(18) if t not in excluded]
        
        # Set up linear system: sum_{r in included} c_r * (U_r ⊗ V_r ⊗ W_r) = T
        # where T[i,j,k] = 1 for all i,j,k
        # This is a linear system in the coefficients c_r
        
        # Build the 18x16 matrix M where M[idx, r] = term_r[idx]
        # idx maps (i,j,k) to a single index
        def idx(i, j, k):
            return i*m*p + j*p + k
        
        M = np.zeros((18, len(included)))
        for r_idx, term_idx in enumerate(included):
            U_t, V_t, W_t = terms[term_idx]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[idx(i,j,k), r_idx] = U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i]
        
        # Target: all ones
        b = np.ones(18)
        
        # Solve least squares (overdetermined: 18 eqs, 16 unknowns)
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if solution is valid (residuals should be ~0)
        if len(residuals) > 0 and residuals[0] < 1e-8:
            # Verify the solution
            U = np.zeros((n*m, 16))
            V = np.zeros((m*p, 16))
            W = np.zeros((p*n, 16))
            for r_idx, term_idx in enumerate(included):
                U_t, V_t, W_t = terms[term_idx]
                U[:, r_idx] = coeffs[r_idx] * U_t
                V[:, r_idx] = coeffs[r_idx] * V_t
                W[:, r_idx] = coeffs[r_idx] * W_t
            
            if _verify(U, V, W):
                return U, V, W
    
    return None


def _try_rank16_from_rank17(deadline):
    """
    Try to find rank 16 by modifying the rank-17 decomposition.
    Strategy: for each term in the rank-17 decomposition, try to express it
    as a combination of the other 16 terms plus one new term that cancels out.
    
    More practically: try zeroing out each term and solving for adjustments
    to the remaining terms.
    """
    n, m, p = 2, 3, 3
    U17, V17, W17 = _build_rank17()
    
    for drop_idx in range(17):
        if time.time() > deadline:
            break
        
        # Keep all terms except drop_idx
        mask = np.ones(17, dtype=bool)
        mask[drop_idx] = False
        U_rem = U17[:, mask]
        V_rem = V17[:, mask]
        W_rem = W17[:, mask]
        
        # The remaining 16 terms produce some tensor T_rem
        # We need: T_rem + adjustment = T_full (all ones)
        # The adjustment must be a rank-1 tensor
        # So: T_full - T_rem must be rank 1
        
        T_full = np.ones((n, m, p))
        T_rem = np.zeros((n, m, p))
        for r in range(16):
            Ur = U_rem[:, r].reshape(n, m)
            Vr = V_rem[:, r].reshape(m, p)
            Wr = W_rem[:, r].reshape(p, n)
            # T_rem[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_rem[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
        
        diff = T_full - T_rem
        
        # Check if diff is rank 1 (matrix flattening)
        # A 3D tensor of shape (n,m,p) is rank 1 if it can be written as a⊗b⊗c
        # Check via unfolding: reshape to (n, m*p) and check rank
        flat = diff.reshape(n, m*p)
        try:
            rank_flat = np.linalg.matrix_rank(flat, tol=1e-9)
        except np.linalg.LinAlgError:
            continue
        
        if rank_flat == 1:
            # It's rank 1! Find the decomposition
            # diff[i,j,k] = a[i] * b[j] * c[k]
            # From flat: diff[i, j*p+k] = a[i] * (b[j]*c[k])
            # Take first non-zero column
            col_sums = np.sum(np.abs(flat), axis=0)
            if np.all(col_sums == 0):
                continue
            # Find a column with non-zero entries
            nonzero_cols = np.where(col_sums > 1e-12)[0]
            if len(nonzero_cols) == 0:
                continue
            col_idx = nonzero_cols[0]
            col = flat[:, col_idx]
            
            # a[i] is proportional to col[i]
            # Scale so that max |a| = 1
            max_a = np.max(np.abs(col))
            if max_a < 1e-12:
                continue
            a = col / max_a
            
            # b[j]*c[k] = flat[i, j*p+k] / a[i] for any i with a[i] != 0
            valid_i = np.where(np.abs(a) > 1e-12)[0]
            if len(valid_i) == 0:
                continue
            i_ref = valid_i[0]
            bc_flat = flat[i_ref] / a[i_ref]
            bc = bc_flat.reshape(m, p)
            
            # Factor bc = b * c^T
            # Use SVD
            try:
                U_bc, S, Vt_bc = np.linalg.svd(bc, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            
            if len(S) > 0 and S[0] > 1e-12:
                # Check if bc is actually rank 1
                if len(S) < 2 or S[1] < 1e-9:
                    b = U_bc[:, 0] * np.sqrt(S[0])
                    c = Vt_bc[0, :] * np.sqrt(S[0])
                    
                    # Construct the rank-16 solution
                    # The 16th term has: U_new = a, V_new = b, W_new = c
                    # But we need to be careful with the trace convention
                    # W[k*n+i] = c[k] for the new term... wait
                    # Actually W is indexed as W[k*n+i, r], and for rank-1:
                    # W[k*n+i] = c_k * a_i... no
                    # Let me reconsider.
                    
                    # For a rank-1 term: U[i*m+j] = u_i * v_j (but U is just a vector)
                    # Actually in our formulation, each rank-1 term has:
                    # U_r[i*m+j], V_r[j*p+k], W_r[k*n+i]
                    # And the product U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] should equal diff[i,j,k]
                    
                    # So we need: U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = a[i]*b[j]*c[k]
                    # This works if: U_r[i*m+j] = a[i]*b[j], V_r[j*p+k] = 1, W_r[k*n+i] = c[k]
                    # Or: U_r[i*m+j] = a[i], V_r[j*p+k] = b[j]*c[k], W_r[k*n+i] = 1
                    # Or: U_r[i*m+j] = a[i]*b[j], V_r[j*p+k] = c[k], W_r[k*n+i] = 1
                    # etc.
                    
                    # Let's use: U_new[i*m+j] = a[i]*b[j], V_new[j*p+k] = c[k], W_new[k*n+i] = 1
                    U_new = np.zeros(n*m)
                    V_new = np.zeros(m*p)
                    W_new = np.zeros(p*n)
                    for i in range(n):
                        for j in range(m):
                            U_new[i*m+j] = a[i] * b[j]
                    for j in range(m):
                        for k in range(p):
                            V_new[j*p+k] = c[k]
                    for k in range(p):
                        for i in range(n):
                            W_new[k*n+i] = 1.0
                    
                    # Build the full rank-16 matrices
                    U16 = np.zeros((n*m, 16))
                    V16 = np.zeros((m*p, 16))
                    W16 = np.zeros((p*n, 16))
                    for r in range(16):
                        U16[:, r] = U_rem[:, r]
                        V16[:, r] = V_rem[:, r]
                        W16[:, r] = W_rem[:, r]
                    U16[:, 15] = U_new
                    V16[:, 15] = V_new
                    W16[:, 15] = W_new
                    
                    if _verify(U16, V16, W16):
                        return U16, V16, W16
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Get baseline rank 17
    U17, V17, W17 = _build_rank17()
    
    # Verify baseline
    if not _verify(U17, V17, W17):
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    best_rank = 17
    best_U, best_V, best_W = U17, V17, W17
    
    # Try to find rank 16 via linear search on natural terms
    if time.time() < deadline - 10:
        result = _try_rank16_linear_search(deadline)
        if result is not None:
            best_rank = 16
            best_U, best_V, best_W = result
    
    # Try to find rank 16 by modifying rank-17 decomposition
    if best_rank > 16 and time.time() < deadline - 10:
        result = _try_rank16_from_rank17(deadline)
        if result is not None:
            best_rank = 16
            best_U, best_V, best_W = result
    
    # Final verification
    if not _verify(best_U, best_V, best_W):
        U17, V17, W17 = _build_rank17()
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
