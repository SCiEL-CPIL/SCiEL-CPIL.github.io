"""Bootstrap MODEL-AND-OPTIMIZE rewrite: parameterize the tensor decomposition
and verify it exactly, then attempt a small search to reduce rank.

Dimensions: A is (n x m) = (2 x 3), B is (m x p) = (3 x 4), C = A B is (n x p) = (2 x 4).
Convention: U index = i*m+j (A entry), V index = j*p+k (B entry), W index = k*n+i (C entry).
Decomposition: C = sum_r U[:,r] * V[:,r] * W[:,r]  (outer product along r).
Score = rank (number of terms). Lower is better.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_BUDGET = 140.0  # evaluator kills at 150s; keep 10s margin


def _tensor_index(n, m, p):
    """Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == T exactly (within tol)."""
    rank = U.shape[1]
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(reconstructed, T, atol=tol, rtol=0.0)


def _try_reduce(U, V, W, T, max_attempts=200):
    """Attempt to find a lower-rank decomposition by identifying linear
    dependencies among the rank terms.

    Strategy:
    1. Check for proportional terms (scalar multiples) - merge them.
    2. For each term r, check if removing it and solving for adjusted
       coefficients of remaining terms still reconstructs T.
    3. Use matricization rank check to find the true rank bound.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]

    # Step 1: Try merging proportional terms
    merged = True
    while merged and rank > 1:
        merged = False
        for r in range(rank):
            for s in range(r + 1, rank):
                if np.linalg.norm(U[:, r]) < 1e-12 or np.linalg.norm(U[:, s]) < 1e-12:
                    continue
                if np.linalg.norm(V[:, r]) < 1e-12 or np.linalg.norm(V[:, s]) < 1e-12:
                    continue
                ratio_u = np.dot(U[:, r], U[:, s]) / np.dot(U[:, s], U[:, s])
                if (np.allclose(U[:, r], ratio_u * U[:, s], atol=1e-12) and
                        np.allclose(V[:, r], ratio_u * V[:, s], atol=1e-12) and
                        np.allclose(W[:, r], ratio_u * W[:, s], atol=1e-12)):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    merged = True
                    break
            if merged:
                break

    # Step 2: Try removing each term and checking if the remaining terms
    # can still span the tensor via a linear system solve.
    # The tensor T (flattened) must equal sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r].
    # If we remove term r, we need to find new coefficients for remaining terms.
    # This is equivalent to checking if T is in the span of {U[:,s]⊗V[:,s]⊗W[:,s] : s≠r}.
    
    n_rows_u = U.shape[0]
    n_rows_v = V.shape[0]
    n_rows_w = W.shape[0]
    
    # Build the "flattened" representation: each term r maps to a vector of size
    # n_rows_u * n_rows_v * n_rows_w. But this is too large (6*12*8=576).
    # Instead, use the fact that T has only 24 non-zero entries.
    
    # For each term r, the contribution to T[i*m+j, j*p+k, k*n+i] is
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # T has 24 non-zero entries (one per (i,j,k) triple).
    
    # Build a system: for each non-zero entry (a,b,c) of T, we have
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a system of 24 equations in rank unknowns (but the products are
    # bilinear/cubic, not linear). So we can't directly solve linearly.
    
    # Alternative: fix all but one term's coefficients and solve.
    # If we remove term r, we have rank-1 terms. We need to check if T is in
    # their span. Since the decomposition is overdetermined (24 eqs, rank-1 unknowns
    # but nonlinear), let's try a different approach.
    
    # Try: for each pair of terms (r, s), check if term r can be expressed as
    # a linear combination of term s and others by checking if the "residual"
    # after removing r is still representable.
    
    # Simpler heuristic: check if any term has zero contribution to all 24 entries
    # (i.e., for every (i,j,k), at least one of U[i*m+j,r], V[j*p+k,r], W[k*n+i,r] is 0)
    n, m, p = T.shape  # T is (n*m, m*p, p*n)
    for r in range(rank):
        all_zero = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if U[a, r] != 0 and V[b, r] != 0 and W[c, r] != 0:
                        all_zero = False
                        break
                if not all_zero:
                    break
            if not all_zero:
                break
        if all_zero:
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            break  # restart loop since indices changed

    # Step 3: Try removing each term and solving for the residual.
    # For each term r to remove, check if T can be written as sum_{s≠r} c_s * (U[:,s]⊗V[:,s]⊗W[:,s])
    # This is nonlinear, but we can check: is the vectorized T in the span of
    # {vec(U[:,s]⊗V[:,s]⊗W[:,s]) : s≠r}?
    # vec of outer product: for each s, the vector is U[:,s] ⊗ V[:,s] ⊗ W[:,s]
    # which has size (n*m)*(m*p)*(p*n) = 6*12*8 = 576.
    # But T is sparse (only 24 non-zeros), so we only need to check those 24 entries.
    
    # For each entry (a,b,c) where T[a,b,c]=1, we need:
    # sum_{s≠r} U[a,s]*V[b,s]*W[c,s] = 1
    # This is a system of 24 equations in (rank-1) unknowns, but the equations
    # are cubic (product of three variables from different terms).
    
    # Alternative: fix U and V, solve for W. For each c, W[c,:] must satisfy
    # for all (a,b) with T[a,b,c]≠0: sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c]
    # But W[c,s] appears linearly for fixed c! So for each c, we have a linear system.
    
    # Let's try: remove term r, then for each c index, solve for W[c, s≠r].
    # The system is: for each (a,b) where T[a,b,c]=1:
    #   sum_{s≠r} U[a,s]*V[b,s]*W[c,s] = 1
    # This is linear in W[c,:].
    
    for r_remove in range(rank):
        if time.time() - START_TIME > TIME_BUDGET - 15:
            break
        U_rem = np.delete(U, r_remove, axis=1)
        V_rem = np.delete(V, r_remove, axis=1)
        W_rem = np.delete(W, r_remove, axis=1).copy()
        r_new = rank - 1
        
        # For each c, solve linear system for W_rem[c, :]
        valid = True
        for c in range(n * m * p):  # c ranges over p*n = 8 values
            # Find all (a,b) such that T[a,b,c] != 0
            # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
            # So c = k*n+i. For given c, k = c//n, i = c%n.
            # Then b = j*p+k for j in 0..m-1, and a = i*m+j for j in 0..m-1.
            # So for given c, the non-zero entries are at (a=i*m+j, b=j*p+k) for j=0..m-1
            # where i = c%n, k = c//n.
            i_c = c % n
            k_c = c // n
            if k_c >= p:
                continue
            # Equations: for each j in 0..m-1:
            #   sum_{s≠r} U[i_c*m+j, s] * V[j*p+k_c, s] * W_rem[c, s] = 1
            # This is: A_c * W_rem[c,:] = 1 where A_c[s] = U[i_c*m+j,s]*V[j*p+k_c,s]
            # But wait, this gives m equations (one per j) in r_new unknowns (W_rem[c,s]).
            # The equations are: for j=0..m-1: sum_s U[i_c*m+j, s]*V[j*p+k_c, s]*W_rem[c,s] = 1
            
            # Build the system: m equations, r_new unknowns
            if r_new < m:
                # Underdetermined, may have solutions
                pass
            # Actually, we need ALL m equations to be satisfied simultaneously.
            # Let's build the matrix and check consistency.
            
            A_sys = np.zeros((m, r_new))
            b_sys = np.ones(m)
            for j in range(m):
                a_idx = i_c * m + j
                b_idx = j * p + k_c
                A_sys[j, :] = U_rem[a_idx, :] * V_rem[b_idx, :]
            
            # Solve A_sys * W_rem[c,:] = b_sys
            # Use least squares and check residual
            sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            if residuals is not None and len(residuals) > 0:
                if residuals[0] > 1e-8:
                    valid = False
                    break
            else:
                # Check directly
                if not np.allclose(A_sys @ sol, b_sys, atol=1e-8):
                    valid = False
                    break
            
            W_rem[c, :] = sol
        
        if valid:
            # Verify the full decomposition
            U_new = U_rem
            V_new = V_rem
            W_new = W_rem
            if _verify(U_new, V_new, W_new, T):
                U = U_new
                V = V_new
                W = W_new
                rank = r_new
                break  # Successfully reduced, try again with new rank

    return U, V, W


def _build_rank22_decomposition(n, m, p):
    """Build the rank-22 Strassen-based decomposition for 2x3 x 3x4.

    Structure:
      C = A_L * B_T1 + A_L * B_T2 + A_R * B_B
    where:
      A_L = A[:, 0:2] (2x2), A_R = A[:, 2:3] (2x1)
      B_T1 = B[0:2, 0:2] (2x2), B_T2 = B[0:2, 2:4] (2x2), B_B = B[2:3, :] (1x4)

    Strassen (rank 7) for each 2x2 block, plus rank-8 schoolbook for 2x1 x 1x4.
    Total: 7 + 7 + 8 = 22.
    """
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def _add_strassen_block(a_idx, b_idx, c_idx, col_start):
        """Add a Strassen rank-7 decomposition for a 2x2x2 block.

        a_idx: dict with keys '00','01','10','11' -> U row indices
        b_idx: dict with keys '00','01','10','11' -> V row indices
        c_idx: dict with keys '00','01','10','11' -> W row indices
        col_start: starting column in U,V,W
        """
        # Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        col = col_start
        # M1
        U[a_idx['00']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M2
        U[a_idx['10']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        W[c_idx['11']][col] += -1
        col += 1
        # M3
        U[a_idx['00']][col] += 1
        V[b_idx['01']][col] += 1; V[b_idx['11']][col] += -1
        W[c_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M4
        U[a_idx['11']][col] += 1
        V[b_idx['10']][col] += 1; V[b_idx['00']][col] += -1
        W[c_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        col += 1
        # M5
        U[a_idx['00']][col] += 1; U[a_idx['01']][col] += 1
        V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += -1
        W[c_idx['01']][col] += 1
        col += 1
        # M6
        U[a_idx['10']][col] += 1; U[a_idx['00']][col] += -1
        V[b_idx['00']][col] += 1; V[b_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M7
        U[a_idx['01']][col] += 1; U[a_idx['11']][col] += -1
        V[b_idx['10']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        return col

    # Block 1: A_L (2x2) * B_T1 (2x2) -> C[:, 0:2]
    # A indices: A[i,j] -> i*m+j. m=3.
    # A_L[0,0]=A[0,0]->idx 0, A_L[0,1]=A[0,1]->idx 1
    # A_L[1,0]=A[1,0]->idx 3, A_L[1,1]=A[1,1]->idx 4
    a1 = {'00': 0, '01': 1, '10': 3, '11': 4}
    # B_T1[j,k] -> j*p+k. p=4.
    # B_T1[0,0]=B[0,0]->0, B_T1[0,1]=B[0,1]->1
    # B_T1[1,0]=B[1,0]->4, B_T1[1,1]=B[1,1]->5
    b1 = {'00': 0, '01': 1, '10': 4, '11': 5}
    # C[i,k] -> k*n+i. n=2.
    # C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
    c1 = {'00': 0, '01': 2, '10': 1, '11': 3}
    col = _add_strassen_block(a1, b1, c1, 0)

    # Block 2: A_L (2x2) * B_T2 (2x2) -> C[:, 2:4]
    # B_T2[j,k] where k in {2,3}: B_T2[0,2]=B[0,2]->2, B_T2[0,3]=B[0,3]->3
    # B_T2[1,2]=B[1,2]->6, B_T2[1,3]=B[1,3]->7
    b2 = {'00': 2, '01': 3, '10': 6, '11': 7}
    # C[0,2]->4, C[0,3]->6, C[1,2]->5, C[1,3]->7
    c2 = {'00': 4, '01': 6, '10': 5, '11': 7}
    col = _add_strassen_block(a1, b2, c2, col)

    # Block 3: A_R (2x1) * B_B (1x4) -> C (2x4), rank-8 schoolbook
    # A_R[0]=A[0,2]->idx 2, A_R[1]=A[1,2]->idx 5
    # B_B[k]=B[2,k]->idx 8+k for k=0..3
    # C[i,k] -> k*2+i
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1.0  # A[i,2]
            V[2 * p + k][col] = 1.0  # B[2,k]
            W[k * n + i][col] = 1.0  # C[i,k]
            col += 1

    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    T = _tensor_index(n, m, p)

    # Start with rank-22 Strassen decomposition
    U_list, V_list, W_list = _build_rank22_decomposition(n, m, p)
    U = np.array(U_list, dtype=np.float64)
    V = np.array(V_list, dtype=np.float64)
    W = np.array(W_list, dtype=np.float64)

    if not _verify(U, V, W, T):
        # Fallback: schoolbook rank n*m*p = 2*3*4 = 24
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Save the verified rank-22 as fallback
    U_fallback = U.copy()
    V_fallback = V.copy()
    W_fallback = W.copy()

    # Attempt rank reduction
    if time.time() - START_TIME < TIME_BUDGET - 20:
        U, V, W = _try_reduce(U, V, W, T)

    # Verify after reduction
    if not _verify(U, V, W, T):
        # Revert to rank 22
        U = U_fallback
        V = V_fallback
        W = W_fallback

    # Convert to list format with clean values
    # Round to handle floating point noise
    U = np.round(U, 10)
    V = np.round(V, 10)
    W = np.round(W, 10)

    # Final safety check
    if not _verify(U, V, W, T):
        U = np.round(U_fallback, 10)
        V = np.round(V_fallback, 10)
        W = np.round(W_fallback, 10)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Rank: {rank}")
    T = _tensor_index(2, 3, 4)
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Valid: {_verify(U, V, W, T)}")
