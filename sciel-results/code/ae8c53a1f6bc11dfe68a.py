"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Block-partition into six 2x2 @ 2x2 multiplications, each computed
via Strassen's algorithm (rank 7). Total rank = 6 * 7 = 42.

Additionally attempts rank reduction via linear dependency analysis.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7)."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1

    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1

    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1

    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1

    # M5 = (A00+A01)B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 Strassen blocks."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local
                    for s in range(7):
                        coeff = U1[i * 2 + j_local, s]
                        if coeff != 0:
                            U[i * m + a, col_offset + s] += coeff

            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local, s]
                        if coeff != 0:
                            V[a * p + bcol, col_offset + s] += coeff

            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i, s]
                        if coeff != 0:
                            W[bcol * n + i, col_offset + s] += coeff

    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    rank = U.shape[1]
    # For each (i,k), check sum_j U[i*m+j,:]*V[j*p+k,:] elementwise * W[k*n+i,:] sums to 1
    for i in range(n):
        for k in range(p):
            # sum over j: U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            acc = np.zeros(rank)
            for j in range(m):
                acc += U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
            if np.any(np.abs(acc - 1.0) > tol):
                return False
    return True


def _to_output(mat):
    """Convert numpy array to list of lists with int where possible, and half-integers."""
    def _clean_val(x):
        # Try integer first
        r = round(x)
        if abs(x - r) < 1e-9:
            return int(r)
        # Try half-integer
        r2 = round(x * 2) / 2.0
        if abs(x - r2) < 1e-9:
            return r2
        return float(x)
    return [[_clean_val(x) for x in row] for row in mat]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj][col] = 1
                V[jj * p + kk][col] = 1
                W[kk * n + ii][col] = 1
                col += 1
    return U, V, W


def _try_rank_reduction(U, V, W, n, m, p, max_reduce=10, deadline=None):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 components.
    
    For a decomposition of rank r, if removing column s and adjusting the remaining
    columns can still satisfy the tensor identity, we can reduce the rank.
    
    Approach: For each column s, remove it and solve for adjustments to the remaining
    U, V, W columns such that the tensor identity still holds. This is a linear system
    in the adjustment variables.
    """
    if deadline is not None and time.time() > deadline - 5:
        return U, V, W
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    current_rank = U_cur.shape[1]
    
    # Precompute all (i,j,k) indices
    nmk = n * m * p
    
    def _build_residual_system(Uc, Vc, Wc, s_remove, n, m, p):
        """
        Build the linear system for adjusting remaining columns after removing s_remove.
        
        We want: sum_{s != s_remove} (Uc_ij^s + dU_ij^s)(Vc_jk^s + dV_jk^s)(Wc_ki^s + dW_ki^s) = 1
        
        Linearizing around the current values (assuming small adjustments):
        sum_{s != s_remove} [Uc_ij^s Vc_jk^s Wc_ki^s 
            + dU_ij^s * Vc_jk^s * Wc_ki^s 
            + Uc_ij^s * dV_jk^s * Wc_ki^s 
            + Uc_ij^s * Vc_jk^s * dW_ki^s] = 1
        
        The constant term is: sum_{s != s_remove} Uc_ij^s Vc_jk^s Wc_ki^s
        The residual is: 1 - sum_{s != s_remove} Uc_ij^s Vc_jk^s Wc_ki^s
                     = Uc_ij^{s_remove} * Vc_jk^{s_remove} * Wc_ki^{s_remove}
        
        We need: sum_{s != s_remove} [dU_ij^s Vc_jk^s Wc_ki^s + Uc_ij^s dV_jk^s Wc_ki^s + Uc_ij^s Vc_jk^s dW_ki^s] = residual_ij^k
        
        This is a linear system in the variables dU, dV, dW.
        """
        r = Uc.shape[1]
        # Number of unknowns: (r-1) * (n*m + m*p + p*n)
        # But we can be smarter: only adjust columns that interact with the removed one
        
        # For each (i,j,k), the equation involves:
        # sum over s != s_remove of terms involving dU[i*m+j, s], dV[j*p+k, s], dW[k*n+i, s]
        
        # Let's build the system row by row
        # Row for (i,j,k): sum_{s != s_remove} [Vc[j*p+k, s]*Wc[k*n+i, s]*dU[i*m+j, s] 
        #                                      + Uc[i*m+j, s]*Wc[k*n+i, s]*dV[j*p+k, s]
        #                                      + Uc[i*m+j, s]*Vc[j*p+k, s]*dW[k*n+i, s]] = residual
        
        # Variables: for each s != s_remove, we have dU[i*m+j,s], dV[j*p+k,s], dW[k*n+i,s]
        # But these are shared across different (i,j,k) equations.
        
        # Total variables: (r-1) * (n*m + m*p + p*n)
        nm = n * m
        mp = m * p
        pn = p * n
        total_vars = (r - 1) * (nm + mp + pn)
        
        # Build A matrix (nmk x total_vars) and b vector (nmk)
        # This can be large, so let's use a sparse-like approach or just dense if small enough
        
        if total_vars * nmk > 50_000_000:
            return None, None  # Too large
        
        A = np.zeros((nmk, total_vars), dtype=float)
        b = np.zeros(nmk, dtype=float)
        
        # Variable indexing:
        # dU variables: index = s_idx * nm + (i*m + j) where s_idx is the index in remaining columns
        # dV variables: index = nm + s_idx * mp + (j*p + k)
        # dW variables: index = nm + mp + s_idx * pn + (k*n + i)
        
        # Map from original column index to remaining index
        rem_cols = [s for s in range(r) if s != s_remove]
        col_to_rem = {orig: new for new, orig in enumerate(rem_cols)}
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    residual = Uc[i*m+j, s_remove] * Vc[j*p+k, s_remove] * Wc[k*n+i, s_remove]
                    b[eq_idx] = residual
                    
                    for s in rem_cols:
                        s_idx = col_to_rem[s]
                        # dU[i*m+j, s] coefficient: Vc[j*p+k, s] * Wc[k*n+i, s]
                        coeff_U = Vc[j*p+k, s] * Wc[k*n+i, s]
                        if abs(coeff_U) > 1e-15:
                            A[eq_idx, s_idx * nm + (i*m + j)] += coeff_U
                        
                        # dV[j*p+k, s] coefficient: Uc[i*m+j, s] * Wc[k*n+i, s]
                        coeff_V = Uc[i*m+j, s] * Wc[k*n+i, s]
                        if abs(coeff_V) > 1e-15:
                            A[eq_idx, nm + s_idx * mp + (j*p + k)] += coeff_V
                        
                        # dW[k*n+i, s] coefficient: Uc[i*m+j, s] * Vc[j*p+k, s]
                        coeff_W = Uc[i*m+j, s] * Vc[j*p+k, s]
                        if abs(coeff_W) > 1e-15:
                            A[eq_idx, nm + mp + s_idx * pn + (k*n + i)] += coeff_W
        
        return A, b
    
    # Try to eliminate columns one at a time
    for attempt in range(min(max_reduce, current_rank)):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        eliminated = False
        for s_elim in range(current_rank - 1, -1, -1):
            if deadline is not None and time.time() > deadline - 3:
                break
            
            # Check if column is zero
            if (np.allclose(U_cur[:, s_elim], 0, atol=1e-12) and
                np.allclose(V_cur[:, s_elim], 0, atol=1e-12) and
                np.allclose(W_cur[:, s_elim], 0, atol=1e-12)):
                U_cur = np.delete(U_cur, s_elim, axis=1)
                V_cur = np.delete(V_cur, s_elim, axis=1)
                W_cur = np.delete(W_cur, s_elim, axis=1)
                current_rank -= 1
                eliminated = True
                break
            
            # Try to find adjustments
            A, b = _build_residual_system(U_cur, V_cur, W_cur, s_elim, n, m, p)
            if A is None:
                continue
            
            # Solve the linear system (least squares if overdetermined)
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is good enough
            if len(residuals) > 0:
                if residuals[0] > 1e-8 * max(1.0, np.linalg.norm(b)):
                    continue
            
            # Apply the solution
            r = U_cur.shape[1]
            rem_cols = [s for s in range(r) if s != s_elim]
            
            dU = sol[:len(rem_cols) * n * m].reshape(len(rem_cols), n * m)
            dV = sol[len(rem_cols) * n * m : len(rem_cols) * (n * m + m * p)].reshape(len(rem_cols), m * p)
            dW = sol[len(rem_cols) * (n * m + m * p):].reshape(len(rem_cols), p * n)
            
            # Update
            for new_idx, orig_s in enumerate(rem_cols):
                U_cur[:, orig_s] += dU[new_idx, :]
                V_cur[:, orig_s] += dV[new_idx, :]
                W_cur[:, orig_s] += dW[new_idx, :]
            
            # Remove the column
            U_cur = np.delete(U_cur, s_elim, axis=1)
            V_cur = np.delete(V_cur, s_elim, axis=1)
            W_cur = np.delete(W_cur, s_elim, axis=1)
            current_rank -= 1
            eliminated = True
            break
        
        if not eliminated:
            break
    
    # Verify
    if _verify_fast(U_cur, V_cur, W_cur, n, m, p):
        return U_cur, V_cur, W_cur
    
    return U, V, W


def _try_alternative_partition(n, m, p):
    """
    Try an alternative partition: split B into 4x3 + 4x3 instead of three 4x2 blocks.
    
    A (2x4) = [AL | AR], each 2x2
    B (4x6) = [B1 | B2], each 4x3
    C = A*B = [AL*B1 + AR*B1 | AL*B2 + AR*B2]
    
    For each 4x3 block:
    B_j (4x3) = [BjL | BjM | BjR], each 4x1... no, let's think differently.
    
    Actually, 2x2 * 4x3: 
    AL (2x2) * B1_top (2x3) + AR (2x2) * B1_bot (2x3)
    
    2x2 * 2x3 has rank: we can use Strassen-like. 
    2x2 * 2x1 has rank 4 (trivial) or can we do better?
    2x2 * 2x3 = (2x2*2x1) + (2x2*2x1) + (2x2*2x1) = 3 * 4 = 12? No.
    
    Actually 2x2 * 2x2 is rank 7 (Strassen).
    2x2 * 2x3: split the 2x3 into 2x2 + 2x1.
    2x2*2x2 = rank 7, 2x2*2x1 = rank 4. Total 11 per 2x2*2x3.
    Two of those (for AL and AR): 22 per 4x3 block.
    Two 4x3 blocks: 44. That's worse than 42.
    
    What about 2x2 * 4x2 = 2x2*2x2 + 2x2*2x2 = 7+7 = 14.
    Three 4x2 blocks: 42. Same as before.
    
    Let me try: split A differently.
    A (2x4): don't split into 2x2 blocks. Instead, treat it as 2 rows of 4.
    Row 0 of A: [a00, a01, a02, a03]
    Row 1 of A: [a10, a11, a12, a13]
    
    C[0,k] = a00*B[0,k] + a01*B[1,k] + a02*B[2,k] + a03*B[3,k]
    C[1,k] = a10*B[0,k] + a11*B[1,k] + a12*B[2,k] + a13*B[3,k]
    
    This is just the schoolbook approach, rank 48.
    
    Alternative: use a 2x4 * 4x2 = 2x2 multiplication with rank 7?
    No, 2x4 * 4x2 is not a square matrix multiplication.
    
    I think 42 is likely optimal for this partitioning scheme.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Build rank-42 Strassen block decomposition
    U, V, W = _build_rank42()

    # Verify correctness
    if _verify_fast(U, V, W, n, m, p):
        # Try to reduce rank using linear dependency analysis
        U2, V2, W2 = _try_rank_reduction(U, V, W, n, m, p, max_reduce=10, deadline=deadline)
        if U2.shape[1] < U.shape[1]:
            return {"U": _to_output(U2), "V": _to_output(V2), "W": _to_output(W2)}
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}

    # Check if close enough
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(U.shape[1]):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                max_err = max(max_err, abs(val - 1.0))

    if max_err < 1e-6:
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}

    # Fallback: schoolbook rank 48
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    print(f"Rank: {rank}")
    print(f"Verified: {_verify_fast(U, V, W, n, m, p)}")
