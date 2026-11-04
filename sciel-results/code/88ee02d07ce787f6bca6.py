import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # First try to find rank-16 via structured algebraic search
    result_16 = _try_rank16_algebraic(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Fallback: rank-17
    return _rank17_decomposition(n, m, p)


def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition with half-integer rounding."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    Ui = np.rint(U_s * 2).astype(np.int64)
    Vi = np.rint(V_s * 2).astype(np.int64)
    Wi = np.rint(W_s * 2).astype(np.int64)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None


def _try_rank16_algebraic(n, m, p, deadline):
    """
    Structured algebraic search for rank-16 decomposition.
    Uses Strassen-based initialization for the 2x2x2 sub-block plus structured
    extensions for the remaining indices, then refines via integer rounding.
    """
    rank = 16
    target = np.ones((n, m, p))
    
    np.random.seed(42)
    
    def _verify_rank16(U, V, W):
        """Verify exact decomposition with half-integer rounding."""
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        Ui = np.rint(U_s * 2).astype(np.int64).reshape(n, m, rank)
        Vi = np.rint(V_s * 2).astype(np.int64).reshape(m, p, rank)
        Wi = np.rint(W_s * 2).astype(np.int64).reshape(p, n, rank)
        prod = np.einsum('ijr,jkr,kir->ijk', Ui, Vi, Wi)
        if np.all(prod == 8):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def _solve_for_W(U, V):
        """Solve linear system for W given fixed U, V."""
        A = np.zeros((n * m * p, p * n * rank))
        b = np.ones(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        col_idx = (k * n + i) * rank + r
                        A[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
        try:
            W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return W_flat.reshape(p * n, rank)
        except:
            return None

    def _solve_for_V(U, W):
        """Solve linear system for V given fixed U, W."""
        A = np.zeros((n * m * p, m * p * rank))
        b = np.ones(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        col_idx = (j * p + k) * rank + r
                        A[eq_idx, col_idx] = U[i * m + j, r] * W[k * n + i, r]
        try:
            V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return V_flat.reshape(m * p, rank)
        except:
            return None

    def _solve_for_U(V, W):
        """Solve linear system for U given fixed V, W."""
        A = np.zeros((n * m * p, n * m * rank))
        b = np.ones(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        col_idx = (i * m + j) * rank + r
                        A[eq_idx, col_idx] = V[j * p + k, r] * W[k * n + i, r]
        try:
            U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return U_flat.reshape(n * m, rank)
        except:
            return None

    # Strategy A: Strassen-based initialization for 2x2x2 block
    # Strassen 7 terms for (i,j,k) in {0,1}^3, then 9 terms for remaining
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2x2 block (i,j,k in {0,1})
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0*3+0, 0] = 1; U[1*3+1, 0] = 1
    V[0*3+0, 0] = 1; V[1*3+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[1*3+0, 1] = 1; U[1*3+1, 1] = 1
    V[0*3+0, 1] = 1
    W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
    
    # M3 = A00(B10-B00) -> C01 += M3, C11 -= M3
    U[0*3+0, 2] = 1
    V[1*3+0, 2] = 1; V[0*3+0, 2] = -1
    W[0*2+0, 2] = -1; W[1*2+1, 2] = 1
    
    # M4 = A11(B01-B10) -> C00 += M4, C10 -= M4
    U[1*3+1, 3] = 1
    V[0*3+1, 3] = 1; V[1*3+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = -1
    
    # M5 = (A00+A01)B11 -> C00 += M5, C01 += M5
    U[0*3+0, 4] = 1; U[0*3+1, 4] = 1
    V[1*3+1, 4] = 1
    W[0*2+0, 4] = 1; W[0*2+1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10) -> C10 += M6, C11 -= M6
    U[1*3+0, 5] = 1; U[0*3+0, 5] = -1
    V[0*3+0, 5] = 1; V[1*3+0, 5] = 1
    W[0*2+1, 5] = 1; W[1*2+1, 5] = -1
    
    # M7 = (A01-A11)(B01+B11) -> C01 += M7, C11 -= M7
    U[0*3+1, 6] = 1; U[1*3+1, 6] = -1
    V[0*3+1, 6] = 1; V[1*3+1, 6] = 1
    W[0*2+1, 6] = -1; W[1*2+1, 6] = 1
    
    # Now we need to handle terms involving index 2 (j=2 or k=2)
    # Remaining contributions needed:
    # For k=2: C[i,2] = sum_j A[i,j]*B[j,2] for i=0,1
    # For j=2: C[i,k] += A[i,2]*B[2,k] for i=0,1, k=0,1,2
    # Total: 2 (for k=2, j=0,1) + 3 (for j=2, k=0,1) + 1 (for j=2,k=2) = 6 more terms
    # But we have 9 terms left (r=7..15). Let's use them.
    
    # Direct terms for j=2: A[i,2]*B[2,k] for i=0,1; k=0,1,2 -> 6 terms
    for idx, (i, k) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]):
        r = 7 + idx
        U[i*3+2, r] = 1
        V[2*3+k, r] = 1
        W[k*2+i, r] = 1
    
    # That uses r=7..12. We have r=13,14,15 left.
    # Check if the Strassen terms already cover k=2 contributions... they don't.
    # Actually, the Strassen terms only cover k in {0,1}. We need k=2 contributions.
    # The direct terms above handle j=2 for all k. But we still need j=0,1 with k=2.
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    # The A[i,2]*B[2,2] part is covered by r=9 (i=0,k=2) and r=12 (i=1,k=2)
    # We need A[i,0]*B[0,2] + A[i,1]*B[1,2] for i=0,1 -> 2 more terms
    # But we only have 3 terms left (r=13,14,15). Let's use 2 for these and 1 spare.
    
    # A[0,0]*B[0,2] -> C[0,2]
    U[0*3+0, 13] = 1
    V[0*3+2, 13] = 1
    W[2*2+0, 13] = 1
    
    # A[1,0]*B[0,2] -> C[1,2]
    U[1*3+0, 14] = 1
    V[0*3+2, 14] = 1
    W[2*2+1, 14] = 1
    
    # A[0,1]*B[1,2] -> C[0,2]
    U[0*3+1, 15] = 1
    V[1*3+2, 15] = 1
    W[2*2+0, 15] = 1
    
    # A[1,1]*B[1,2] -> C[1,2] -- PROBLEM: we need this but have no more terms!
    # This means the simple Strassen + direct approach needs 7+6+3 = 16 terms,
    # but the last one (A[1,1]*B[1,2]) doesn't fit.
    
    # Let's verify what we have so far and see if it works:
    ok, U_s, V_s, W_s = _verify_rank16(U, V, W)
    if ok:
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # The above doesn't work because we're missing A[1,1]*B[1,2].
    # Let's try a different approach: use the spare term to create a combined term.
    # Instead of separate terms for A[0,1]*B[1,2] and A[1,1]*B[1,2],
    # use (A[0,1]+A[1,1])*B[1,2] and distribute to C[0,2] and C[1,2].
    
    # Reset and try again with modified structure
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7 terms (same as before)
    U[0*3+0, 0] = 1; U[1*3+1, 0] = 1
    V[0*3+0, 0] = 1; V[1*3+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    U[1*3+0, 1] = 1; U[1*3+1, 1] = 1
    V[0*3+0, 1] = 1
    W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
    
    U[0*3+0, 2] = 1
    V[1*3+0, 2] = 1; V[0*3+0, 2] = -1
    W[0*2+0, 2] = -1; W[1*2+1, 2] = 1
    
    U[1*3+1, 3] = 1
    V[0*3+1, 3] = 1; V[1*3+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = -1
    
    U[0*3+0, 4] = 1; U[0*3+1, 4] = 1
    V[1*3+1, 4] = 1
    W[0*2+0, 4] = 1; W[0*2+1, 4] = 1
    
    U[1*3+0, 5] = 1; U[0*3+0, 5] = -1
    V[0*3+0, 5] = 1; V[1*3+0, 5] = 1
    W[0*2+1, 5] = 1; W[1*2+1, 5] = -1
    
    U[0*3+1, 6] = 1; U[1*3+1, 6] = -1
    V[0*3+1, 6] = 1; V[1*3+1, 6] = 1
    W[0*2+1, 6] = -1; W[1*2+1, 6] = 1
    
    # Terms for j=2: A[i,2]*B[2,k] for i=0,1; k=0,1,2 -> 6 terms (r=7..12)
    for idx, (i, k) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]):
        r = 7 + idx
        U[i*3+2, r] = 1
        V[2*3+k, r] = 1
        W[k*2+i, r] = 1
    
    # Combined term: (A[0,1]+A[1,1])*B[1,2] -> contributes to C[0,2] and C[1,2]
    U[0*3+1, 13] = 1; U[1*3+1, 13] = 1
    V[1*3+2, 13] = 1
    W[2*2+0, 13] = 1; W[2*2+1, 13] = 1
    
    # A[0,0]*B[0,2] -> C[0,2]
    U[0*3+0, 14] = 1
    V[0*3+2, 14] = 1
    W[2*2+0, 14] = 1
    
    # A[1,0]*B[0,2] -> C[1,2]
    U[1*3+0, 15] = 1
    V[0*3+2, 15] = 1
    W[2*2+1, 15] = 1
    
    ok, U_s, V_s, W_s = _verify_rank16(U, V, W)
    if ok:
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try random perturbations and refinement
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        # Start from a random sparse initialization
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        for r in range(rank):
            nz_u = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
            for idx in nz_u:
                U[idx, r] = np.random.choice([-1, 1])
            nz_v = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
            for idx in nz_v:
                V[idx, r] = np.random.choice([-1, 1])
        
        W = _solve_for_W(U, V)
        if W is None:
            continue
        
        ok, U_s, V_s, W_s = _verify_rank16(U, V, W)
        if ok:
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        # Try solving for V given U, W
        V2 = _solve_for_V(U, W)
        if V2 is not None:
            ok, U_s, V_s, W_s = _verify_rank16(U, V2, W)
            if ok:
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        # Try solving for U given V, W
        U2 = _solve_for_U(V, W)
        if U2 is not None:
            ok, U_s, V_s, W_s = _verify_rank16(U2, V, W)
            if ok:
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return None
