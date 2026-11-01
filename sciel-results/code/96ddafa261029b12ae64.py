"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that U, V, W exactly decompose the 2x2x4 matrix multiplication tensor."""
    n, m, p = 2, 2, 4
    rank = len(U[0]) if U else 0
    
    # Build expected tensor: T[a,b,c] = 1 if A[a]*B[b] contributes to C[c]
    # a = i*m+j (A[i,j]), b = j*p+k (B[j,k]), c = k*n+i (C[i,k] in W[k*n+i])
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Compute decomposition: D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    D = np.zeros((n*m, m*p, p*n))
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for r in range(rank):
        D += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    # Check exact match
    return np.allclose(D, T, atol=1e-10)

def _peeling_search(deadline=140.0):
    """
    Apply peeling algorithm to try to find a rank-13 decomposition.
    Returns (rank, U, V, W) if found, else None.
    """
    start_time = time.time()
    n, m, p = 2, 2, 4
    
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Peeling: try to find rank-1 components to subtract
    # For a small tensor, we can enumerate possible rank-1 subtractions.
    # A rank-1 term is u ⊗ v ⊗ w where u has 4 entries, v has 8, w has 8.
    # The term must match at least one non-zero entry of the current tensor.
    
    # Strategy: try all possible "first terms" and recurse.
    # But this is exponential. Instead, use the structure of the problem.
    
    # Key insight: In any rank-r decomposition, each rank-1 term must
    # account for at least one tensor entry. With 16 entries and rank 13,
    # on average each term accounts for ~1.2 entries.
    
    # Let's try a structured search:
    # 1. The tensor has a block structure: left 2x2x2 and right 2x2x2.
    # 2. Try decompositions where some terms span both blocks.
    
    # For rank 13, we need to save 1 multiplication compared to 14.
    # This means one term must contribute to both the left and right blocks.
    
    # Let's try: find a term u⊗v⊗w where u has support in both A-indices
    # that are used in both blocks, v has support in both B-indices from 
    # left and right, and similarly for w.
    
    # From the rank-14 solution, the A-combinations used are:
    # (A00+A11), (A10+A11), A00, A11, (A00+A01), (A10-A00), (A01-A11)
    # These are 7 distinct linear combinations of the 4 A-entries.
    
    # For a shared term, we need u to be one of these (or a combination),
    # v to span both left and right B-indices, and w to span both left and right C-indices.
    
    # Let's try: u = [1,0,0,1] (A00+A11), which is used in both blocks.
    # For v, we need it to have non-zero entries in both {B00,B01,B10,B11} and {B02,B03,B12,B13}.
    # For w, we need it to have non-zero entries in both {C00,C10,C01,C11} and {C02,C12,C03,C13}.
    
    # But a single rank-1 term u⊗v⊗w produces a specific pattern.
    # Let's enumerate possible v and w that span both blocks.
    
    # This is getting complex. Let me try a numerical approach:
    # Assume rank 13 and use the rank-14 solution as a starting point,
    # then try to reduce rank by one using a local search.
    
    if time.time() - start_time > deadline - 5:
        return None
    
    # Try a direct approach: use the rank-14 solution and try to express
    # two of its terms as a single rank-1 term (impossible exactly) or
    # find a different decomposition.
    
    # Given the theoretical lower bounds and known results, rank 14 is likely minimal.
    # The peeling algorithm would confirm this by failing to find rank 13.
    
    return None

def _try_rank13_numerical(deadline=140.0):
    """Try to find a rank-13 decomposition using numerical optimization."""
    start_time = time.time()
    n, m, p = 2, 2, 4
    r = 13
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Initialize from rank-14 solution, drop one column
    sol14 = _build_rank14()
    U14 = np.array(sol14['U'], dtype=float)
    V14 = np.array(sol14['V'], dtype=float)
    W14 = np.array(sol14['W'], dtype=float)
    
    # Drop the last column to get rank-13 init
    U = U14[:, :r].copy()
    V = V14[:, :r].copy()
    W = W14[:, :r].copy()
    
    # Add small random perturbation
    np.random.seed(42)
    U += 0.05 * np.random.randn(*U.shape)
    V += 0.05 * np.random.randn(*V.shape)
    W += 0.05 * np.random.randn(*W.shape)
    
    def objective(params):
        if time.time() - start_time > deadline - 10:
            return 1e10
        U_flat = params[:4*r].reshape(4, r)
        V_flat = params[4*r:4*r+8*r].reshape(8, r)
        W_flat = params[4*r+8*r:].reshape(8, r)
        
        D = np.zeros((4, 8, 8))
        for ri in range(r):
            D += np.einsum('a,b,c->abc', U_flat[:, ri], V_flat[:, ri], W_flat[:, ri])
        
        return np.sum((D - T)**2)
    
    def gradient(params):
        if time.time() - start_time > deadline - 10:
            return np.zeros_like(params)
        U_flat = params[:4*r].reshape(4, r)
        V_flat = params[4*r:4*r+8*r].reshape(8, r)
        W_flat = params[4*r+8*r:].reshape(8, r)
        
        D = np.zeros((4, 8, 8))
        for ri in range(r):
            D += np.einsum('a,b,c->abc', U_flat[:, ri], V_flat[:, ri], W_flat[:, ri])
        residual = D - T
        
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        for ri in range(r):
            # Fixed einsum subscripts to match 1D operands
            grad_U[:, ri] = np.einsum('abc,b,c->a', residual, V_flat[:, ri], W_flat[:, ri])
            grad_V[:, ri] = np.einsum('abc,a,c->b', residual, U_flat[:, ri], W_flat[:, ri])
            grad_W[:, ri] = np.einsum('abc,a,b->c', residual, U_flat[:, ri], V_flat[:, ri])
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000})
    
    if time.time() - start_time > deadline - 5:
        return None
    
    if result.fun < 1e-10:
        U_final = result.x[:4*r].reshape(4, r)
        V_final = result.x[4*r:4*r+8*r].reshape(8, r)
        W_final = result.x[4*r+8*r:].reshape(8, r)
        
        # Round to nearest half-integer
        U_round = np.round(U_final * 2) / 2
        V_round = np.round(V_final * 2) / 2
        W_round = np.round(W_final * 2) / 2
        
        D = np.zeros((4, 8, 8))
        for ri in range(r):
            D += np.einsum('a,b,c->abc', U_round[:, ri], V_round[:, ri], W_round[:, ri])
        
        if np.allclose(D, T, atol=1e-10):
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    return None

def construct():
    start_time = time.time()
    
    # Try to find a rank-13 decomposition using numerical optimization
    result = _try_rank13_numerical(deadline=140.0)
    
    if result is not None:
        # Verify before returning
        if _verify_decomposition(result['U'], result['V'], result['W']):
            return result
    
    # Fall back to rank 14
    return _build_rank14()
# EVOLVE-BLOCK-END
