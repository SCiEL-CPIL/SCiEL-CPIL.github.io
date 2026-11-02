"""Matrix multiplication 2x2x4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Should equal sum_j A[i,j]*B[j,k]
    """
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = len(U[0])
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal delta (i.e., T[i,j,k] = 1 if we're computing C[i,k]=sum_j A[i,j]B[j,k])
    # Actually T[i,j,k] should be 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, no. The tensor for matrix mult is: T[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So we need: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if total != 1:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using numpy with float, then exact check on promising candidates."""
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should all equal 1.
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build tensor via einsum or manual
    # U reshaped: (n, m, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check if all entries are approximately 1
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    
    # Exact verification with fractions
    return _verify_decomposition(U, V, W, n, m, p)

def _schoolbook_terms(n, m, p):
    """Return the 16 schoolbook rank-1 terms for n x m x p matrix mult.
    Term (i,j,k): U has 1 at position (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
    """
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u, v, w))
    return terms

def _random_rank_r_search(n, m, p, rank, num_trials, deadline):
    """Try to find a rank-r decomposition by random linear combinations of schoolbook terms."""
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    schoolbook = _schoolbook_terms(n, m, p)
    num_schoolbook = len(schoolbook)
    
    best_rank = rank + 1  # We'll try to find rank-r
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Randomly select 'rank' terms and random integer coefficients
        # Actually, we need to form rank linear combinations of the schoolbook terms
        # such that the resulting rank-r decomposition is exact.
        
        # Strategy: pick 'rank' random subsets of schoolbook terms with random coefficients
        # and check if the resulting tensor equals the matrix mult tensor.
        
        # A rank-r term is: sum_{s=1}^{r} c_s * (u_s, v_s, w_s)
        # where each (u_s, v_s, w_s) is a linear combination of schoolbook terms.
        
        # Let's generate random linear combinations:
        U = [[0]*rank for _ in range(u_dim)]
        V = [[0]*rank for _ in range(v_dim)]
        W = [[0]*rank for _ in range(w_dim)]
        
        for r in range(rank):
            # Pick a random linear combination of schoolbook terms
            # Use small random integer coefficients
            coeffs = [random.randint(-2, 2) for _ in range(num_schoolbook)]
            # Ensure not all zero
            if all(c == 0 for c in coeffs):
                coeffs[0] = 1
            
            for s, (u_s, v_s, w_s) in enumerate(schoolbook):
                c = coeffs[s]
                if c != 0:
                    for idx in range(u_dim):
                        U[idx][r] += c * u_s[idx]
                    for idx in range(v_dim):
                        V[idx][r] += c * v_s[idx]
                    for idx in range(w_dim):
                        W[idx][r] += c * w_s[idx]
        
        # Verify
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 4
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Known rank-14 decomposition (Strassen-based)
    rank14 = _strassen_rank14(n, m, p)
    
    # Try to find rank-13
    # Use randomized search
    random.seed(42)
    
    result_13 = _random_rank_r_search(n, m, p, 13, num_trials=50000, deadline=deadline)
    
    if result_13 is not None:
        U, V, W = result_13
        return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-14
    U, V, W = rank14
    return {"U": U, "V": V, "W": W}

def _strassen_rank14(n, m, p):
    """Construct the known rank-14 Strassen-based decomposition for 2x2x4."""
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    rank = 14
    
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen 7-term decomposition for 2x2x2:
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
    # M2 = (a10+a11)b00       -> contributes to c10, c11
    # M3 = a00(b01-b11)       -> contributes to c01, c11
    # M4 = a11(b10-b00)       -> contributes to c00, c10
    # M5 = (a00+a01)b11       -> contributes to c01, c00
    # M6 = (a10-a00)(b00+b01) -> contributes to c11
    # M7 = (a01-a11)(b10+b11) -> contributes to c00
    
    # For 2x2x4, output C has 2 rows (i=0,1) and 4 columns (k=0,1,2,3)
    # W is indexed as W[k*n + i, r]
    # U is indexed as U[i*m + j, r]
    # V is indexed as V[j*p + k, r]
    
    # Block 1: columns k=0,1 of B
    terms_block1 = [
        # (u_entries: list of (i, j, val), v_entries: list of (j, k, val), w_entries: list of (k, i, val))
        # M1: (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2: (a10+a11)b00 -> c10 += M2, c11 -= M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        # M3: a00(b01-b11) -> c01 += M3, c11 += M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        # M4: a11(b10-b00) -> c00 += M4, c10 += M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # M5: (a00+a01)b11 -> c01 += M5, c00 -= M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1),(0,0,-1)]),
        # M6: (a10-a00)(b00+b01) -> c11 += M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7: (a01-a11)(b10+b11) -> c00 += M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Block 2: columns k=2,3 of B
    terms_block2 = [
        # M1: (a00+a11)(b02+b13) -> c02 += M1, c13 += M1
        ([(0,0,1),(1,1,1)], [(0,2,1),(1,3,1)], [(2,0,1),(3,1,1)]),
        # M2: (a10+a11)b02 -> c12 += M2, c13 -= M2
        ([(1,0,1),(1,1,1)], [(0,2,1)], [(2,1,1),(3,1,-1)]),
        # M3: a00(b03-b13) -> c03 += M3, c13 += M3
        ([(0,0,1)], [(0,3,1),(1,3,-1)], [(3,0,1),(3,1,1)]),
        # M4: a11(b12-b02) -> c02 += M4, c12 += M4
        ([(1,1,1)], [(1,2,1),(0,2,-1)], [(2,0,1),(2,1,1)]),
        # M5: (a00+a01)b13 -> c03 += M5, c02 -= M5
        ([(0,0,1),(0,1,1)], [(1,3,1)], [(3,0,1),(2,0,-1)]),
        # M6: (a10-a00)(b02+b03) -> c13 += M6
        ([(1,0,1),(0,0,-1)], [(0,2,1),(0,3,1)], [(3,1,1)]),
        # M7: (a01-a11)(b12+b13) -> c02 += M7
        ([(0,1,1),(1,1,-1)], [(1,2,1),(1,3,1)], [(2,0,1)]),
    ]
    
    for r in range(7):
        u_ent, v_ent, w_ent = terms_block1[r]
        col = r
        for i, j, val in u_ent:
            U[i*m+j][col] += val
        for j, k, val in v_ent:
            V[j*p+k][col] += val
        for k, i, val in w_ent:
            W[k*n+i][col] += val
    
    for r in range(7):
        u_ent, v_ent, w_ent = terms_block2[r]
        col = 7 + r
        for i, j, val in u_ent:
            U[i*m+j][col] += val
        for j, k, val in v_ent:
            V[j*p+k][col] += val
        for k, i, val in w_ent:
            W[k*n+i][col] += val
    
    return U, V, W
# EVOLVE-BLOCK-END
