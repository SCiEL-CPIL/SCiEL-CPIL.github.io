"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i,j,r]*V[j,k,r] == C[i,k] for all i,j,k.
    Uses the trace convention: W[k*n+i, r].
    Returns True if valid.
    """
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]... 
    # Actually the decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] for the product,
    # and W encodes the output. The condition is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] = sum_r W[k*n+i, r] * delta... 
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = delta (1 if this is the "1" in the multiplication tensor)
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait no. The tensor is: T[i,j,k] such that C = A*B means
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is indexed by (i, j, k) and T[i,j,k] = 1.
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # But W is indexed by (k*n+i), so W[k*n+i, r].
    # Let me just check: for random A, B, does A*B match?
    
    # Simpler check: verify with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed[i, k] = s
    
    return np.allclose(C_expected, C_computed, atol=1e-10)

def _build_2x3x2_rank11():
    """Build rank-11 decomposition for 2x3 x 3x2 -> 2x2.
    A (2x3): flat = i*3+j, i in 0..1, j in 0..2
    B (3x2): flat = j*2+k, j in 0..2, k in 0..1
    C (2x2): flat = i*2+k, i in 0..1, k in 0..1
    Returns list of 11 terms, each with 'u', 'v', 'w' dicts.
    """
    terms = []
    
    # Strassen-like for 2x2x2 part (using A[:,0:2], B[0:2,:]):
    # M1 = (a00+a11)(b00+b10)
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 2: 1}, 'w': {0: 1, 2: 1}})
    # M2 = (a10+a11)(b00)
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {1: 1, 2: -1}})
    # M3 = a00(b01-b11)
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 2: 1}})
    # M4 = a11(b10-b00)
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 1: 1}})
    # M5 = (a00+a01)(b11)
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    # M6 = (a10-a00)(b00+b01)
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {2: 1}})
    # M7 = (a01-a11)(b10+b11)
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    
    # Scalar terms for column 2 of A:
    # K1 = a02 * b20
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    # K2 = a02 * b21
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    # K3 = a12 * b20
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    # K4 = a12 * b21
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    return terms

def construct():
    _t0 = time.time()
    _deadline = _t0 + 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Use 3 blocks of 2x3x2, each with rank 11. Total rank 33.
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = _build_2x3x2_rank11()
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    result = {"U": U, "V": V, "W": W}
    
    # Quick verification
    if time.time() < _deadline:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback: naive rank 36
            rank36 = n * m * p
            U36 = [[0] * rank36 for _ in range(n * m)]
            V36 = [[0] * rank36 for _ in range(m * p)]
            W36 = [[0] * rank36 for _ in range(p * n)]
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U36[i*m+j][r] = 1
                        V36[j*p+k][r] = 1
                        W36[k*n+i][r] = 1
                        r += 1
            result = {"U": U36, "V": V36, "W": W36}
    
    return result
# EVOLVE-BLOCK-END
