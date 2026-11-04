"""Matrix multiplication 5x5x5 - Block decomposition 1+4 with Strassen-Kronecker for 4x4."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_strassen_matrices():
    """
    Returns U, V, W matrices (4x7) for Strassen 2x2 decomposition.
    Convention: U[i, c], V[j, c], W[k, c] where i,j,k are local 0..3 (row-major 2x2).
    Local indices:
    A: 00=0, 01=1, 10=2, 11=3
    B: 00=0, 01=1, 10=2, 11=3
    C: 00=0, 01=1, 10=2, 11=3
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (a10+a11)b00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    
    # M3 = a00(b01-b11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # M4 = a11(b10-b00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # M5 = (a00+a01)b11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute the tensor: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # The matrix multiplication tensor should be T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False, np.max(np.abs(T - 1.0))
    
    return True, 0.0

def _build_4x4_strassen_kronecker():
    """
    Build the 49-term Strassen-Kronecker decomposition for 4x4 matrix multiplication.
    Returns list of (u_dict, v_dict, w_dict) where keys are (local_i, local_j) etc.
    """
    S_U, S_V, S_W = _get_strassen_matrices()
    
    terms = []
    for c1 in range(7):
        for c2 in range(7):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # U: Kronecker product of S_U[:,c1] and S_U[:,c2]
            for i1 in range(2):
                for j1 in range(2):
                    v1 = S_U[i1, c1]
                    if v1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            v2 = S_U[i2, c2]
                            if v2 == 0:
                                continue
                            li = 2 * i1 + i2
                            lj = 2 * j1 + j2
                            val = v1 * v2
                            u_dict[(li, lj)] = u_dict.get((li, lj), 0) + val
            
            # V: Kronecker product of S_V[:,c1] and S_V[:,c2]
            for j1 in range(2):
                for k1 in range(2):
                    v1 = S_V[j1, c1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = S_V[j2, c2]
                            if v2 == 0:
                                continue
                            lj = 2 * j1 + j2
                            lk = 2 * k1 + k2
                            val = v1 * v2
                            v_dict[(lj, lk)] = v_dict.get((lj, lk), 0) + val
            
            # W: Kronecker product of S_W[:,c1] and S_W[:,c2]
            for k1 in range(2):
                for i1 in range(2):
                    v1 = S_W[k1, c1]
                    if v1 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            v2 = S_W[k2, c2]
                            if v2 == 0:
                                continue
                            lk = 2 * k1 + k2
                            li = 2 * i1 + i2
                            val = v1 * v2
                            w_dict[(lk, li)] = w_dict.get((lk, li), 0) + val
            
            terms.append((u_dict, v_dict, w_dict))
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Partition 5 into 1 + 4
    blocks = [
        [0],
        [1, 2, 3, 4]
    ]
    
    # Pre-compute 4x4 Strassen-Kronecker terms
    M4_terms = _build_4x4_strassen_kronecker()
    
    # Build global U, V, W
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    terms = []
    
    # Iterate over block triples
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                I = blocks[bi]
                J = blocks[bj]
                K = blocks[bk]
                
                if bi == 1 and bj == 1 and bk == 1:
                    # Use 4x4 Strassen-Kronecker decomposition
                    for u_dict, v_dict, w_dict in M4_terms:
                        gu_dict = {}
                        gv_dict = {}
                        gw_dict = {}
                        
                        for (li, lj), coeff in u_dict.items():
                            gi = I[li]
                            gj = J[lj]
                            g_idx = gi * m + gj
                            gu_dict[g_idx] = gu_dict.get(g_idx, 0) + coeff
                            
                        for (lj, lk), coeff in v_dict.items():
                            gj = J[lj]
                            gk = K[lk]
                            g_idx = gj * p + gk
                            gv_dict[g_idx] = gv_dict.get(g_idx, 0) + coeff
                            
                        for (lk, li), coeff in w_dict.items():
                            gk = K[lk]
                            gi = I[li]
                            g_idx = gk * n + gi
                            gw_dict[g_idx] = gw_dict.get(g_idx, 0) + coeff
                            
                        terms.append((gu_dict, gv_dict, gw_dict))
                else:
                    # Schoolbook for mixed blocks
                    for li in I:
                        for lj in J:
                            for lk in K:
                                gu = li * m + lj
                                gv = lj * p + lk
                                gw = lk * n + li
                                terms.append(({gu: 1}, {gv: 1}, {gw: 1}))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            if 0 <= idx < U_rows:
                U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            if 0 <= idx < V_rows:
                V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            if 0 <= idx < W_rows:
                W[idx][c] = coeff
    
    # Verify
    valid, err = _verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
