import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _strassen_2x2x2():
    """
    Returns 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_sparse, v_sparse, w_sparse) where each is a dict {local_idx: coeff}.
    U local idx: i*2+j (i=row of A, j=col of A = row of B)
    V local idx: j*2+k (j=row of B, k=col of B)
    W local idx: k*2+i (k=col of C, i=row of C) -- trace convention W[k*n+i]
    """
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # W indices: C11 -> k=0,i=0 -> 0; C12 -> k=1,i=0 -> 2; C21 -> k=0,i=1 -> 1; C22 -> k=1,i=1 -> 3
    
    terms = []
    
    # M1: U={0:1, 3:1}, V={0:1, 3:1}, W={0:1, 3:1}
    terms.append((
        {0: 1, 3: 1},
        {0: 1, 3: 1},
        {0: 1, 3: 1}
    ))
    
    # M2: U={2:1, 3:1}, V={0:1}, W={1:1, 3:-1}
    terms.append((
        {2: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1}
    ))
    
    # M3: U={0:1}, V={2:1, 3:-1}, W={2:1, 3:1}
    terms.append((
        {0: 1},
        {2: 1, 3: -1},
        {2: 1, 3: 1}
    ))
    
    # M4: U={3:1}, V={1:1, 0:-1}, W={0:1, 1:1}
    terms.append((
        {3: 1},
        {1: 1, 0: -1},
        {0: 1, 1: 1}
    ))
    
    # M5: U={0:1, 1:1}, V={3:1}, W={0:-1, 2:1}
    terms.append((
        {0: 1, 1: 1},
        {3: 1},
        {0: -1, 2: 1}
    ))
    
    # M6: U={2:1, 0:-1}, V={0:1, 1:1}, W={3:1}
    terms.append((
        {2: 1, 0: -1},
        {0: 1, 1: 1},
        {3: 1}
    ))
    
    # M7: U={1:1, 3:-1}, V={2:1, 3:1}, W={0:1}
    terms.append((
        {1: 1, 3: -1},
        {2: 1, 3: 1},
        {0: 1}
    ))
    
    return terms

def _verify_strassen_2x2x2():
    """Verify the Strassen decomposition is correct for 2x2x2."""
    terms = _strassen_2x2x2()
    n = 2
    # Build the tensor decomposition
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    for r, (u_sp, v_sp, w_sp) in enumerate(terms):
        for idx, coeff in u_sp.items():
            U[idx, r] = coeff
        for idx, coeff in v_sp.items():
            V[idx, r] = coeff
        for idx, coeff in w_sp.items():
            W[idx, r] = coeff
    
    # Target tensor
    T_target = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_target[i*2+j, j*2+k, k*2+i] = 1
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_decomp, T_target, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # First, verify Strassen works
    strassen_ok = _verify_strassen_2x2x2()
    
    if not strassen_ok:
        # Fall back to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Use block decomposition with Strassen where possible
    # Split n=3 as [2,1], m=5 as [2,2,1], p=5 as [2,2,1]
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    U_entries = {}
    V_entries = {}
    W_entries = {}
    
    rank = 0
    strassen_terms = _strassen_2x2x2()
    
    for ni, ni_sz in enumerate(n_parts):
        for mj, mj_sz in enumerate(m_parts):
            for pk, pk_sz in enumerate(p_parts):
                n_off = n_offsets[ni]
                m_off = m_offsets[mj]
                p_off = p_offsets[pk]
                
                if ni_sz == 2 and mj_sz == 2 and pk_sz == 2:
                    for u_sp, v_sp, w_sp in strassen_terms:
                        col = rank
                        U_entries[col] = {}
                        V_entries[col] = {}
                        W_entries[col] = {}
                        
                        for u_rel, coeff in u_sp.items():
                            u_i = u_rel // 2
                            u_j = u_rel % 2
                            u_idx = (n_off + u_i) * m + (m_off + u_j)
                            U_entries[col][u_idx] = coeff
                            
                        for v_rel, coeff in v_sp.items():
                            v_j = v_rel // 2
                            v_k = v_rel % 2
                            v_idx = (m_off + v_j) * p + (p_off + v_k)
                            V_entries[col][v_idx] = coeff
                            
                        for w_rel, coeff in w_sp.items():
                            w_k = w_rel // 2
                            w_i = w_rel % 2
                            w_idx = (p_off + w_k) * n + (n_off + w_i)
                            W_entries[col][w_idx] = coeff
                            
                        rank += 1
                else:
                    for i in range(ni_sz):
                        for j in range(mj_sz):
                            for k in range(pk_sz):
                                col = rank
                                u_idx = (n_off + i) * m + (m_off + j)
                                v_idx = (m_off + j) * p + (p_off + k)
                                w_idx = (p_off + k) * n + (n_off + i)
                                
                                U_entries[col] = {u_idx: 1}
                                V_entries[col] = {v_idx: 1}
                                W_entries[col] = {w_idx: 1}
                                rank += 1
                                
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for u_idx, coeff in U_entries[col].items():
            U[u_idx][col] = coeff
        for v_idx, coeff in V_entries[col].items():
            V[v_idx][col] = coeff
        for w_idx, coeff in W_entries[col].items():
            W[w_idx][col] = coeff
            
    # Verify
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1
        
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        if not np.allclose(T_decomp, target, atol=1e-10):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n * m)]
            V_s = [[0] * rank_school for _ in range(m * p)]
            W_s = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][col] = 1
                        V_s[j * p + k][col] = 1
                        W_s[k * n + i][col] = 1
                        col += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    except Exception:
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    if time.time() > deadline:
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
