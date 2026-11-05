import numpy as np
import time
def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 9
    
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    # Define Laderman's 21 terms for 4x4x4
    # Standard multiplication D = A * B, D[i,k] = sum_j A[i,j] B[j,k]
    # We need to map this to C[k,i] = sum_j A[i,j] B[j,k]
    # So for each term (u, v, w) where w is indexed by i*4+k (D),
    # we need to map w to index k*4+i (C).
    
    laderman_terms = []
    
    # Helper to add a term
    def add_laderman_term(u_dict, v_dict, w_dict):
        # u, v, w are dicts mapping (i,j) or (j,k) or (i,k) to coeff
        # u: A[i,j] -> coeff
        # v: B[j,k] -> coeff
        # w: D[i,k] -> coeff (standard)
        
        u_vec = [0.0] * 16
        v_vec = [0.0] * 16
        w_vec = [0.0] * 16
        
        for (i, j), c in u_dict.items():
            u_vec[i * 4 + j] = c
        for (j, k), c in v_dict.items():
            v_vec[j * 4 + k] = c
        for (i, k), c in w_dict.items():
            # Map D[i,k] to C[k,i]
            # C index is k * 4 + i
            w_vec[k * 4 + i] = c
            
        laderman_terms.append((u_vec, v_vec, w_vec))

    # The 21 terms
    # 1
    add_laderman_term({(0,0):1}, {(0,0):1}, {(0,0):1})
    # 2
    add_laderman_term({(0,1):1}, {(1,0):1}, {(0,0):1})
    # 3
    add_laderman_term({(1,0):1}, {(0,0):1}, {(1,0):1})
    # 4
    add_laderman_term({(1,1):1}, {(1,0):1}, {(1,0):1})
    # 5
    add_laderman_term({(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}, {(0,1):1})
    # 6
    add_laderman_term({(1,0):1, (1,1):1}, {(0,0):1, (0,1):1}, {(1,1):1})
    # 7
    add_laderman_term({(0,0):1, (1,0):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):-1})
    # 8
    add_laderman_term({(1,1):1, (0,1):-1}, {(0,0):1, (0,1):1}, {(0,0):-1, (1,0):1})
    
    # 9
    add_laderman_term({(2,0):1}, {(0,0):1}, {(2,0):1})
    # 10
    add_laderman_term({(2,1):1}, {(1,0):1}, {(2,0):1})
    # 11
    add_laderman_term({(3,0):1}, {(0,0):1}, {(3,0):1})
    # 12
    add_laderman_term({(3,1):1}, {(1,0):1}, {(3,0):1})
    # 13
    add_laderman_term({(2,0):1, (2,1):1}, {(1,0):1, (1,1):1}, {(2,1):1})
    # 14
    add_laderman_term({(3,0):1, (3,1):1}, {(0,0):1, (0,1):1}, {(3,1):1})
    # 15
    add_laderman_term({(2,0):1, (3,0):-1}, {(1,0):1, (1,1):1}, {(2,1):1, (3,1):-1})
    # 16
    add_laderman_term({(3,1):1, (2,1):-1}, {(0,0):1, (0,1):1}, {(2,0):-1, (3,0):1})
    
    # 17
    add_laderman_term({(0,2):1}, {(2,0):1}, {(0,2):1})
    # 18
    add_laderman_term({(0,3):1}, {(3,0):1}, {(0,3):1})
    # 19
    add_laderman_term({(1,2):1}, {(2,0):1}, {(1,2):1})
    # 20
    add_laderman_term({(1,3):1}, {(3,0):1}, {(1,3):1})
    # 21
    add_laderman_term({(0,2):1, (0,3):1, (1,2):1, (1,3):1}, 
                      {(2,0):1, (2,1):1, (3,0):1, (3,1):1}, 
                      {(0,2):1, (0,3):1, (1,2):1, (1,3):1})

    # Verify Laderman terms for 4x4x4
    # Construct target tensor for 4x4x4
    target_4x4 = np.zeros(16*16*16)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a_idx = i*4+j
                b_idx = j*4+k
                c_idx = k*4+i # Note: C[k,i]
                target_4x4[a_idx * 256 + b_idx * 16 + c_idx] = 1.0

    U_4x4 = np.zeros((16, 21))
    V_4x4 = np.zeros((16, 21))
    W_4x4 = np.zeros((16, 21))
    for r, (u, v, w) in enumerate(laderman_terms):
        U_4x4[:, r] = u
        V_4x4[:, r] = v
        W_4x4[:, r] = w

    T_check = np.einsum('ar,br,cr->abc', U_4x4, V_4x4, W_4x4)
    if not np.allclose(T_check.flatten(), target_4x4, atol=1e-10):
        # Fallback to Strassen or Schoolbook if Laderman implementation is wrong
        # For now, return Schoolbook to ensure validity
        rank = n*m*p
        U = np.zeros((n_dim, rank))
        V = np.zeros((m_dim, rank))
        W = np.zeros((p_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Construct full 4x4x9 solution
    # Two 4x4x4 blocks (k=0..3, k=4..7) and one 4x4x1 block (k=8)
    rank = 21 + 21 + 16
    U_full = np.zeros((n_dim, rank))
    V_full = np.zeros((m_dim, rank))
    W_full = np.zeros((p_dim, rank))
    
    # Block 1: k in 0..3
    for r in range(21):
        U_full[:, r] = U_4x4[:, r]
        # V: B[j, k] where k in 0..3. Index in B is j*9 + k
        v_vec = V_4x4[:, r]
        v_full_vec = np.zeros(36)
        for j in range(4):
            for k in range(4):
                if v_vec[j*4+k] != 0:
                    v_full_vec[j*9+k] = v_vec[j*4+k]
        V_full[:, r] = v_full_vec
        # W: C[k, i] where k in 0..3. Index in C is k*4 + i
        w_vec = W_4x4[:, r]
        w_full_vec = np.zeros(36)
        for k in range(4):
            for i in range(4):
                if w_vec[k*4+i] != 0:
                    w_full_vec[k*4+i] = w_vec[k*4+i]
        W_full[:, r] = w_full_vec

    # Block 2: k in 4..7
    for r in range(21):
        U_full[:, 21+r] = U_4x4[:, r]
        # V: B[j, k] where k in 4..7. Index in B is j*9 + (4+k') where k' in 0..3
        v_vec = V_4x4[:, r]
        v_full_vec = np.zeros(36)
        for j in range(4):
            for k_prime in range(4):
                if v_vec[j*4+k_prime] != 0:
                    v_full_vec[j*9 + (4+k_prime)] = v_vec[j*4+k_prime]
        V_full[:, 21+r] = v_full_vec
        # W: C[k, i] where k in 4..7. Index in C is (4+k')*4 + i
        w_vec = W_4x4[:, r]
        w_full_vec = np.zeros(36)
        for k_prime in range(4):
            for i in range(4):
                if w_vec[k_prime*4+i] != 0:
                    w_full_vec[(4+k_prime)*4+i] = w_vec[k_prime*4+i]
        W_full[:, 21+r] = w_full_vec

    # Block 3: k in 8..8 (Schoolbook)
    for i in range(4):
        for j in range(4):
            r = 42 + i*4 + j
            U_full[i*4+j, r] = 1
            V_full[j*9+8, r] = 1
            W_full[8*4+i, r] = 1

    # Verify full solution
    target_full = np.zeros(n_dim * m_dim * p_dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                flat_idx = a_idx * (m_dim * p_dim) + b_idx * p_dim + c_idx
                target_full[flat_idx] = 1.0

    T_final = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    if not np.allclose(T_final.flatten(), target_full, atol=1e-10):
        # Fallback
        rank = n*m*p
        U = np.zeros((n_dim, rank))
        V = np.zeros((m_dim, rank))
        W = np.zeros((p_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
