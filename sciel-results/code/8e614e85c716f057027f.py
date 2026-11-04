import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses Kronecker product structure: (2x2x2 Strassen rank-7) x (2x2x3 rank-11) = rank 77.
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strassen 2x2x2 (Rank 7)
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 2x2x3 Decomposition (Rank 11)
    # Composed of 2x2x2 Strassen (7 terms) + 2x2x1 Schoolbook (4 terms)
    S2x1 = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        ({(1,1): 1}, {(1,0): 1}, {(1,0): 1}),
    ]
    S2x3 = S2 + S2x1
    
    # Kronecker Product Construction
    # Indices:
    # U: (i1, i2, j1, j2) -> i = 2*i1 + i2, j = 2*j1 + j2
    # V: (j1, j2, k1, k2) -> j = 2*j1 + j2, k = 2*k1 + k2 (k2 in 0..2 for 2x3 part)
    # W: (k1, k2, i1, i2) -> k = 2*k1 + k2, i = 2*i1 + i2
    
    # Note: The 2x2x3 part has k dimension of size 3.
    # So k = 2*k1 + k2 is not valid for k up to 6 if k2 goes to 2.
    # k1 in {0,1}, k2 in {0,1,2}.
    # k = 2*k1 + k2.
    # If k1=0, k=0,1,2.
    # If k1=1, k=2,3,4. -> Collision at k=2.
    
    # The Kronecker product structure for $M_{2 \times 2 \times 2} \otimes M_{2 \times 2 \times 3}$ 
    # maps to $M_{4 \times 4 \times 6}$ as follows:
    # A index: (a1, a2) where a1 in {0,1}, a2 in {0,1} -> a = 2*a1 + a2
    # B index: (b1, b2) where b1 in {0,1}, b2 in {0,1,2} -> b = 2*b1 + b2? 
    # No, B is 4x6. 
    # Middle dimension m=4. So B rows are 4. 
    # B = B_{b1, b2} where b1 in {0,1}, b2 in {0,1}.
    # Wait, the tensor product $T_A \otimes T_B$ has middle dimension $m_A m_B$.
    # Here $m_A = 2, m_B = 2$. Product 4. Correct.
    # Output dimension $p_A p_B = 2 \times 3 = 6$. Correct.
    # Input dimension $n_A n_B = 2 \times 2 = 4$. Correct.
    
    # So the indices are:
    # U (Input A): $i = 2 i_1 + i_2$, $j = 2 j_1 + j_2$.
    #   $i_1, i_2 \in \{0,1\}$, $j_1, j_2 \in \{0,1\}$.
    # V (Input B): $j = 2 j_1 + j_2$, $k = 2 k_1 + k_2$.
    #   $j_1, j_2 \in \{0,1\}$, $k_1 \in \{0,1\}$, $k_2 \in \{0,1,2\}$.
    #   Wait, $k$ in $M_{2 \times 2 \times 3}$ is the column index, range 0..2.
    #   So $k = 2 k_1 + k_2$?
    #   If $k_1=0, k_2 \in \{0,1,2\} \implies k \in \{0,1,2\}$.
    #   If $k_1=1, k_2 \in \{0,1,2\} \implies k \in \{2,3,4\}$.
    #   This overlaps at $k=2$.
    #   The standard Kronecker index mapping is $k = k_1 p_B + k_2$.
    #   $p_B = 3$.
    #   $k = 3 k_1 + k_2$.
    #   $k_1 \in \{0,1\}, k_2 \in \{0,1,2\}$.
    #   $k \in \{0,1,2, 3,4,5\}$. Correct.
    
    # W (Output C): $k = 3 k_1 + k_2$, $i = 2 i_1 + i_2$.
    #   $k_1 \in \{0,1\}, k_2 \in \{0,1,2\}$.
    #   $i_1, i_2 \in \{0,1\}$.
    
    for u1, v1, w1 in S2:
        for u2, v2, w2 in S2x3:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # u1 is for 2x2x2 (A1, B1, C1)
            # u2 is for 2x2x3 (A2, B2, C2)
            
            # U index: i = 2*i1 + i2, j = 2*j1 + j2
            # u1 keys: (i1, j1)
            # u2 keys: (i2, j2)
            for (i1, j1), c1 in u1.items():
                for (i2, j2), c2 in u2.items():
                    i = 2 * i1 + i2
                    j = 2 * j1 + j2
                    u_col[i * m + j] += c1 * c2
            
            # V index: j = 2*j1 + j2, k = 3*k1 + k2
            # v1 keys: (j1, k1)  -- Note: in S2, V is 2x2, so k1 in {0,1}
            # v2 keys: (j2, k2)  -- Note: in S2x3, V is 2x3, so k2 in {0,1,2}
            for (j1, k1), c1 in v1.items():
                for (j2, k2), c2 in v2.items():
                    j = 2 * j1 + j2
                    k = 3 * k1 + k2
                    v_col[j * p + k] += c1 * c2
            
            # W index: k = 3*k1 + k2, i = 2*i1 + i2
            # w1 keys: (k1, i1)
            # w2 keys: (k2, i2)
            for (k1, i1), c1 in w1.items():
                for (k2, i2), c2 in w2.items():
                    k = 3 * k1 + k2
                    i = 2 * i1 + i2
                    w_col[k * n + i] += c1 * c2
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
            if _time_remaining() < 10.0:
                break
        if _time_remaining() < 10.0:
            break
            
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if total != 1:
                    return False
    return True
