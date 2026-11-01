import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 46
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Rank 7
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3,4} -> Rank 11 (Strassen for k in {2,3}, Schoolbook for k=4)
    # Block 3: i in {0,1}, j in {2,3,4}, k in {0,1} -> Rank 11 (Strassen for j in {2,3}, Schoolbook for j=4)
    # Block 4: i in {0,1}, j in {2,3,4}, k in {2,3,4} -> Rank 17 (Strassen for j in {2,3}, k in {2,3}, Schoolbook for rest)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen(i_off, j_off, k_off):
        # Adds 7 columns for a 2x2x2 block starting at i_off, j_off, k_off
        # Local indices il, jl, kl in {0,1}
        # Global indices i = i_off + il, j = j_off + jl, k = k_off + kl
        
        # Strassen matrices U, V, W (local)
        # U: 2x2, V: 2x2, W: 2x2
        U_local = [
            [[1, 0], [0, 1]], # M1
            [[0, 0], [1, 1]], # M2
            [[1, 0], [0, 0]], # M3
            [[0, 0], [0, 1]], # M4
            [[1, 1], [0, 0]], # M5
            [[-1, 0], [1, 0]], # M6
            [[0, 1], [0, -1]]  # M7
        ]
        V_local = [
            [[1, 0], [0, 1]], # M1
            [[1, 0], [0, 0]], # M2
            [[0, 1], [0, -1]], # M3
            [[-1, 0], [1, 0]], # M4
            [[0, 0], [0, 1]], # M5
            [[1, 1], [0, 0]], # M6
            [[0, 0], [1, 1]]  # M7
        ]
        W_local = [
            [[1, 0], [0, 1]], # M1: C00+=1, C11+=1
            [[0, 0], [1, -1]], # M2: C10+=1, C11-=-1
            [[0, 1], [0, 1]], # M3: C01+=1, C11+=1
            [[1, 0], [1, 0]], # M4: C00+=1, C10+=1
            [[-1, 1], [0, 0]], # M5: C00-=-1, C01+=1
            [[0, 0], [0, 1]], # M6: C11+=1
            [[1, 0], [0, 0]]  # M7: C00+=1
        ]
        
        for r in range(7):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for il in range(2):
                for jl in range(2):
                    i = i_off + il
                    j = j_off + jl
                    u_col[i * m + j] = U_local[r][il][jl]
                    
            for jl in range(2):
                for kl in range(2):
                    j = j_off + jl
                    k = k_off + kl
                    v_col[j * p + k] = V_local[r][jl][kl]
                    
            for kl in range(2):
                for il in range(2):
                    k = k_off + kl
                    i = i_off + il
                    w_col[k * n + i] = W_local[r][il][kl] # Note: W_local index is [i][k] in my derivation?
                    # My derivation: W_{r,ik}. So W_local[r][il][kl].
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    def add_schoolbook(i_off, i_count, j_off, j_count, k_off, k_count):
        # Adds i_count * j_count * k_count columns
        for il in range(i_count):
            for jl in range(j_count):
                for kl in range(k_count):
                    i = i_off + il
                    j = j_off + jl
                    k = k_off + kl
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    
                    u_col[i * m + j] = 1
                    v_col[j * p + k] = 1
                    w_col[k * n + i] = 1
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3,4}
    # Strassen for k in {2,3} (local k 0,1)
    add_strassen(0, 0, 2)
    # Schoolbook for k=4
    add_schoolbook(0, 2, 0, 2, 4, 1)
    
    # Block 3: i in {0,1}, j in {2,3,4}, k in {0,1}
    # Strassen for j in {2,3} (local j 0,1)
    # Note: Strassen function assumes j_off, k_off. Here j is the variable for Strassen.
    # My add_strassen takes i_off, j_off, k_off.
    # In Block 3, the 2x2x2 block is on j and k?
    # No, Strassen is for 2x2x2.
    # The block is i in {0,1}, j in {2,3}, k in {0,1}.
    # This is 2 (i) x 2 (j) x 2 (k).
    add_strassen(0, 2, 0)
    # Schoolbook for j=4
    add_schoolbook(0, 2, 4, 1, 0, 2)
    
    # Block 4: i in {0,1}, j in {2,3,4}, k in {2,3,4}
    # Strassen for j in {2,3}, k in {2,3}
    add_strassen(0, 2, 2)
    # Schoolbook for j in {2,3}, k=4
    add_schoolbook(0, 2, 2, 2, 4, 1)
    # Schoolbook for j=4, k in {2,3}
    add_schoolbook(0, 2, 4, 1, 2, 2)
    # Schoolbook for j=4, k=4
    add_schoolbook(0, 2, 4, 1, 4, 1)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
