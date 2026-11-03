import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target Rank 50
    rank = 50
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a block
    def add_block(n_s, m_s, p_s, i_off, j_off, k_off, U_l, V_l, W_l):
        nonlocal col
        R_s = U_l.shape[1]
        for r in range(R_s):
            # Map local indices to global
            # Local a = i_s * m_s + j_s
            # Global a = (i_off + i_s) * m + (j_off + j_s)
            for i_s in range(n_s):
                for j_s in range(m_s):
                    a_l = i_s * m_s + j_s
                    a_g = (i_off + i_s) * m + (j_off + j_s)
                    U[a_g, col + r] = U_l[a_l, r]
            
            # Local b = j_s * p_s + k_s
            # Global b = (j_off + j_s) * p + (k_off + k_s)
            for j_s in range(m_s):
                for k_s in range(p_s):
                    b_l = j_s * p_s + k_s
                    b_g = (j_off + j_s) * p + (k_off + k_s)
                    V[b_g, col + r] = V_l[b_l, r]
            
            # Local c = k_s * n_s + i_s
            # Global c = (k_off + k_s) * n + (i_off + i_s)
            for k_s in range(p_s):
                for i_s in range(n_s):
                    c_l = k_s * n_s + i_s
                    c_g = (k_off + k_s) * n + (i_off + i_s)
                    W[c_g, col + r] = W_l[c_l, r]
        
        col += R_s

    # Define 2x5x1 decomposition (Rank 5)
    def get_2x5x1():
        n_s, m_s, p_s = 2, 5, 1
        R = 5
        U_l = np.zeros((n_s * m_s, R))
        V_l = np.zeros((m_s * p_s, R))
        W_l = np.zeros((p_s * n_s, R))
        
        for r in range(R):
            for j in range(m_s):
                # V
                V_l[j * p_s, r] = 1 if j == r else 0
                # U
                for i in range(n_s):
                    U_l[i * m_s + j, r] = 1 if j == r else 0
            for i in range(n_s):
                # W
                W_l[0 * n_s + i, r] = 1
        return U_l, V_l, W_l

    # Define 2x5x2 decomposition (Rank 10)
    def get_2x5x2():
        n_s, m_s, p_s = 2, 5, 2
        R = 10
        U_l = np.zeros((n_s * m_s, R))
        V_l = np.zeros((m_s * p_s, R))
        W_l = np.zeros((p_s * n_s, R))
        
        col_l = 0
        
        # Part 1: j=0,1,2 as 2x1x2 (Rank 1 each)
        for j in range(3):
            # 2x1x2 rank 1
            # A_{i,j} B_{j,k}
            # U_{i,j,0} = 1
            # V_{j,k,0} = 1
            # W_{k,i,0} = 1
            for i in range(n_s):
                U_l[i * m_s + j, col_l] = 1
            for k in range(p_s):
                V_l[j * p_s + k, col_l] = 1
            for k in range(p_s):
                for i in range(n_s):
                    W_l[k * n_s + i, col_l] = 1
            col_l += 1
            
        # Part 2: j=3,4 as 2x2x2 Strassen (Rank 7)
        # Local indices for this sub-sub-block:
        # j' = j - 3 (0,1)
        # We need to map Strassen terms to U, V, W
        # Strassen uses A_{i,j'} and B_{j',k}
        
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1. V: B00=1, B11=1. W: C00+=1, C11+=1
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col_l
            # u_coeffs: dict (i, j') -> val
            # v_coeffs: dict (j', k) -> val
            # w_coeffs: dict (k, i) -> val
            
            for (i, jp), val in u_coeffs.items():
                j = jp + 3
                U_l[i * m_s + j, col_l] = val
            for (jp, k), val in v_coeffs.items():
                j = jp + 3
                V_l[j * p_s + k, col_l] = val
            for (k, i), val in w_coeffs.items():
                W_l[k * n_s + i, col_l] = val
            col_l += 1

        # Strassen for 2x2x2
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,0), (0,1), (1,0), (1,1)
        
        # M1
        add_strassen_term(
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        )
        # M2
        add_strassen_term(
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1} # C10 gets M2? 
            # C10 = M2 + M7. C11 = M1 - M2 + M3 + M6.
            # Wait, standard Strassen:
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M7
            # C11 = M1 - M2 + M3 + M6
            # So M2 contributes to C10 (+1) and C11 (-1).
        )
        # Correction for M2 W
        W_l[0 * n_s + 1, col_l - 1] = 1 # C10 (k=0, i=1)
        W_l[1 * n_s + 1, col_l - 1] = -1 # C11 (k=1, i=1)
        
        # M3
        add_strassen_term(
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1} # C01 (+1), C11 (+1)
        )
        # M4
        add_strassen_term(
            {(1,0): 1},
            {(0,1): 1, (1,1): 1},
            {(0,0): 1} # C00 (+1)
        )
        # M5
        add_strassen_term(
            {(0,0): 1, (1,0): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1} # C00 (-1), C01 (+1)
        )
        # M6
        add_strassen_term(
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1},
            {(1,1): 1} # C11 (+1)
        )
        # M7
        add_strassen_term(
            {(1,1): 1, (1,0): -1},
            {(0,1): 1, (1,0): 1},
            {(0,0): -1, (0,1): 1} # C00 (-1), C10 (+1)
        )
        
        return U_l, V_l, W_l

    # Assemble 4x5x5
    # i in {0,1}, k in {0,1} -> 2x5x2 (i_off=0, k_off=0)
    U_l, V_l, W_l = get_2x5x2()
    add_block(2, 5, 2, 0, 0, 0, U_l, V_l, W_l)
    
    # i in {0,1}, k in {2,3} -> 2x5x2 (i_off=0, k_off=2)
    U_l, V_l, W_l = get_2x5x2()
    add_block(2, 5, 2, 0, 0, 2, U_l, V_l, W_l)
    
    # i in {0,1}, k in {4} -> 2x5x1 (i_off=0, k_off=4)
    U_l, V_l, W_l = get_2x5x1()
    add_block(2, 5, 1, 0, 0, 4, U_l, V_l, W_l)
    
    # i in {2,3}, k in {0,1} -> 2x5x2 (i_off=2, k_off=0)
    U_l, V_l, W_l = get_2x5x2()
    add_block(2, 5, 2, 2, 0, 0, U_l, V_l, W_l)
    
    # i in {2,3}, k in {2,3} -> 2x5x2 (i_off=2, k_off=2)
    U_l, V_l, W_l = get_2x5x2()
    add_block(2, 5, 2, 2, 0, 2, U_l, V_l, W_l)
    
    # i in {2,3}, k in {4} -> 2x5x1 (i_off=2, k_off=4)
    U_l, V_l, W_l = get_2x5x1()
    add_block(2, 5, 1, 2, 0, 4, U_l, V_l, W_l)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
