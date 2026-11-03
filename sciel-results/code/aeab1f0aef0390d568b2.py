import time
import numpy as np

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 130

def construct():
    """
    Finds a rank-78 exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Strategy:
    Split inner dimension m=4 into two 2s.
    Split output dimension p=7 into 2, 2, 3.
    This creates 6 blocks:
    4 blocks of 3x2x2 (Rank 11 each using Strassen for 2x2x2 part + Schoolbook for 1x2x2 part)
    2 blocks of 3x2x3 (Rank 17 each using Strassen for 2x2x2 part + Schoolbook for 2x2x1 and 1x2x3 parts)
    Total Rank = 4*11 + 2*17 = 44 + 34 = 78.
    """
    n, m, p = 3, 4, 7
    
    rank_total = 78
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    col_offset = 0
    
    def get_indices(i, j, k):
        return i * m + j, j * p + k, k * n + i

    def add_schoolbook(n_b, m_b, p_b, I_base, J_base, K_base):
        nonlocal col_offset
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx, v_idx, w_idx = get_indices(I_base + i, J_base + j, K_base + k)
                    U[u_idx, col_offset] = 1
                    V[v_idx, col_offset] = 1
                    W[w_idx, col_offset] = 1
                    col_offset += 1

    def add_strassen_2x2x2(I_base, J_base, K_base):
        nonlocal col_offset
        
        # Helper to add a specific term to U, V, W
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col_offset
            # u_coeffs: dict {(i,j): val}
            # v_coeffs: dict {(j,k): val}
            # w_coeffs: dict {(k,i): val}
            for (i, j), val in u_coeffs.items():
                u_idx, _, _ = get_indices(I_base + i, J_base + j, 0)
                U[u_idx, col_offset] = val
            for (j, k), val in v_coeffs.items():
                _, v_idx, _ = get_indices(0, J_base + j, K_base + k)
                V[v_idx, col_offset] = val
            for (k, i), val in w_coeffs.items():
                _, _, w_idx = get_indices(I_base + i, 0, K_base + k)
                W[w_idx, col_offset] = val
            col_offset += 1

        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
        
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1})
        
        # M3: A00(B01-B11) -> C01 += M3
        add_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1})
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        add_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1})
        
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        add_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1})
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        add_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1})
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        add_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})

    def add_3x2x2(I_base, J_base, K_base):
        # i in [0,2) part: 2x2x2 Strassen
        add_strassen_2x2x2(I_base, J_base, K_base)
        # i in [2,3) part: 1x2x2 Schoolbook
        add_schoolbook(1, 2, 2, I_base + 2, J_base, K_base)

    def add_3x2x3(I_base, J_base, K_base):
        # i in [0,2), k in [0,2) part: 2x2x2 Strassen
        add_strassen_2x2x2(I_base, J_base, K_base)
        # i in [0,2), k in [2,3) part: 2x2x1 Schoolbook
        add_schoolbook(2, 2, 1, I_base, J_base, K_base + 2)
        # i in [2,3), k in [0,3) part: 1x2x3 Schoolbook
        add_schoolbook(1, 2, 3, I_base + 2, J_base, K_base)

    # Block 1: 3x2x2 at I=0, J=0, K=0
    add_3x2x2(0, 0, 0)
    
    # Block 2: 3x2x2 at I=0, J=0, K=2
    add_3x2x2(0, 0, 2)
    
    # Block 3: 3x2x3 at I=0, J=0, K=4
    add_3x2x3(0, 0, 4)
    
    # Block 4: 3x2x2 at I=0, J=2, K=0
    add_3x2x2(0, 2, 0)
    
    # Block 5: 3x2x2 at I=0, J=2, K=2
    add_3x2x2(0, 2, 2)
    
    # Block 6: 3x2x3 at I=0, J=2, K=4
    add_3x2x3(0, 2, 4)
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
