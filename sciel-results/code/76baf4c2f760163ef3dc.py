import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of (idx, val)
        v_vec: list of (idx, val)
        w_vec: list of (idx, val)
        """
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def schoolbook(n_prime, m_prime, p_prime, i_offset, j_offset, k_offset):
        """
        Adds schoolbook decomposition for a sub-block.
        """
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])
                    
    def strassen_2x2x2(i_offset, j_offset, k_offset):
        """
        Adds Strassen decomposition for a 2x2x2 sub-block.
        """
        def gI(i_l, j_l): return (i_offset + i_l) * m + (j_offset + j_l)
        def gJ(j_l, k_l): return (j_offset + j_l) * p + (k_offset + k_l)
        def gK(k_l, i_l): return (k_offset + k_l) * n + (i_offset + i_l)
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        
        # M3: A00(B01-B11) -> C01, C11
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        
        # M4: A11(B10-B00) -> C00, C10
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )

    def strassen_2x2x2_offset(i_off, j_off, k_off, i_start, j_start, k_start):
        """
        Adds Strassen for 2x2x2 block at offsets within the larger block.
        i_off, j_off, k_off are offsets within the sub-block.
        """
        def gI(i_l, j_l): return (i_start + i_off + i_l) * m + (j_start + j_off + j_l)
        def gJ(j_l, k_l): return (j_start + j_off + j_l) * p + (k_start + k_off + k_l)
        def gK(k_l, i_l): return (k_start + k_off + k_l) * n + (i_start + i_off + i_l)
        
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )

    def handle_2x2x3(i_start, j_start, k_start):
        """
        2x2x3: A is 2x2, B is 2x3, C is 2x3.
        Split B into B_left (2x2) and B_right (2x1).
        C = A*B_left + A*B_right
        A*B_left: 2x2x2 Strassen (rank 7)
        A*B_right: 2x2x1 schoolbook (rank 4)
        """
        # Strassen for 2x2x2 part (k indices 0,1)
        strassen_2x2x2_offset(0, 0, 0, i_start, j_start, k_start)
        # Schoolbook for 2x2x1 part (k index 2)
        schoolbook(2, 2, 1, i_start, j_start, k_start + 2)

    def handle_2x3x2(i_start, j_start, k_start):
        """
        2x3x2: A is 2x3, B is 3x2, C is 2x2.
        Split A into A_left (2x2) and A_right (2x1).
        Split B into B_top (2x2) and B_bottom (1x2).
        C = A_left*B_top + A_right*B_bottom
        A_left*B_top: 2x2x2 Strassen (rank 7)
        A_right*B_bottom: 2x1x2 schoolbook (rank 4)
        """
        # Strassen for 2x2x2 part (j indices 0,1; k indices 0,1)
        strassen_2x2x2_offset(0, 0, 0, i_start, j_start, k_start)
        # Schoolbook for 2x1x2 part (j index 2; k indices 0,1)
        schoolbook(2, 1, 2, i_start, j_start + 2, k_start)

    def handle_2x3x3(i_start, j_start, k_start):
        """
        2x3x3: A is 2x3, B is 3x3, C is 2x3.
        Split A into A_left (2x2) and A_right (2x1).
        Split B into B_top (2x3) and B_bottom (1x3).
        C = A_left*B_top + A_right*B_bottom
        A_left*B_top: 2x2x3 (rank 11)
        A_right*B_bottom: 2x1x3 (rank 3)
        """
        # 2x2x3 part: A_left (2x2) * B_top (2x3)
        # Strassen for 2x2x2 (j 0,1; k 0,1)
        strassen_2x2x2_offset(0, 0, 0, i_start, j_start, k_start)
        # Schoolbook for 2x2x1 (j 0,1; k 2)
        schoolbook(2, 2, 1, i_start, j_start, k_start + 2)
        # 2x1x3 part: A_right (2x1) * B_bottom (1x3)
        # C[i,k] += A_right[i] * B_bottom[k] for i in {0,1}, k in {0,1,2}
        # This is rank 3
        for i_loc in range(2):
            for k_loc in range(3):
                I = (i_start + i_loc) * m + (j_start + 2)
                J = (j_start + 2) * p + (k_start + k_loc)
                K = (k_start + k_loc) * n + (i_start + i_loc)
                add_term([(I, 1)], [(J, 1)], [(K, 1)])

    # Partitioning:
    # I: [0,1], [2,3]
    # J: [0,1,2], [3,4]
    # K: [0,1,2], [3,4]
    
    i_groups = [(0, 2), (2, 2)] # (start, size)
    j_groups = [(0, 3), (3, 2)]
    k_groups = [(0, 3), (3, 2)]
    
    for i_start, i_size in i_groups:
        for j_start, j_size in j_groups:
            for k_start, k_size in k_groups:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    strassen_2x2x2(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 2 and k_size == 3:
                    handle_2x2x3(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 3 and k_size == 2:
                    handle_2x3x2(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 3 and k_size == 3:
                    handle_2x3x3(i_start, j_start, k_start)
                else:
                    schoolbook(i_size, j_size, k_size, i_start, j_start, k_start)
    
    if not U_cols:
        return {"U": [[0]*100 for _ in range(20)], "V": [[0]*100 for _ in range(25)], "W": [[0]*100 for _ in range(20)]}
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
