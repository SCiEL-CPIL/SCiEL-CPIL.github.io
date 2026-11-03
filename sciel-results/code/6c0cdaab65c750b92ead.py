import random
import time

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i

def verify(U, V, W, r):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for l in range(r):
                    s += U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if s != 1:
                    return False
    return True

def construct():
    # We will build a rank 44 decomposition.
    # Structure:
    # A is 3x4. Split into A1 (3x2) and A2 (3x2).
    # B is 4x6. Split into B1 (2x6) and B2 (2x6).
    # C = A1*B1 + A2*B2.
    # B1 is 2x6. Split into B11 (2x3), B12 (2x3).
    # B2 is 2x6. Split into B21 (2x3), B22 (2x3).
    # So we have 4 terms of type 3x2x3.
    # Wait, my previous analysis used 3x3x3 blocks.
    # Let's stick to the 3x3x3 + 3x1x3 construction which gave rank 42/44.
    
    # Let's use the 3x3x3 (rank 21) + 3x1x3 (rank 1) construction.
    # A is 3x4. Split cols into 3 and 1. A = [A_left (3x3), A_right (3x1)].
    # B is 4x6. Split rows into 3 and 1. B = [B_top (3x6); B_bottom (1x6)].
    # C = A_left * B_top + A_right * B_bottom.
    # A_left (3x3) * B_top (3x6).
    # Split B_top cols into 3 and 3. B_top = [B_top_left (3x3), B_top_right (3x3)].
    # C = A_left * B_top_left + A_left * B_top_right + A_right * B_bottom.
    # Term 1: 3x3x3 (Rank 21)
    # Term 2: 3x3x3 (Rank 21)
    # Term 3: 3x1x6.
    # A_right (3x1) * B_bottom (1x6).
    # Split B_bottom cols into 3 and 3. B_bottom = [B_bot_left (1x3), B_bot_right (1x3)].
    # Term 3a: 3x1x3 (Rank 1)
    # Term 3b: 3x1x3 (Rank 1)
    # Total Rank = 21 + 21 + 1 + 1 = 44.

    U = []
    V = []
    W = []
    
    # We need to define the local decompositions and map them to global indices.
    
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off, U_loc, V_loc, W_loc, r_loc):
        # U_loc is (n_b*m_b, r_loc)
        # V_loc is (m_b*p_b, r_loc)
        # W_loc is (p_b*n_b, r_loc)
        # Global U row: Aidx(i_off + i, j_off + j)
        # Global V row: Bidx(j_off + j, k_off + k)
        # Global W row: Cidx(i_off + i, k_off + k)
        
        for i in range(n_b):
            for j in range(m_b):
                global_u_row = Aidx(i_off + i, j_off + j)
                for l in range(r_loc):
                    if U_loc[i*m_b+j][l] != 0:
                        U[global_u_row][l + len(U[0])] = U_loc[i*m_b+j][l]
                        
        for j in range(m_b):
            for k in range(p_b):
                global_v_row = Bidx(j_off + j, k_off + k)
                for l in range(r_loc):
                    if V_loc[j*p_b+k][l] != 0:
                        V[global_v_row][l + len(V[0])] = V_loc[j*p_b+k][l]
                        
        for i in range(n_b):
            for k in range(p_b):
                global_w_row = Cidx(i_off + i, k_off + k)
                for l in range(r_loc):
                    if W_loc[k*n_b+i][l] != 0:
                        W[global_w_row][l + len(W[0])] = W_loc[k*n_b+i][l]

    # Initialize global matrices with 0s. We don't know the total rank yet, so we'll use lists of dicts or append.
    # Easier: collect all terms in a list, then assemble.
    
    terms = [] # list of (u_dict, v_dict, w_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))

    # --- Decomposition of 3x3x3 (Rank 21) ---
    # A (3x3) = [A11(2x2) A12(2x1); A21(1x2) A22(1x1)]
    # B (3x3) = [B11(2x2) B12(2x1); B21(1x2) B22(1x1)]
    # C = A11B11 + A11B12 + A12B21 + A12B22 + A21B11 + A21B12 + A22B21 + A22B22
    
    # 1. A11(2x2) * B11(2x2) -> C11(2x2). Rank 7 (Strassen)
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for 2x2x2 block at (0,0) of 3x3x3
    # A11 is A[0:2, 0:2]
    # B11 is B[0:2, 0:2]
    # C11 is C[0:2, 0:2]
    
    def add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        # Adds 7 terms
        A00 = (a_r, a_c); A01 = (a_r, a_c+1); A10 = (a_r+1, a_c); A11 = (a_r+1, a_c+1)
        B00 = (b_r, b_c); B01 = (b_r, b_c+1); B10 = (b_r+1, b_c); B11 = (b_r+1, b_c+1)
        C00 = (c_r, c_c); C01 = (c_r, c_c+1); C10 = (c_r+1, c_c); C11 = (c_r+1, c_c+1)
        
        # M1
        u = {A00:1, A11:1}
        v = {B00:1, B11:1}
        w = {C00:1, C11:1}
        add_term(u, v, w)
        
        # M2
        u = {A10:1, A11:1}
        v = {B00:1}
        w = {C10:1, C11:-1}
        add_term(u, v, w)
        
        # M3
        u = {A00:1}
        v = {B01:1, B11:-1}
        w = {C01:1, C11:1}
        add_term(u, v, w)
        
        # M4
        u = {A11:1}
        v = {B10:1, B00:-1}
        w = {C00:1, C10:1}
        add_term(u, v, w)
        
        # M5
        u = {A00:1, A01:1}
        v = {B11:1}
        w = {C00:-1, C01:1}
        add_term(u, v, w)
        
        # M6
        u = {A10:1, A00:-1}
        v = {B00:1, B01:1}
        w = {C11:1}
        add_term(u, v, w)
        
        # M7
        u = {A01:1, A11:-1}
        v = {B10:1, B11:1}
        w = {C00:1}
        add_term(u, v, w)

    # 2. A11(2x2) * B12(2x1) -> C12(2x1). Rank 4
    # C12 = A11 * B12.
    # C12_0 = A11_00*B12_0 + A11_01*B12_1
    # C12_1 = A11_10*B12_0 + A11_11*B12_1
    def add_2x2_x_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        # A is 2x2 at (a_r, a_c)
        # B is 2x1 at (b_r, b_c)
        # C is 2x1 at (c_r, c_c)
        A00 = (a_r, a_c); A01 = (a_r, a_c+1)
        A10 = (a_r+1, a_c); A11 = (a_r+1, a_c+1)
        B0 = (b_r, b_c); B1 = (b_r+1, b_c)
        C0 = (c_r, c_c); C1 = (c_r+1, c_c)
        
        # Term 1: A00*B0 -> C0
        add_term({A00:1}, {B0:1}, {C0:1})
        # Term 2: A01*B1 -> C0
        add_term({A01:1}, {B1:1}, {C0:1})
        # Term 3: A10*B0 -> C1
        add_term({A10:1}, {B0:1}, {C1:1})
        # Term 4: A11*B1 -> C1
        add_term({A11:1}, {B1:1}, {C1:1})

    # 3. A12(2x1) * B21(1x2) -> C21(2x2). Rank 1
    def add_2x1_x_1x2(a_r, a_c, b_r, b_c, c_r, c_c):
        # A is 2x1 at (a_r, a_c)
        # B is 1x2 at (b_r, b_c)
        # C is 2x2 at (c_r, c_c)
        A0 = (a_r, a_c); A1 = (a_r+1, a_c)
        B0 = (b_r, b_c); B1 = (b_r, b_c+1)
        C00 = (c_r, c_c); C01 = (c_r, c_c+1)
        C10 = (c_r+1, c_c); C11 = (c_r+1, c_c+1)
        
        u = {A0:1, A1:1}
        v = {B0:1, B1:1}
        w = {C00:1, C01:1, C10:1, C11:1}
        add_term(u, v, w)

    # 4. A12(2x1) * B22(1x1) -> C22(2x1). Rank 1
    def add_2x1_x_1x1(a_r, a_c, b_r, b_c, c_r, c_c):
        A0 = (a_r, a_c); A1 = (a_r+1, a_c)
        B0 = (b_r, b_c)
        C0 = (c_r, c_c); C1 = (c_r+1, c_c)
        
        u = {A0:1, A1:1}
        v = {B0:1}
        w = {C0:1, C1:1}
        add_term(u, v, w)

    # 5. A21(1x2) * B11(2x2) -> C11(2x2). Rank 4
    def add_1x2_x_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        A0 = (a_r, a_c); A1 = (a_r, a_c+1)
        B00 = (b_r, b_c); B01 = (b_r, b_c+1)
        B10 = (b_r+1, b_c); B11 = (b_r+1, b_c+1)
        C00 = (c_r, c_c); C01 = (c_r, c_c+1)
        
        # C00 = A0*B00 + A1*B10
        # C01 = A0*B01 + A1*B11
        add_term({A0:1}, {B00:1}, {C00:1})
        add_term({A1:1}, {B10:1}, {C00:1})
        add_term({A0:1}, {B01:1}, {C01:1})
        add_term({A1:1}, {B11:1}, {C01:1})

    # 6. A21(1x2) * B12(2x1) -> C12(2x1). Rank 2
    def add_1x2_x_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        A0 = (a_r, a_c); A1 = (a_r, a_c+1)
        B0 = (b_r, b_c); B1 = (b_r+1, b_c)
        C0 = (c_r, c_c)
        
        # C0 = A0*B0 + A1*B1
        add_term({A0:1}, {B0:1}, {C0:1})
        add_term({A1:1}, {B1:1}, {C0:1})

    # 7. A22(1x1) * B21(1x2) -> C21(2x2). Rank 1
    def add_1x1_x_1x2(a_r, a_c, b_r, b_c, c_r, c_c):
        A0 = (a_r, a_c)
        B0 = (b_r, b_c); B1 = (b_r, b_c+1)
        C00 = (c_r, c_c); C10 = (c_r+1, c_c)
        
        u = {A0:1}
        v = {B0:1, B1:1}
        w = {C00:1, C10:1}
        add_term(u, v, w)

    # 8. A22(1x1) * B22(1x1) -> C22(2x1). Rank 1
    def add_1x1_x_1x1(a_r, a_c, b_r, b_c, c_r, c_c):
        A0 = (a_r, a_c)
        B0 = (b_r, b_c)
        C0 = (c_r, c_c)
        
        add_term({A0:1}, {B0:1}, {C0:1})

    # --- Assembly for 3x3x3 block ---
    def add_3x3x3_block(i_off, j_off, k_off):
        # A11 (2x2) at (i_off, j_off)
        # B11 (2x2) at (j_off, k_off)
        # C11 (2x2) at (i_off, k_off)
        add_strassen_2x2(i_off, j_off, j_off, k_off, i_off, k_off)
        
        # A11 (2x2) * B12 (2x1) -> C12 (2x1)
        # B12 is at (j_off, k_off+2)
        # C12 is at (i_off, k_off+2)
        add_2x2_x_2x1(i_off, j_off, j_off, k_off+2, i_off, k_off+2)
        
        # A12 (2x1) * B21 (1x2) -> C21 (2x2)
        # A12 is at (i_off, j_off+2)
        # B21 is at (j_off+2, k_off)
        # C21 is at (i_off+2, k_off)
        add_2x1_x_1x2(i_off, j_off+2, j_off+2, k_off, i_off+2, k_off)
        
        # A12 (2x1) * B22 (1x1) -> C22 (2x1)
        # B22 is at (j_off+2, k_off+2)
        # C22 is at (i_off+2, k_off+2)
        add_2x1_x_1x1(i_off, j_off+2, j_off+2, k_off+2, i_off+2, k_off+2)
        
        # A21 (1x2) * B11 (2x2) -> C11 (2x2)
        # A21 is at (i_off+2, j_off)
        # B11 is at (j_off, k_off)
        # C11 is at (i_off, k_off)
        add_1x2_x_2x2(i_off+2, j_off, j_off, k_off, i_off, k_off)
        
        # A21 (1x2) * B12 (2x1) -> C12 (2x1)
        # B12 is at (j_off, k_off+2)
        # C12 is at (i_off, k_off+2)
        add_1x2_x_2x1(i_off+2, j_off, j_off, k_off+2, i_off, k_off+2)
        
        # A22 (1x1) * B21 (1x2) -> C21 (2x2)
        # A22 is at (i_off+2, j_off+2)
        # B21 is at (j_off+2, k_off)
        # C21 is at (i_off+2, k_off)
        add_1x1_x_1x2(i_off+2, j_off+2, j_off+2, k_off, i_off+2, k_off)
        
        # A22 (1x1) * B22 (1x1) -> C22 (2x1)
        # B22 is at (j_off+2, k_off+2)
        # C22 is at (i_off+2, k_off+2)
        add_1x1_x_1x1(i_off+2, j_off+2, j_off+2, k_off+2, i_off+2, k_off+2)

    # --- Assembly for 3x1x3 block (Rank 1) ---
    def add_3x1x3_block(i_off, j_off, k_off):
        # A (3x1) at (i_off, j_off)
        # B (1x3) at (j_off, k_off)
        # C (3x3) at (i_off, k_off)
        u = {}
        for i in range(3):
            u[(i_off+i, j_off)] = 1
        v = {}
        for k in range(3):
            v[(j_off, k_off+k)] = 1
        w = {}
        for i in range(3):
            for k in range(3):
                w[(i_off+i, k_off+k)] = 1
        add_term(u, v, w)

    # --- Main Assembly ---
    # A (3x4) = [A_left (3x3), A_right (3x1)]
    # B (4x6) = [B_top (3x6); B_bottom (1x6)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left is A[:, 0:3]. i_off=0, j_off=0.
    # B_top is B[0:3, :]. j_off=0, k_off=0.
    # B_top split into B_top_left (3x3) and B_top_right (3x3).
    
    # Term 1: A_left (3x3) * B_top_left (3x3) -> C_left (3x3)
    add_3x3x3_block(0, 0, 0)
    
    # Term 2: A_left (3x3) * B_top_right (3x3) -> C_right (3x3)
    # B_top_right is B[0:3, 3:6]. k_off=3.
    add_3x3x3_block(0, 0, 3)
    
    # Term 3: A_right (3x1) * B_bottom (1x6) -> C_bottom (3x6)
    # A_right is A[:, 3]. j_off=3.
    # B_bottom is B[3, :]. j_off=3.
    # B_bottom split into B_bot_left (1x3) and B_bot_right (1x3).
    
    # Term 3a: A_right (3x1) * B_bot_left (1x3) -> C_bot_left (3x3)
    # B_bot_left is B[3, 0:3]. k_off=0.
    add_3x1x3_block(0, 3, 0)
    
    # Term 3b: A_right (3x1) * B_bot_right (1x3) -> C_bot_right (3x3)
    # B_bot_right is B[3, 3:6]. k_off=3.
    add_3x1x3_block(0, 3, 3)

    # Now convert terms to U, V, W matrices
    r = len(terms)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for l, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[Aidx(i, j)][l] = val
        for (j, k), val in v_dict.items():
            V[Bidx(j, k)][l] = val
        for (i, k), val in w_dict.items():
            W[Cidx(i, k)][l] = val

    # Verify
    if not verify(U, V, W, r):
        return {"U": [[0]*66 for _ in range(n*m)], "V": [[0]*66 for _ in range(m*p)], "W": [[0]*66 for _ in range(p*n)]}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
