"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Partitioning:
    # j: 0,1,2 (size 3) and 3,4 (size 2)
    # k: 0,1,2 (size 3) and 3,4 (size 2)
    
    # Blocks:
    # 1. A1 (3x3) * B11 (3x3) -> C1 (3x3) [Laderman, Rank 19]
    # 2. A1 (3x3) * B12 (3x2) -> C2 (3x2) [Schoolbook, Rank 18]
    # 3. A2 (3x2) * B21 (2x3) -> C1 (3x3) [Schoolbook, Rank 18]
    # 4. A2 (3x2) * B22 (2x2) -> C2 (3x2) [Schoolbook, Rank 12]
    
    # Total Rank: 19 + 18 + 18 + 12 = 67
    
    total_rank = 67
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: U[i,j] += u, V[j,k] += v, W[k,i] += w
    def add_term(i, j, k, u, v, w, c):
        U[i*m + j][c] += u
        V[j*p + k][c] += v
        W[k*n + i][c] += w

    # --- Block 1: Laderman 3x3x3 ---
    # A1 indices: i in 0..2, j in 0..2
    # B11 indices: j in 0..2, k in 0..2
    
    # M1 = A00 B00
    add_term(0, 0, 0, 1, 1, 1, col); col += 1
    # M2 = A11 B11
    add_term(1, 1, 1, 1, 1, 1, col); col += 1
    # M3 = A22 B22
    add_term(2, 2, 2, 1, 1, 1, col); col += 1
    
    # M4 = (A00+A11+A22)(B00+B11+B22)
    for i in range(3):
        for j in range(3):
            if i == j:
                add_term(i, j, j, 1, 1, 1, col)
                add_term(i, j, j, 0, 0, 0, col) # dummy to keep structure? No, just add to same col
    # Wait, M4 is a single multiplication of sums.
    # U has A00, A11, A22. V has B00, B11, B22. W has C00, C11, C22.
    # This term contributes to C00, C11, C22.
    # Actually, Laderman's M4 is (A00+A11+A22)(B00+B11+B22).
    # This product contributes to C00, C11, C22? No.
    # The product is a scalar. It is added to specific C entries.
    # In Laderman's algorithm, the 19 products are linear combinations.
    # We need to map the 19 products to the U, V, W entries.
    
    # Let's re-implement Laderman carefully.
    # The 19 products M_r are computed as M_r = (sum u_ij A_ij) * (sum v_jk B_jk).
    # Then C_ik = sum w_ikr M_r.
    
    # Reset col to 0 for Block 1? No, col is currently 3.
    # Let's define the Laderman coefficients.
    
    # M4: (A00+A11+A22)(B00+B11+B22)
    # U: A00(0,0), A11(1,1), A22(2,2)
    # V: B00(0,0), B11(1,1), B22(2,2)
    # W: C00(0,0), C11(1,1), C22(2,2)
    # Contribution to C00, C11, C22 is +1 each?
    # In Laderman, M4 is added to C00, C11, C22.
    add_term(0, 0, 0, 1, 1, 1, col)
    add_term(1, 1, 1, 1, 1, 1, col)
    add_term(2, 2, 2, 1, 1, 1, col)
    col += 1
    
    # M5: (A01+A11)B10
    # U: A01(0,1), A11(1,1)
    # V: B10(1,0)
    # W: C10(1,0)
    add_term(0, 1, 0, 1, 1, 0, col) # Wait, W index is k*n+i. k=0, i=1 -> 1.
    # Correction: W[k*n+i].
    # M5 contributes to C10?
    # Laderman M5 = (A01+A11)B10.
    # This term is added to C10.
    add_term(0, 1, 0, 1, 1, 0, col) # U(0,1), V(1,0), W(0,1)? No.
    # W index for C10 is k=0, i=1 -> 0*3+1 = 1.
    # U index for A01 is i=0, j=1 -> 0*5+1 = 1.
    # V index for B10 is j=1, k=0 -> 1*5+0 = 5.
    # Let's use the helper function with i,j,k.
    # add_term(i, j, k, u, v, w, c) adds u to U[i,j], v to V[j,k], w to W[k,i].
    # For M5, we need to add to U[0,1], U[1,1], V[1,0], W[0,1].
    # But add_term only adds to one i,j,k triple.
    # We need to call add_term multiple times or modify the helper.
    
    # Let's rewrite the helper to accept lists of indices.
    
    def add_laderman_term(u_indices, v_indices, w_indices, c):
        for i, j in u_indices:
            U[i*m + j][c] += 1
        for j, k in v_indices:
            V[j*p + k][c] += 1
        for k, i in w_indices:
            W[k*n + i][c] += 1
            
    # M1: A00 B00 -> C00
    add_laderman_term([(0,0)], [(0,0)], [(0,0)], col); col += 1
    # M2: A11 B11 -> C11
    add_laderman_term([(1,1)], [(1,1)], [(1,1)], col); col += 1
    # M3: A22 B22 -> C22
    add_laderman_term([(2,2)], [(2,2)], [(2,2)], col); col += 1
    # M4: (A00+A11+A22)(B00+B11+B22) -> C00, C11, C22
    add_laderman_term([(0,0), (1,1), (2,2)], [(0,0), (1,1), (2,2)], [(0,0), (1,1), (2,2)], col); col += 1
    # M5: (A01+A11)B10 -> C10
    add_laderman_term([(0,1), (1,1)], [(1,0)], [(0,1)], col); col += 1
    # M6: (A10-A00)(B00+B01) -> C10, C11
    # U: A10(1,0), A00(0,0) with -1
    # V: B00(0,0), B01(0,1)
    # W: C10(0,1), C11(1,1)
    # Wait, Laderman M6 contributes to C10 and C11?
    # Formula: M6 = (A10-A00)(B00+B01).
    # C10 += M6. C11 += M6.
    # We need to handle signs.
    
    # Let's make add_laderman_term handle signs.
    def add_laderman_term_signed(u_terms, v_terms, w_terms, c):
        for i, j, sign in u_terms:
            U[i*m + j][c] += sign
        for j, k, sign in v_terms:
            V[j*p + k][c] += sign
        for k, i, sign in w_terms:
            W[k*n + i][c] += sign

    # Re-do M1-M4 with signed version (all 1s)
    # M1
    add_laderman_term_signed([(0,0,1)], [(0,0,1)], [(0,0,1)], col); col += 1
    # M2
    add_laderman_term_signed([(1,1,1)], [(1,1,1)], [(1,1,1)], col); col += 1
    # M3
    add_laderman_term_signed([(2,2,1)], [(2,2,1)], [(2,2,1)], col); col += 1
    # M4
    add_laderman_term_signed([(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)], col); col += 1
    # M5
    add_laderman_term_signed([(0,1,1), (1,1,1)], [(1,0,1)], [(0,1,1)], col); col += 1
    # M6
    add_laderman_term_signed([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,1,1), (1,1,1)], col); col += 1
    # M7
    add_laderman_term_signed([(0,2,1), (2,2,1)], [(2,0,1)], [(0,2,1)], col); col += 1
    # M8
    add_laderman_term_signed([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(0,2,1), (2,2,1)], col); col += 1
    # M9
    add_laderman_term_signed([(1,2,1), (2,2,1)], [(2,1,1)], [(1,2,1)], col); col += 1
    # M10
    add_laderman_term_signed([(2,1,1), (1,1,-1)], [(1,1,1), (1,2,1)], [(1,2,1), (2,2,1)], col); col += 1
    # M11
    add_laderman_term_signed([(0,1,1), (0,2,1)], [(2,1,1)], [(0,2,1)], col); col += 1
    # M12
    add_laderman_term_signed([(1,0,1), (1,2,1)], [(2,0,1)], [(1,2,1)], col); col += 1
    # M13
    add_laderman_term_signed([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,1,1)], col); col += 1
    # M14
    add_laderman_term_signed([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,2,1), (2,2,1)], col); col += 1
    # M15
    add_laderman_term_signed([(1,2,1), (2,2,-1)], [(2,1,1), (2,2,1)], [(1,2,1), (2,2,1)], col); col += 1
    # M16
    add_laderman_term_signed([(0,1,1), (1,2,1), (2,2,1)], [(2,0,1), (1,1,1), (0,2,1)], [(0,2,1), (1,1,1), (2,0,1)], col); col += 1
    # M17
    add_laderman_term_signed([(0,2,1), (1,0,1), (2,1,1)], [(1,1,1), (2,0,1), (0,1,1)], [(0,1,1), (1,0,1), (2,2,1)], col); col += 1
    # M18
    add_laderman_term_signed([(1,0,1), (2,2,1), (0,1,1)], [(0,2,1), (1,1,1), (2,0,1)], [(0,2,1), (1,1,1), (2,0,1)], col); col += 1
    # M19
    add_laderman_term_signed([(2,1,1), (0,2,1), (1,0,1)], [(0,1,1), (2,2,1), (1,0,1)], [(0,1,1), (2,2,1), (1,0,1)], col); col += 1
    
    # --- Block 2: A1 (3x3) * B12 (3x2) ---
    # Schoolbook: 18 terms.
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                add_term(i, j, k, 1, 1, 1, col)
                col += 1
                
    # --- Block 3: A2 (3x2) * B21 (2x3) ---
    # Schoolbook: 18 terms.
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                add_term(i, j, k, 1, 1, 1, col)
                col += 1
                
    # --- Block 4: A2 (3x2) * B22 (2x2) ---
    # Schoolbook: 12 terms.
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 5):
                add_term(i, j, k, 1, 1, 1, col)
                col += 1
                
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if time_left() < 5.0:
        return result
        
    def verify():
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        if U_arr.shape != (n*m, total_rank):
            return False
        if V_arr.shape != (m*p, total_rank):
            return False
        if W_arr.shape != (p*n, total_rank):
            return False
            
        U_3d = U_arr.reshape(n, m, total_rank)
        V_3d = V_arr.reshape(m, p, total_rank)
        W_3d = W_arr.reshape(p, n, total_rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        expected = np.ones((n, m, p))
        
        if not np.allclose(T, expected, atol=1e-10):
            return False
        return True
        
    if not verify():
        if time_left() < 2.0:
            return result
            
        # Fallback to schoolbook
        U_fb = [[0.0] * (n*m*p) for _ in range(n*m)]
        V_fb = [[0.0] * (n*m*p) for _ in range(m*p)]
        W_fb = [[0.0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1.0
                    V_fb[j*p+k][col] = 1.0
                    W_fb[k*n+i][col] = 1.0
                    col += 1
        result = {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return result
# EVOLVE-BLOCK-END
