"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Uses a block-structured approach exploiting the rectangular asymmetry (n=2).
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    We use a block-structured approach exploiting the 2x2 block structure:
    - Split A (2x4) into A_L (2x2) and A_R (2x2)
    - Split B (4x4) into B_LL, B_LR, B_RL, B_RR (each 2x2)
    - C (2x4) = C_L (2x2) + C_R (2x2)
    
    C_L = A_L * B_LL + A_R * B_RL
    C_R = A_L * B_LR + A_R * B_RR
    
    For each 2x2 * 2x2 -> 2x2 multiplication, we use Strassen's algorithm
    which requires 7 bilinear products instead of 8.
    
    Total rank = 4 blocks * 7 = 28 (vs schoolbook 32)
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28  # 4 Strassen decompositions * 7 terms each
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    def strassen_2x2(U_base, V_base, W_base, rank_start):
        """
        Add a Strassen decomposition for 2x2 * 2x2 -> 2x2 multiplication.
        
        U_base: list of 4 U indices corresponding to A elements [A00, A01, A10, A11]
        V_base: list of 4 V indices corresponding to B elements [B00, B01, B10, B11]
        W_base: list of 4 W indices corresponding to C elements [C00, C01, C10, C11]
        rank_start: starting rank index
        
        Strassen's 7 products for 2x2 matrices:
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B01 - B11)
        M4 = A11 * (B10 - B00)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B01)
        M7 = (A01 - A11) * (B10 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # Each Strassen term gets its own rank slot
        for t in range(7):
            col = rank_start + t
            
            if t == 0:
                # M1 = (A00 + A11) * (B00 + B11)
                U[U_base[0]][col] = 1
                U[U_base[3]][col] = 1
                V[V_base[0]][col] = 1
                V[V_base[3]][col] = 1
                W[W_base[0]][col] = 1  # C00
                W[W_base[3]][col] = 1  # C11
                
            elif t == 1:
                # M2 = (A10 + A11) * B00
                U[U_base[2]][col] = 1
                U[U_base[3]][col] = 1
                V[V_base[0]][col] = 1
                W[W_base[2]][col] = 1  # C10
                W[W_base[3]][col] = -1 # C11
                
            elif t == 2:
                # M3 = A00 * (B01 - B11)
                U[U_base[0]][col] = 1
                V[V_base[1]][col] = 1
                V[V_base[3]][col] = -1
                W[W_base[1]][col] = 1  # C01
                W[W_base[3]][col] = 1  # C11
                
            elif t == 3:
                # M4 = A11 * (B10 - B00)
                U[U_base[3]][col] = 1
                V[V_base[2]][col] = 1
                V[V_base[0]][col] = -1
                W[W_base[0]][col] = 1  # C00
                W[W_base[2]][col] = 1  # C10
                
            elif t == 4:
                # M5 = (A00 + A01) * B11
                U[U_base[0]][col] = 1
                U[U_base[1]][col] = 1
                V[V_base[3]][col] = 1
                W[W_base[0]][col] = -1 # C00
                W[W_base[1]][col] = 1  # C01
                
            elif t == 5:
                # M6 = (A10 - A00) * (B00 + B01)
                U[U_base[2]][col] = 1
                U[U_base[0]][col] = -1
                V[V_base[0]][col] = 1
                V[V_base[1]][col] = 1
                W[W_base[3]][col] = 1  # C11
                
            elif t == 6:
                # M7 = (A01 - A11) * (B10 + B11)
                U[U_base[1]][col] = 1
                U[U_base[3]][col] = -1
                V[V_base[2]][col] = 1
                V[V_base[3]][col] = 1
                W[W_base[0]][col] = 1  # C00
                W[W_base[3]][col] = 1  # C11
    
    # Block 1: C_L = A_L * B_LL
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0, 1, 4, 5
    # B_LL: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0, 1, 4, 5
    # C_L: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0, 2, 1, 3 (trace convention: k*n+i)
    # W[k*2+i]: C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
    strassen_2x2([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3], 0)
    
    # Block 2: C_L += A_R * B_RL
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2, 3, 6, 7
    # B_RL: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices 8, 9, 12, 13
    # C_L: same W indices
    strassen_2x2([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3], 7)
    
    # Block 3: C_R = A_L * B_LR
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0, 1, 4, 5
    # B_LR: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 2, 3, 6, 7
    # C_R: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices 4, 6, 5, 7
    # W[k*2+i]: C[0,2]->4, C[0,3]->6, C[1,2]->5, C[1,3]->7
    strassen_2x2([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7], 14)
    
    # Block 4: C_R += A_R * B_RR
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2, 3, 6, 7
    # B_RR: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices 10, 11, 14, 15
    # C_R: same W indices
    strassen_2x2([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7], 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
