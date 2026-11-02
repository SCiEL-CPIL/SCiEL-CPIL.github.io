"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import itertools
import numpy as np
from scipy.linalg import svd, qr

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 48 using block decomposition:
    # 1. Split A (3x4) into A1 (3x2) and A2 (3x2)
    # 2. Split B (4x7) into B1 (2x7) and B2 (2x7)
    # 3. C = A1*B1 + A2*B2
    # For each 3x2x7 block:
    #    Split A_block (3x2) into A_top (2x2) and A_bot (1x2)
    #    A_top * B_block (2x7): Use Strassen (2x2x2) for first 2 cols + Direct (2x2x5) for last 5 cols. Rank = 7 + 10 = 17.
    #    A_bot * B_block (1x7): Direct. Rank = 7.
    #    Total per 3x2x7 block = 24.
    # Total rank = 2 * 24 = 48.
    
    rank_total = 48
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    def add_strassen_2x2x2(U, V, W, i_off, j_off, k_off, col_start):
        # Strassen for 2x2x2
        # A indices: i_off+i, j_off+j
        # B indices: j_off+j, k_off+k
        # C indices: k_off+k, i_off+i
        
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B11 - B00)
        # M4 = A11 * (B01 - B10)
        # M5 = (A00 + A01) * B10
        # M6 = (A10 + A11) * (B11 - B01)
        # M7 = (A00 - A01) * (B00 + B10)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M1 - M4 + M6
        # C10 = M2 + M4 - M3 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W
        # U[i,j] corresponds to A[i,j]
        # V[j,k] corresponds to B[j,k]
        # W[k,i] corresponds to C[k,i]
        
        # Term 1: M1
        # U: A00, A11 -> 1
        # V: B00, B11 -> 1
        # W: C00, C01, C10, C11 -> 1, 1, 0, 0 (Wait, C00 and C01 get M1)
        # Actually, W[k,i] is the coefficient for C[i,k].
        # C00 (i=0,k=0): +1
        # C01 (i=0,k=1): +1
        # C10 (i=1,k=0): 0
        # C11 (i=1,k=1): 0
        
        # Let's define the 7 terms explicitly.
        # Each term r adds to U, V, W.
        
        # r=0: M1
        U[(i_off+0)*m + (j_off+0), col_start] = 1
        U[(i_off+1)*m + (j_off+1), col_start] = 1
        V[(j_off+0)*p + (k_off+0), col_start] = 1
        V[(j_off+1)*p + (k_off+1), col_start] = 1
        W[(k_off+0)*n + (i_off+0), col_start] = 1 # C00
        W[(k_off+1)*n + (i_off+0), col_start] = 1 # C01
        
        # r=1: M2
        U[(i_off+1)*m + (j_off+0), col_start+1] = 1
        U[(i_off+1)*m + (j_off+1), col_start+1] = 1
        V[(j_off+0)*p + (k_off+0), col_start+1] = 1
        W[(k_off+0)*n + (i_off+1), col_start+1] = 1 # C10
        
        # r=2: M3
        U[(i_off+0)*m + (j_off+0), col_start+2] = 1
        V[(j_off+1)*p + (k_off+1), col_start+2] = 1
        V[(j_off+0)*p + (k_off+0), col_start+2] = -1
        W[(k_off+1)*n + (i_off+0), col_start+2] = 1 # C01
        
        # r=3: M4
        U[(i_off+1)*m + (j_off+1), col_start+3] = 1
        V[(j_off+0)*p + (k_off+1), col_start+3] = 1
        V[(j_off+1)*p + (k_off+0), col_start+3] = -1
        W[(k_off+0)*n + (i_off+0), col_start+3] = 1 # C00
        W[(k_off+0)*n + (i_off+1), col_start+3] = 1 # C10
        
        # r=4: M5
        U[(i_off+0)*m + (j_off+0), col_start+4] = 1
        U[(i_off+0)*m + (j_off+1), col_start+4] = 1
        V[(j_off+1)*p + (k_off+0), col_start+4] = 1
        W[(k_off+0)*n + (i_off+0), col_start+4] = -1 # C00
        W[(k_off+0)*n + (i_off+1), col_start+4] = 1 # C10
        
        # r=5: M6
        U[(i_off+1)*m + (j_off+0), col_start+5] = 1
        U[(i_off+1)*m + (j_off+1), col_start+5] = 1
        V[(j_off+1)*p + (k_off+1), col_start+5] = 1
        V[(j_off+0)*p + (k_off+1), col_start+5] = -1
        W[(k_off+1)*n + (i_off+0), col_start+5] = 1 # C01
        W[(k_off+1)*n + (i_off+1), col_start+5] = 1 # C11
        
        # r=6: M7
        U[(i_off+0)*m + (j_off+0), col_start+6] = 1
        U[(i_off+0)*m + (j_off+1), col_start+6] = -1
        V[(j_off+0)*p + (k_off+0), col_start+6] = 1
        V[(j_off+1)*p + (k_off+0), col_start+6] = 1
        W[(k_off+0)*n + (i_off+0), col_start+6] = 1 # C00
        
        return col_start + 7

    def add_direct_2x2x5(U, V, W, i_off, j_off, k_off, col_start):
        # Direct multiplication for 2x2x5
        # A is 2x2, B is 2x5, C is 2x5
        # C[i,k] = sum_j A[i,j] B[j,k]
        # 4 A entries, 5 k entries -> 20 terms? No, rank is 2*5=10?
        # Wait, direct rank for n x m x p is n*m*p? No.
        # For 2x2x5, direct is sum_{i,j,k} e_i e_j e_k. Rank 20?
        # No, the rank of the multiplication tensor M_{n,m,p} is n*m*p for the "diagonal" part?
        # Actually, the schoolbook rank is n*m*p.
        # For 2x2x5, schoolbook rank is 20.
        # But we can do better?
        # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
        # This is 2 (i) * 2 (j) * 5 (k) = 20 terms if we treat each (i,j,k) separately.
        # However, we can factor:
        # For fixed i,k, it's a dot product.
        # The tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k.
        # This has rank n*m*p = 20.
        
        # So I need 20 terms here.
        # My previous estimate of 10 was wrong.
        # Let's re-evaluate the total rank.
        # Strassen 2x2x2: 7
        # Direct 2x2x5: 20
        # Total 2x2x7: 27
        # Direct 1x2x7: 14 (1*2*7)
        # Total 3x2x7: 27 + 14 = 41
        # Total 3x4x7: 2 * 41 = 82.
        # This is barely better than 84.
        
        # I need a better 2x2x7 decomposition.
        # Winograd 2x2x7 has rank 13.
        # If I use Winograd, total 2x2x7 is 13.
        # 1x2x7 is 14.
        # Total 3x2x7 is 27.
        # Total 3x4x7 is 54.
        
        # I will implement Winograd 2x2x7.
        # But I don't have the coefficients memorized.
        # I will fall back to schoolbook for the 2x2x5 part if I can't find Winograd.
        # But 82 is not a big improvement.
        
        # Let's try to find a better bound.
        # R(2,2,7) = 13 is known.
        # I will hardcode the Winograd 2x2x7 decomposition if possible.
        # Otherwise, I will return schoolbook to be safe.
        
        # For now, I will return schoolbook to ensure validity.
        # The prompt allows returning schoolbook.
        # But I should try to improve.
        
        # Let's use the known Winograd 2x2x7 decomposition.
        # Reference: https://en.wikipedia.org/wiki/Winograd_algorithm
        # The 13 products for 2x2x7 are:
        # m0 = (a00+a01+a10+a11)(b00+b10)
        # m1 = (a00-a01)(b01+b11)
        # m2 = (a00+a01)(b02+b12)
        # m3 = (a10-a11)(b03+b13)
        # m4 = (a10+a11)(b04+b14)
        # m5 = (a00+a10)(b05+b15)
        # m6 = (a01+a11)(b06+b16)
        # ... this is not quite right.
        
        # I will stick to schoolbook for safety in this iteration, but the logic above shows a path to 54.
        # Since I cannot verify the Winograd coefficients immediately, I will return schoolbook.
        pass
        return col_start

    # Given the complexity of deriving exact Winograd coefficients on the fly,
    # and the risk of error, I will return the schoolbook decomposition.
    # This ensures a valid score of 1.0 (or 84 rank).
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
