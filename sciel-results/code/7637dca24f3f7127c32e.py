import numpy as np
import time

def construct():
    """
    Construct the rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
    The minimum rank is 56.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We decompose the multiplication into 8 independent 2x2 * 2x2 blocks.
    # A is 2x4. Split into A_L (2x2) and A_R (2x2).
    # B is 4x8. Split into B_L (2x8) and B_R (2x8).
    # B_L is split into 4 blocks of 2x2: B_L0, B_L1, B_L2, B_L3.
    # B_R is split into 4 blocks of 2x2: B_R0, B_R1, B_R2, B_R3.
    
    # For each pair (A_part, B_part), we apply Strassen.
    # There are 2 A_parts (L, R) and 4 B_parts per A_part.
    # Total 8 Strassen instances.
    
    col = 0
    for a_part in range(2):
        # a_part 0 corresponds to A[:, 0:2] (indices 0,1 in j)
        # a_part 1 corresponds to A[:, 2:4] (indices 2,3 in j)
        
        # A indices:
        # i=0, j=2*a_part -> a00
        # i=0, j=2*a_part+1 -> a01
        # i=1, j=2*a_part -> a10
        # i=1, j=2*a_part+1 -> a11
        
        a00_idx = 0 * m + (2 * a_part)
        a01_idx = 0 * m + (2 * a_part + 1)
        a10_idx = 1 * m + (2 * a_part)
        a11_idx = 1 * m + (2 * a_part + 1)
        
        for b_block in range(4):
            # b_block 0..3 corresponds to columns 2*b_block, 2*b_block+1 of B
            # B rows are determined by a_part?
            # Wait.
            # C = A_L B_L + A_R B_R.
            # A_L is A[:, 0:2]. It multiplies B_L which is B[0:2, :].
            # So for a_part=0 (A_L), we use B rows 0,1.
            # For a_part=1 (A_R), we use B rows 2,3.
            
            b_row_start = 2 * a_part
            
            b00_idx = b_row_start * p + (2 * b_block)
            b01_idx = b_row_start * p + (2 * b_block + 1)
            b10_idx = (b_row_start + 1) * p + (2 * b_block)
            b11_idx = (b_row_start + 1) * p + (2 * b_block + 1)
            
            # Output C indices:
            # C is 2x8.
            # The result of this block contributes to C[:, 2*b_block : 2*b_block+2].
            # c00: i=0, k=2*b_block
            # c01: i=0, k=2*b_block+1
            # c10: i=1, k=2*b_block
            # c11: i=1, k=2*b_block+1
            
            c00_idx = (2 * b_block) * n + 0
            c01_idx = (2 * b_block + 1) * n + 0
            c10_idx = (2 * b_block) * n + 1
            c11_idx = (2 * b_block + 1) * n + 1
            
            # Strassen Multiplications
            # m1 = (a00+a11)*(b00+b11)
            U[a00_idx][col] += 1
            U[a11_idx][col] += 1
            V[b00_idx][col] += 1
            V[b11_idx][col] += 1
            W[c00_idx][col] += 1
            W[c11_idx][col] += 1
            
            # m2 = (a10+a11)*b00
            U[a10_idx][col+1] += 1
            U[a11_idx][col+1] += 1
            V[b00_idx][col+1] += 1
            W[c10_idx][col+1] += 1
            W[c11_idx][col+1] += -1
            
            # m3 = a00*(b01-b11)
            U[a00_idx][col+2] += 1
            V[b01_idx][col+2] += 1
            V[b11_idx][col+2] += -1
            W[c01_idx][col+2] += 1
            W[c11_idx][col+2] += 1
            
            # m4 = a11*b10
            U[a11_idx][col+3] += 1
            V[b10_idx][col+3] += 1
            W[c00_idx][col+3] += 1
            W[c10_idx][col+3] += 1
            
            # m5 = (a00+a01)*b11
            U[a00_idx][col+4] += 1
            U[a01_idx][col+4] += 1
            V[b11_idx][col+4] += 1
            W[c00_idx][col+4] += -1
            W[c01_idx][col+4] += 1
            
            # m6 = (a10-a00)*(b00+b01)
            U[a10_idx][col+5] += 1
            U[a00_idx][col+5] += -1
            V[b00_idx][col+5] += 1
            V[b01_idx][col+5] += 1
            W[c11_idx][col+5] += 1
            
            # m7 = (a01-a11)*(b10+b11)
            U[a01_idx][col+6] += 1
            U[a11_idx][col+6] += -1
            V[b10_idx][col+6] += 1
            V[b11_idx][col+6] += 1
            W[c00_idx][col+6] += 1
            W[c11_idx][col+6] += 1
            
            col += 7
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
