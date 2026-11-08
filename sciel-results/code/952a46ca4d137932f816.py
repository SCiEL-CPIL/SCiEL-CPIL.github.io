import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # rank = 51
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # We have 3 blocks of 3 columns in B (and C)
    for block in range(3):
        # Block corresponds to columns 3*block to 3*block+2
        # Local k indices: 0, 1, 2
        # Global k indices: 3*block + local_k
        
        # Part 1: T_2,2,3 (j=0,1)
        # Rank 11
        # Factors 1-7: Strassen for 2x2x2 (local k=0,1)
        # Factors 8-11: Schoolbook for 2x2x1 (local k=2)
        
        # Strassen factors for 2x2x2
        # m1 = a00(b00+b01)
        U[0*2+0][col_offset] = 1
        V[0*3+0][col_offset] = 1
        V[0*3+1][col_offset] = 1
        W[0*2+0][col_offset] = 1
        W[1*2+0][col_offset] = 1
        col_offset += 1
        
        # m2 = (a00+a01)b10
        U[0*2+0][col_offset] = 1
        U[0*2+1][col_offset] = 1
        V[1*3+0][col_offset] = 1
        W[0*2+0][col_offset] = 1
        W[1*2+0][col_offset] = 1
        col_offset += 1
        
        # m3 = (a10+a11)b00
        U[1*2+0][col_offset] = 1
        U[1*2+1][col_offset] = 1
        V[0*3+0][col_offset] = 1
        W[0*2+0][col_offset] = 1
        W[1*2+0][col_offset] = 1
        col_offset += 1
        
        # m4 = a11(b10+b11)
        U[1*2+1][col_offset] = 1
        V[1*3+0][col_offset] = 1
        V[1*3+1][col_offset] = 1
        W[0*2+0][col_offset] = 1
        W[1*2+0][col_offset] = 1
        col_offset += 1
        
        # m5 = (a00+a11)(b00+b11)
        U[0*2+0][col_offset] = 1
        U[1*2+1][col_offset] = 1
        V[0*3+0][col_offset] = 1
        V[1*3+1][col_offset] = 1
        W[0*2+0][col_offset] = 1
        W[1*2+0][col_offset] = 1
        col_offset += 1
        
        # m6 = (a01-a11)(b10+b11)
        U[0*2+1][col_offset] = 1
        U[1*2+1][col_offset] = -1
        V[1*3+0][col_offset] = 1
        V[1*3+1][col_offset] = 1
        W[0*2+0][col_offset] = 1
        W[1*2+0][col_offset] = 1
        col_offset += 1
        
        # m7 = (a00+a10)(b00-b10)
        U[0*2+0][col_offset] = 1
        U[1*2+0][col_offset] = 1
        V[0*3+0][col_offset] = 1
        V[1*3+0][col_offset] = -1
        W[0*2+0][col_offset] = 1
        W[1*2+0][col_offset] = 1
        col_offset += 1
        
        # Now we need to map these to the correct W indices for local k=0,1
        # Wait, the W indices in the above Strassen factors are for a 2x2x2 tensor.
        # The output C is 2x2.
        # C00, C01, C10, C11.
        # In our global tensor, for this block, the output columns are 3*block, 3*block+1.
        # So local k=0 -> global k=3*block.
        # local k=1 -> global k=3*block+1.
        
        # I need to correct the W indices in the Strassen factors above.
        # The W index is k*n + i.
        # For local k=0, W index is (3*block)*2 + i.
        # For local k=1, W index is (3*block+1)*2 + i.
        
        # Let's restart the block construction with correct indices.
        # I will revert col_offset to the start of this block.
        col_offset -= 7
        
        # Redo Strassen with correct W indices
        # m1 = a00(b00+b01)
        U[0*2+0][col_offset] = 1
        V[0*3+0][col_offset] = 1
        V[0*3+1][col_offset] = 1
        W[(3*block)*2+0][col_offset] = 1
        W[(3*block+1)*2+0][col_offset] = 1
        col_offset += 1
        
        # m2 = (a00+a01)b10
        U[0*2+0][col_offset] = 1
        U[0*2+1][col_offset] = 1
        V[1*3+0][col_offset] = 1
        W[(3*block)*2+0][col_offset] = 1
        W[(3*block+1)*2+0][col_offset] = 1
        col_offset += 1
        
        # m3 = (a10+a11)b00
        U[1*2+0][col_offset] = 1
        U[1*2+1][col_offset] = 1
        V[0*3+0][col_offset] = 1
        W[(3*block)*2+0][col_offset] = 1
        W[(3*block+1)*2+0][col_offset] = 1
        col_offset += 1
        
        # m4 = a11(b10+b11)
        U[1*2+1][col_offset] = 1
        V[1*3+0][col_offset] = 1
        V[1*3+1][col_offset] = 1
        W[(3*block)*2+0][col_offset] = 1
        W[(3*block+1)*2+0][col_offset] = 1
        col_offset += 1
        
        # m5 = (a00+a11)(b00+b11)
        U[0*2+0][col_offset] = 1
        U[1*2+1][col_offset] = 1
        V[0*3+0][col_offset] = 1
        V[1*3+1][col_offset] = 1
        W[(3*block)*2+0][col_offset] = 1
        W[(3*block+1)*2+0][col_offset] = 1
        col_offset += 1
        
        # m6 = (a01-a11)(b10+b11)
        U[0*2+1][col_offset] = 1
        U[1*2+1][col_offset] = -1
        V[1*3+0][col_offset] = 1
        V[1*3+1][col_offset] = 1
        W[(3*block)*2+0][col_offset] = 1
        W[(3*block+1)*2+0][col_offset] = 1
        col_offset += 1
        
        # m7 = (a00+a10)(b00-b10)
        U[0*2+0][col_offset] = 1
        U[1*2+0][col_offset] = 1
        V[0*3+0][col_offset] = 1
        V[1*3+0][col_offset] = -1
        W[(3*block)*2+0][col_offset] = 1
        W[(3*block+1)*2+0][col_offset] = 1
        col_offset += 1
        
        # Factors 8-11: Schoolbook for 2x2x1 (local k=2)
        # m8 = a00 b02
        U[0*2+0][col_offset] = 1
        V[0*3+2][col_offset] = 1
        W[(3*block+2)*2+0][col_offset] = 1
        col_offset += 1
        
        # m9 = a01 b12
        U[0*2+1][col_offset] = 1
        V[1*3+2][col_offset] = 1
        W[(3*block+2)*2+0][col_offset] = 1
        col_offset += 1
        
        # m10 = a10 b02
        U[1*2+0][col_offset] = 1
        V[0*3+2][col_offset] = 1
        W[(3*block+2)*2+1][col_offset] = 1
        col_offset += 1
        
        # m11 = a11 b12
        U[1*2+1][col_offset] = 1
        V[1*3+2][col_offset] = 1
        W[(3*block+2)*2+1][col_offset] = 1
        col_offset += 1
        
        # Part 2: T_2,1,3 (j=2)
        # Rank 6
        # Factors 12-17: Schoolbook
        # m12 = a02 b20
        U[0*3+2][col_offset] = 1
        V[2*3+0][col_offset] = 1
        W[(3*block)*2+0][col_offset] = 1
        col_offset += 1
        
        # m13 = a02 b21
        U[0*3+2][col_offset] = 1
        V[2*3+1][col_offset] = 1
        W[(3*block+1)*2+0][col_offset] = 1
        col_offset += 1
        
        # m14 = a02 b22
        U[0*3+2][col_offset] = 1
        V[2*3+2][col_offset] = 1
        W[(3*block+2)*2+0][col_offset] = 1
        col_offset += 1
        
        # m15 = a12 b20
        U[1*3+2][col_offset] = 1
        V[2*3+0][col_offset] = 1
        W[(3*block)*2+1][col_offset] = 1
        col_offset += 1
        
        # m16 = a12 b21
        U[1*3+2][col_offset] = 1
        V[2*3+1][col_offset] = 1
        W[(3*block+1)*2+1][col_offset] = 1
        col_offset += 1
        
        # m17 = a12 b22
        U[1*3+2][col_offset] = 1
        V[2*3+2][col_offset] = 1
        W[(3*block+2)*2+1][col_offset] = 1
        col_offset += 1

    return {"U": U, "V": V, "W": W}
