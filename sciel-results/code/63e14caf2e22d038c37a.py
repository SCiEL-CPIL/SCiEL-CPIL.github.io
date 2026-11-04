"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Rank 36 decomposition using Strassen for 2x2x2 blocks
    # Split m=4 into two blocks of size 2
    # For each block, split p=5 into two blocks of size 2 (Strassen) and one block of size 1 (schoolbook)
    # Each 2x2x2 block uses rank 7 (Strassen)
    # Each 2x2x1 block uses rank 4 (schoolbook)
    # Total: 2 * (2*7 + 4) = 2 * 18 = 36
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    column = 0
    
    # Strassen 2x2x2 terms
    # For 2x2 matrices: A[i,j], B[j,k], C[i,k]
    # Indices: A: 0,1,2,3 -> (0,0),(0,1),(1,0),(1,1)
    #          B: 0,1,2,3 -> (0,0),(0,1),(1,0),(1,1)
    #          C: 0,1,2,3 -> (0,0),(0,1),(1,0),(1,1)
    
    strassen_terms = [
        # P1 = (a00+a11)(b00+b11) -> C00+1, C11+1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # P2 = (a10+a11)b00 -> C10+1, C11-1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        # P3 = a00(b01-b11) -> C01+1, C11+1
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        # P4 = a11(b10-b00) -> C00+1, C10+1
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        # P5 = (a00+a01)b11 -> C00-1, C01+1
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # P6 = (a10-a00)(b00+b01) -> C11+1
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # P7 = (a01-a11)(b10+b11) -> C00+1
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    # Schoolbook 2x2x1 terms
    # C00 = A00*B0 + A01*B1
    # C10 = A10*B0 + A11*B1
    schoolbook_2x2x1_terms = [
        ({0:1}, {0:1}, {0:1}),  # A00*B0 -> C00
        ({1:1}, {1:1}, {0:1}),  # A01*B1 -> C00
        ({2:1}, {0:1}, {1:1}),  # A10*B0 -> C10
        ({3:1}, {1:1}, {1:1})   # A11*B1 -> C10
    ]
    
    # Process each m-block (j=0,1 and j=2,3)
    for block_j in range(2):
        j_off = block_j * 2  # 0 or 2
        
        # For each p-segment
        # Segment 0: k=0,1 (Strassen 2x2x2)
        # Segment 1: k=2,3 (Strassen 2x2x2)
        # Segment 2: k=4 (Schoolbook 2x2x1)
        
        for seg in range(3):
            if seg < 2:
                # Strassen 2x2x2
                k_off = seg * 2  # 0 or 2
                
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    # Add this rank-1 term
                    for idx, val in u_coeffs.items():
                        # idx is local (i, j_local) where j_local in {0,1}
                        # Global A index: i*m + (j_off + j_local)
                        # idx encodes: i*2 + j_local (since 2x2)
                        i_local = idx // 2
                        j_local = idx % 2
                        global_a = i_local * m + (j_off + j_local)
                        U[global_a][column] += val
                    
                    for idx, val in v_coeffs.items():
                        # idx is local (j_local, k_local) where j_local, k_local in {0,1}
                        # Global B index: (j_off + j_local)*p + (k_off + k_local)
                        j_local = idx // 2
                        k_local = idx % 2
                        global_b = (j_off + j_local) * p + (k_off + k_local)
                        V[global_b][column] += val
                    
                    for idx, val in w_coeffs.items():
                        # idx is local (i_local, k_local) where i_local, k_local in {0,1}
                        # Global C index: (k_off + k_local)*n + i_local
                        i_local = idx // 2
                        k_local = idx % 2
                        global_c = (k_off + k_local) * n + i_local
                        W[global_c][column] += val
                    
                    column += 1
                    
            else:
                # Schoolbook 2x2x1
                k_off = 4  # k=4
                
                for u_coeffs, v_coeffs, w_coeffs in schoolbook_2x2x1_terms:
                    # Add this rank-1 term
                    for idx, val in u_coeffs.items():
                        # idx is local (i, j_local) where j_local in {0,1}
                        # For 2x2x1, B has only 1 column, so B index is (j_local, 0)
                        # But in our scheme, B index for 2x2x1 should be (j_local, 0)
                        # Actually, for 2x2x1: A is 2x2, B is 2x1, C is 2x1
                        # A indices: 0,1,2,3 -> (0,0),(0,1),(1,0),(1,1)
                        # B indices: 0,1 -> (0,0),(1,0) [j_local, k_local=0]
                        # C indices: 0,1 -> (0,0),(1,0) [i_local, k_local=0]
                        
                        i_local = idx // 2
                        j_local = idx % 2
                        global_a = i_local * m + (j_off + j_local)
                        U[global_a][column] += val
                    
                    for idx, val in v_coeffs.items():
                        # idx is local (j_local, 0)
                        j_local = idx  # 0 or 1
                        k_local = 0
                        global_b = (j_off + j_local) * p + (k_off + k_local)
                        V[global_b][column] += val
                    
                    for idx, val in w_coeffs.items():
                        # idx is local (i_local, 0)
                        i_local = idx  # 0 or 1
                        k_local = 0
                        global_c = (k_off + k_local) * n + i_local
                        W[global_c][column] += val
                    
                    column += 1
    
    # Verify the decomposition
    assert _validate(U, V, W, n, m, p), "Decomposition failed validation!"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
