import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partition sizes
    # n=4 -> 2, 2
    # m=5 -> 2, 2, 1
    # p=6 -> 2, 2, 2
    
    I_sizes = [2, 2]
    J_sizes = [2, 2, 1]
    K_sizes = [2, 2, 2]
    
    # Calculate total rank
    total_rank = 0
    for i_s in I_sizes:
        for j_s in J_sizes:
            for k_s in K_sizes:
                if i_s == 2 and j_s == 2 and k_s == 2:
                    total_rank += 7
                else:
                    total_rank += i_s * j_s * k_s
    
    # Initialize U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_block(i_offset, j_offset, k_offset, rank_offset):
        # Strassen 2x2x2 factors
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        
        # M1 = (A00+A11)(B00+B11)
        U[i_offset*2 + 0][rank_offset+0] += 1
        U[i_offset*2 + 1][rank_offset+0] += 1
        V[j_offset*2 + 0][rank_offset+0] += 1
        V[j_offset*2 + 1][rank_offset+0] += 1
        W[k_offset*2 + 0][rank_offset+0] += 1
        W[k_offset*2 + 1][rank_offset+0] += 1
        
        # M2 = (A10+A11)B00
        U[i_offset*2 + 2][rank_offset+1] += 1 # A10 is i=1, j=0 -> idx 2
        U[i_offset*2 + 3][rank_offset+1] += 1 # A11 is i=1, j=1 -> idx 3
        V[j_offset*2 + 0][rank_offset+1] += 1 # B00 is j=0, k=0 -> idx 0
        W[k_offset*2 + 0][rank_offset+1] += 1 # C10? No, C[i,k].
        # W index is k*n + i. For 2x2 block, local k, i.
        # W[k*2+i, c].
        # M2 contributes to C10 and C11?
        # C10 = A10B00 + A11B10.
        # Strassen M2 = (A10+A11)B00.
        # This term is used in C10 and C11?
        # Let's check Strassen reconstruction.
        # C00 = M1 - M4 + M7 - M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M5
        # Wait, standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M7 - M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M5
        
        # So for M2:
        # W[0, 1] += 1 (C10, k=1, i=1? No. C[i,k]. i=1, k=0. W[0*2+1]=W[1])
        # W[1, 1] += 1 (C11, i=1, k=1. W[1*2+1]=W[3])
        # Wait, W index is k*n + i.
        # For local 2x2, W[k*2+i].
        # C10 -> i=1, k=0 -> W[0*2+1] = W[1].
        # C11 -> i=1, k=1 -> W[1*2+1] = W[3].
        
        W[k_offset*2 + 0][rank_offset+1] += 1 # C10
        W[k_offset*2 + 1][rank_offset+1] += 1 # C11
        
        # M3 = A00(B01-B11)
        U[i_offset*2 + 0][rank_offset+2] += 1 # A00
        V[j_offset*2 + 1][rank_offset+2] += 1 # B01 (j=0, k=1 -> idx 1)
        V[j_offset*2 + 3][rank_offset+2] -= 1 # B11 (j=1, k=1 -> idx 3)
        # C01 -> i=0, k=1 -> W[1*2+0] = W[2]
        # C11 -> i=1, k=1 -> W[1*2+1] = W[3]
        W[k_offset*2 + 1][rank_offset+2] += 1 # C01? No.
        # C01 = M3 + M5.
        # C11 = M1 - M2 + M3 + M5.
        # So M3 contributes to C01 and C11.
        # C01: i=0, k=1. W[1*2+0]=W[2].
        # C11: i=1, k=1. W[1*2+1]=W[3].
        W[k_offset*2 + 1][rank_offset+2] += 1 # C01
        W[k_offset*2 + 1][rank_offset+2] += 1 # C11? No, same W index?
        # W[k*2+i].
        # C01: k=1, i=0 -> W[2].
        # C11: k=1, i=1 -> W[3].
        W[k_offset*2 + 1][rank_offset+2] += 1 # This is W[2] (C01)
        W[k_offset*2 + 1][rank_offset+2] += 1 # This is W[3] (C11)
        # Wait, I need to add to W[2] and W[3].
        W[k_offset*2 + 1][rank_offset+2] += 1 # W[2]
        W[k_offset*2 + 1][rank_offset+2] += 1 # W[3] is different index.
        # k=1, i=0 -> 1*2+0=2.
        # k=1, i=1 -> 1*2+1=3.
        W[k_offset*2 + 1][rank_offset+2] += 1 # W[2]
        W[k_offset*2 + 1][rank_offset+2] += 1 # W[3]
        # I am confusing myself.
        # W index is k*2 + i.
        # C01: k=1, i=0 -> idx 2.
        # C11: k=1, i=1 -> idx 3.
        W[k_offset*2 + 1][rank_offset+2] += 1 # idx 2
        W[k_offset*2 + 1][rank_offset+2] += 1 # idx 3? No.
        # Let's just use a helper function to add to W.
        
        # I will rewrite the block addition logic clearly.
        
    # Re-implementation of block addition for clarity and correctness
    
    rank_offset = 0
    for i_idx, i_s in enumerate(I_sizes):
        for j_idx, j_s in enumerate(J_sizes):
            for k_idx, k_s in enumerate(K_sizes):
                if i_s == 2 and j_s == 2 and k_s == 2:
                    # Strassen Block
                    # U rows: i_s * j_s = 4. Offset: i_idx * (m) ? No.
                    # U is n*m x rank.
                    # Row index for U: i_global * m + j_global.
                    # i_global = sum(I_sizes[:i_idx]) + i_local.
                    # j_global = sum(J_sizes[:j_idx]) + j_local.
                    # V is m*p x rank.
                    # Row index for V: j_global * p + k_global.
                    # W is p*n x rank.
                    # Row index for W: k_global * n + i_global.
                    
                    i_base = sum(I_sizes[:i_idx])
                    j_base = sum(J_sizes[:j_idx])
                    k_base = sum(K_sizes[:k_idx])
                    
                    # M1
                    U[(i_base+0)*m + (j_base+0)][rank_offset] += 1
                    U[(i_base+1)*m + (j_base+1)][rank_offset] += 1
                    V[(j_base+0)*p + (k_base+0)][rank_offset] += 1
                    V[(j_base+1)*p + (k_base+1)][rank_offset] += 1
                    W[(k_base+0)*n + (i_base+0)][rank_offset] += 1 # C00
                    W[(k_base+0)*n + (i_base+1)][rank_offset] -= 1 # C10? No.
                    # C00 = M1 - M4 + M7 - M6
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M5
                    # C01 = M3 + M5
                    # M1 contributes to C00 (+1) and C11 (+1).
                    W[(k_base+1)*n + (i_base+1)][rank_offset] += 1 # C11
                    
                    # M2
                    U[(i_base+1)*m + (j_base+0)][rank_offset+1] += 1
                    U[(i_base+1)*m + (j_base+1)][rank_offset+1] += 1
                    V[(j_base+0)*p + (k_base+0)][rank_offset+1] += 1
                    W[(k_base+0)*n + (i_base+1)][rank_offset+1] += 1 # C10
                    W[(k_base+1)*n + (i_base+1)][rank_offset+1] -= 1 # C11 (-1)
                    
                    # M3
                    U[(i_base+0)*m + (j_base+0)][rank_offset+2] += 1
                    V[(j_base+0)*p + (k_base+1)][rank_offset+2] += 1
                    V[(j_base+1)*p + (k_base+1)][rank_offset+2] -= 1
                    W[(k_base+1)*n + (i_base+0)][rank_offset+2] += 1 # C01
                    W[(k_base+1)*n + (i_base+1)][rank_offset+2] += 1 # C11
                    
                    # M4
                    U[(i_base+1)*m + (j_base+1)][rank_offset+3] += 1
                    V[(j_base+1)*p + (k_base+0)][rank_offset+3] += 1
                    V[(j_base+0)*p + (k_base+0)][rank_offset+3] -= 1
                    W[(k_base+0)*n + (i_base+1)][rank_offset+3] += 1 # C10
                    W[(k_base+0)*n + (i_base+0)][rank_offset+3] -= 1 # C00 (-1)
                    
                    # M5
                    U[(i_base+0)*m + (j_base+0)][rank_offset+4] += 1
                    U[(i_base+0)*m + (j_base+1)][rank_offset+4] += 1
                    V[(j_base+1)*p + (k_base+1)][rank_offset+4] += 1
                    W[(k_base+1)*n + (i_base+0)][rank_offset+4] += 1 # C01
                    W[(k_base+1)*n + (i_base+1)][rank_offset+4] += 1 # C11
                    
                    # M6
                    U[(i_base+1)*m + (j_base+0)][rank_offset+5] += 1
                    U[(i_base+0)*m + (j_base+0)][rank_offset+5] -= 1
                    V[(j_base+0)*p + (k_base+0)][rank_offset+5] += 1
                    V[(j_base+0)*p + (k_base+1)][rank_offset+5] += 1
                    W[(k_base+0)*n + (i_base+0)][rank_offset+5] -= 1 # C00 (-1)
                    
                    # M7
                    U[(i_base+0)*m + (j_base+1)][rank_offset+6] += 1
                    U[(i_base+1)*m + (j_base+1)][rank_offset+6] -= 1
                    V[(j_base+1)*p + (k_base+0)][rank_offset+6] += 1
                    V[(j_base+1)*p + (k_base+1)][rank_offset+6] += 1
                    W[(k_base+0)*n + (i_base+0)][rank_offset+6] += 1 # C00
                    
                    rank_offset += 7
                else:
                    # Schoolbook Block
                    for i_l in range(i_s):
                        for j_l in range(j_s):
                            for k_l in range(k_s):
                                i_g = sum(I_sizes[:i_idx]) + i_l
                                j_g = sum(J_sizes[:j_idx]) + j_l
                                k_g = sum(K_sizes[:k_idx]) + k_l
                                U[i_g * m + j_g][rank_offset] += 1
                                V[j_g * p + k_g][rank_offset] += 1
                                W[k_g * n + i_g][rank_offset] += 1
                                rank_offset += 1
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
