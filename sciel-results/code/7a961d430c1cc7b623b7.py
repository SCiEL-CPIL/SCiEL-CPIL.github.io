"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We'll build a rank-36 decomposition using Strassen on 2x2x2 blocks
    # Structure: split j into {0,1} and {2,3}, split k into {0,1},{2,3},{4}
    # For each (j-block, k-block), use Strassen (rank 7) for 2x2x2 or rank 2 for 2x2x1
    
    # Strassen decomposition for 2x2 x 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2 x 2x1: A (2x2) * B (2x1) = C (2x1)
    # C0 = a00*b0 + a01*b1
    # C1 = a10*b0 + a11*b1
    # Rank 2: two products a00*b0, a01*b1, a10*b0, a11*b1 -> actually rank 4 schoolbook
    # Wait, 2x2 x 2x1: T[i,j,k] where i in {0,1}, j in {0,1}, k in {0} (only one column)
    # C[i,0] = a[i,0]*b[0,0] + a[i,1]*b[1,0]
    # This is rank 2: use products a[i,0]*b[0,0] and a[i,1]*b[1,0] for i=0,1
    # So rank 2 for each i, total rank 4? No...
    
    # Actually for 2x2 x 2x1, the tensor is T[i,j,0] = 1 for i,j in {0,1}
    # We need sum_r U[i*2+j, r] * V[j*1+0, r] * W[0*2+i, r] = 1
    # Schoolbook: rank 4 (one for each (i,j))
    # Can we do rank 2? T[i,j,0] = 1 = sum_r U[i,j,r]*V[j,0,r]*W[0,i,r]
    # If we set W[0,0,r] and W[0,1,r] appropriately...
    # Let's say r=0: U[0,0,0]=1, V[0,0,0]=1, W[0,0,0]=1 -> contributes to T[0,0,0]
    # r=1: U[0,1,1]=1, V[1,0,1]=1, W[0,0,1]=1 -> contributes to T[0,1,0]
    # r=2: U[1,0,2]=1, V[0,0,2]=1, W[0,1,2]=1 -> contributes to T[1,0,0]
    # r=3: U[1,1,3]=1, V[1,0,3]=1, W[0,1,3]=1 -> contributes to T[1,1,0]
    # That's rank 4. Can we do better?
    # T[i,j,0] = 1 for all i,j. This is a 2x2x1 tensor. Its rank is 2 (since it's a matrix of all 1s, rank 1 as a matrix, but tensor rank...).
    # Actually, the 2x2x1 tensor with all entries 1: T = [1,1;1,1] as a matrix (collapsing i,j). 
    # Tensor rank: we need sum_r u_r[i,j] * v_r[j] * w_r[i] = 1
    # This is equivalent to expressing the all-ones 2x2 matrix as sum_r (u_r[i]*w_r[i]) * v_r[j]... no.
    # Let me just use rank 4 for 2x2x1 to be safe.
    
    # Hmm, this is getting complicated. Let me just implement the known rank-36 solution.
    
    # Actually, let me try a cleaner approach: use the fact that M_{2,4,5} can be decomposed
    # by treating it as two copies of M_{2,2,5} with shared W.
    
    # For M_{2,2,5}, use rank 18 (9+9: two Strassen 2x2x2 blocks + one 2x2x1)
    # Wait, 7+7+4 = 18 for 2x2x5 (two 2x2x2 Strassen + one 2x2x1 schoolbook rank 4)
    # Then M_{2,4,5} with shared W: rank <= 18+18 = 36
    
    # Let me construct this explicitly.
    
    # Helper: Strassen 2x2x2 decomposition
    # Returns (U_block, V_block, W_block) for a 2x2 x 2x2 subproblem
    # U_block: 4x7 (indices i*2+j for i,j in {0,1})
    # V_block: 4x7 (indices j*2+k for j,k in {0,1})
    # W_block: 4x7 (indices k*2+i for k,i in {0,1})
    
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1 = (a00+a11)(b00+b11): U: a00,a11; V: b00,b11; W: c00,c11
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # c00, c11
        # M2 = (a10+a11)b00: U: a10,a11; V: b00; W: c10
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # c10 (k=1,i=0 -> index 1*2+0=2)
        # M3 = a00(b01-b11): U: a00; V: b01,-b11; W: c01
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1  # c01 (k=1,i=0 -> index 1)
        # M4 = a11(b10-b00): U: a11; V: b10,-b00; W: c00,c10
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1  # c00, c10
        # M5 = (a00+a01)b11: U: a00,a01; V: b11; W: -c00,c01
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1  # -c00, c01
        # M6 = (a10-a00)(b00+b10): U: a10,-a00; V: b00,b10; W: c11
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1  # c11 (k=1,i=1 -> index 1*2+1=3)
        # M7 = (a01-a11)(b01+b11): U: a01,-a11; V: b01,b11; W: c00,c11
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1  # c00, c11
        return U, V, W
    
    # Helper: schoolbook 2x2x1 decomposition
    def schoolbook_2x2x1():
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                col = i*2+j
                U[i*2+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1  # k=0, so index k*2+i = i
        return U, V, W
    
    # Now build the full decomposition
    # j-blocks: 0 (j=0,1), 1 (j=2,3)
    # k-blocks: 0 (k=0,1), 1 (k=2,3), 2 (k=4)
    
    # For each j-block and k-block, we have a subproblem
    # j-block 0: A indices i*4+j for i in {0,1}, j in {0,1} -> U rows 0,1,2,3
    #             B indices j*5+k for j in {0,1}, k in {0,1,2,3,4} -> V rows 0-9
    # j-block 1: A indices i*4+j for i in {0,1}, j in {2,3} -> U rows 4,5,6,7
    #             B indices j*5+k for j in {2,3}, k in {0,1,2,3,4} -> V rows 10-19
    # W indices: k*2+i for k in {0,1,2,3,4}, i in {0,1} -> W rows 0-9
    
    # Subproblem (jb, kb): 
    #   jb=0, kb=0: 2x2x2, Strassen rank 7
    #   jb=0, kb=1: 2x2x2, Strassen rank 7
    #   jb=0, kb=2: 2x2x1, schoolbook rank 4
    #   jb=1, kb=0: 2x2x2, Strassen rank 7
    #   jb=1, kb=1: 2x2x2, Strassen rank 7
    #   jb=1, kb=2: 2x2x1, schoolbook rank 4
    # Total rank: 7+7+4+7+7+4 = 36
    
    total_rank = 36
    U = [[0]*total_rank for _ in range(8)]   # n*m = 2*4 = 8
    V = [[0]*total_rank for _ in range(20)]  # m*p = 4*5 = 20
    W = [[0]*total_rank for _ in range(10)]  # p*n = 5*2 = 10
    
    col = 0
    for jb in range(2):  # j-block
        for kb in range(3):  # k-block
            if kb < 2:
                # 2x2x2 Strassen
                sU, sV, sW = strassen_2x2x2()
                for i in range(2):
                    for j in range(2):
                        U[i*4 + jb*2 + j][col] = sU[i*2+j][0]  # Wait, sU has 7 cols
                # Need to handle each column of the Strassen decomposition
                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            U[i*4 + jb*2 + j][col+r] = sU[i*2+j][r]
                    for j in range(2):
                        for k in range(2):
                            V[(jb*2+j)*5 + kb*2 + k][col+r] = sV[j*2+k][r]
                    for k in range(2):
                        for i in range(2):
                            W[(kb*2+k)*2 + i][col+r] = sW[k*2+i][r]
                col += 7
            else:
                # 2x2x1 schoolbook
                sU, sV, sW = schoolbook_2x2x1()
                for r in range(4):
                    for i in range(2):
                        for j in range(2):
                            U[i*4 + jb*2 + j][col+r] = sU[i*2+j][r]
                    for j in range(2):
                        V[(jb*2+j)*5 + 4][col+r] = sV[j][r]  # k=4
                    for i in range(2):
                        W[4*2 + i][col+r] = sW[i][r]  # k=4, index 4*2+i
                col += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
