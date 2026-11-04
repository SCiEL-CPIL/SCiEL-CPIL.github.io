"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def strassen2_decomp():
        """2x2x2 Strassen, rank 7"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2=(A21+A22)B21: C21+=M2
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        # M3=A11(B12-B22): C12+=M3
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        # M4=A22(B21-B11): C11+=M4, C21+=M4
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5=(A11+A12)B22: C11-=M5, C12+=M5
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6=(A21-A11)(B11+B12): C22+=M6
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7=(A12-A22)(B21+B22): C11+=M7, C22+=M7
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W

    def schoolbook_decomp(nn, mm, pp):
        """Schoolbook decomposition for nn x mm x pp"""
        rank = nn * mm * pp
        U = [[0]*rank for _ in range(nn*mm)]
        V = [[0]*rank for _ in range(mm*pp)]
        W = [[0]*rank for _ in range(pp*nn)]
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i*mm+j][col] = 1
                    V[j*pp+k][col] = 1
                    W[k*nn+i][col] = 1
                    col += 1
        return U, V, W

    # Partition:
    # n = 2 + 2
    # m = 2 + 2 + 1
    # p = 2 + 2 + 2
    
    n_splits = [2, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Calculate total rank
    total_rank = 0
    for ni in n_splits:
        for mj in m_splits:
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
                    
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    n_offset = 0
    for ni in n_splits:
        m_offset = 0
        for mj in m_splits:
            p_offset = 0
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    U_block, V_block, W_block = strassen2_decomp()
                else:
                    U_block, V_block, W_block = schoolbook_decomp(ni, mj, pk)
                
                # Copy block into full matrices
                for i in range(ni):
                    for j in range(mj):
                        for r in range(len(U_block[0])):
                            U_full[n_offset + i*mj + j][col + r] = U_block[i*mj+j][r]
                
                for j in range(mj):
                    for k in range(pk):
                        for r in range(len(V_block[0])):
                            V_full[m_offset + j*pk + k][col + r] = V_block[j*pk+k][r]
                
                for k in range(pk):
                    for i in range(ni):
                        for r in range(len(W_block[0])):
                            W_full[p_offset + k*ni + i][col + r] = W_block[k*ni+i][r]
                
                col += len(U_block[0])
                p_offset += pk
            m_offset += mj
        n_offset += ni
        
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U_full[i*m+j][r] * V_full[j*p+k][r] * W_full[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        
    if valid:
        return {"U": U_full, "V": V_full, "W": W_full}
        
    return best_result
# EVOLVE-BLOCK-END
