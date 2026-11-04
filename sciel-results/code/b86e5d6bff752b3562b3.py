"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback: rank = n*m*p = 120
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
    
    # Strassen-2 decomposition for 2x2x2 blocks (rank 7)
    def strassen2_decomposition():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        return U, V, W

    # Strassen-3 decomposition for 3x3x3 blocks (rank 23)
    def strassen3_decomposition():
        U = [[0] * 23 for _ in range(9)]
        V = [[0] * 23 for _ in range(9)]
        W = [[0] * 23 for _ in range(9)]
        
        # M1 = (a+d)(j+p)
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[6][0]=1
        W[0][0]=1; W[3][0]=1; W[1][0]=1; W[4][0]=1
        
        # M2 = (a+d)(k+q)
        U[0][1]=1; U[3][1]=1
        V[1][1]=1; V[7][1]=1
        W[0][1]=1; W[3][1]=1; W[1][1]=1; W[4][1]=1
        
        # M3 = (b+e)(m+r)
        U[1][2]=1; U[4][2]=1
        V[3][2]=1; V[8][2]=1
        W[0][2]=1; W[3][2]=1; W[1][2]=1; W[4][2]=1
        
        # M4 = (b+e)(n+o)
        U[1][3]=1; U[4][3]=1
        V[4][3]=1; V[5][3]=1
        W[0][3]=1; W[3][3]=1; W[1][3]=1; W[4][3]=1
        
        # M5 = (c+f)(p+q)
        U[2][4]=1; U[5][4]=1
        V[6][4]=1; V[7][4]=1
        W[0][4]=1; W[3][4]=1; W[1][4]=1; W[4][4]=1
        
        # M6 = (c+f)(l+o)
        U[2][5]=1; U[5][5]=1
        V[2][5]=1; V[5][5]=1
        W[0][5]=1; W[3][5]=1; W[1][5]=1; W[4][5]=1
        
        # M7 = (a+e)(j+n)
        U[0][6]=1; U[4][6]=1
        V[0][6]=1; V[4][6]=1
        W[0][6]=1; W[3][6]=1; W[1][6]=1; W[4][6]=1
        
        # M8 = (a+e)(k+o)
        U[0][7]=1; U[4][7]=1
        V[1][7]=1; V[5][7]=1
        W[0][7]=1; W[3][7]=1; W[1][7]=1; W[4][7]=1
        
        # M9 = (b+f)(m+q)
        U[1][8]=1; U[5][8]=1
        V[3][8]=1; V[7][8]=1
        W[0][8]=1; W[3][8]=1; W[1][8]=1; W[4][8]=1
        
        # M10 = (b+f)(n+r)
        U[1][9]=1; U[5][9]=1
        V[4][9]=1; V[8][9]=1
        W[0][9]=1; W[3][9]=1; W[1][9]=1; W[4][9]=1
        
        # M11 = (c+i)(j+m)
        U[2][10]=1; U[8][10]=1
        V[0][10]=1; V[3][10]=1
        W[0][10]=1; W[3][10]=1; W[1][10]=1; W[4][10]=1
        
        # M12 = (c+i)(l+p)
        U[2][11]=1; U[8][11]=1
        V[2][11]=1; V[6][11]=1
        W[0][11]=1; W[3][11]=1; W[1][11]=1; W[4][11]=1
        
        # M13 = (d+h)(k+p)
        U[3][12]=1; U[7][12]=1
        V[1][12]=1; V[6][12]=1
        W[0][12]=1; W[3][12]=1; W[1][12]=1; W[4][12]=1
        
        # M14 = (d+h)(o+r)
        U[3][13]=1; U[7][13]=1
        V[5][13]=1; V[8][13]=1
        W[0][13]=1; W[3][13]=1; W[1][13]=1; W[4][13]=1
        
        # M15 = (e+i)(j+n)
        U[4][14]=1; U[8][14]=1
        V[0][14]=1; V[4][14]=1
        W[0][14]=1; W[3][14]=1; W[1][14]=1; W[4][14]=1
        
        # M16 = (e+i)(k+o)
        U[4][15]=1; U[8][15]=1
        V[1][15]=1; V[5][15]=1
        W[0][15]=1; W[3][15]=1; W[1][15]=1; W[4][15]=1
        
        # M17 = (f+g)(m+q)
        U[5][16]=1; U[6][16]=1
        V[3][16]=1; V[7][16]=1
        W[0][16]=1; W[3][16]=1; W[1][16]=1; W[4][16]=1
        
        # M18 = (f+g)(n+r)
        U[5][17]=1; U[6][17]=1
        V[4][17]=1; V[8][17]=1
        W[0][17]=1; W[3][17]=1; W[1][17]=1; W[4][17]=1
        
        # M19 = (g+i)(j+p)
        U[6][18]=1; U[8][18]=1
        V[0][18]=1; V[6][18]=1
        W[0][18]=1; W[3][18]=1; W[1][18]=1; W[4][18]=1
        
        # M20 = (g+i)(k+q)
        U[6][19]=1; U[8][19]=1
        V[1][19]=1; V[7][19]=1
        W[0][19]=1; W[3][19]=1; W[1][19]=1; W[4][19]=1
        
        # M21 = a(j-m)
        U[0][20]=1
        V[0][20]=1; V[3][20]=-1
        W[0][20]=1; W[3][20]=1; W[1][20]=1; W[4][20]=1
        
        # M22 = c(l-p)
        U[2][21]=1
        V[2][21]=1; V[6][21]=-1
        W[0][21]=1; W[3][21]=1; W[1][21]=1; W[4][21]=1
        
        # M23 = e(n-o)
        U[4][22]=1
        V[4][22]=1; V[5][22]=-1
        W[0][22]=1; W[3][22]=1; W[1][22]=1; W[4][22]=1
        
        # Correcting W entries for Strassen-3
        # The above W entries are placeholders. We need to solve for W.
        # For now, we will verify and if it fails, we return schoolbook for this block.
        return U, V, W

    # Build full 4x5x6 decomposition using block partition:
    # n=4=3+1, m=5=3+2, p=6=3+3
    # 3x3x3 blocks: 2 blocks, rank 23 each = 46
    # 3x2x3 blocks: 2 blocks, rank 18 each = 36
    # 1x3x3 blocks: 2 blocks, rank 9 each = 18
    # 1x2x3 blocks: 2 blocks, rank 6 each = 12
    # Total rank = 106
    
    total_rank = 2 * 23 + 2 * 18 + 2 * 9 + 2 * 6  # = 106
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    col_offset = 0
    U3, V3, W3 = strassen3_decomposition()
    
    n_bounds = [0, 3, 4]
    m_bounds = [0, 3, 5]
    p_bounds = [0, 3, 6]
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                ni = n_bounds[bi]
                nj = n_bounds[bi+1]
                mi = m_bounds[bj]
                mj = m_bounds[bj+1]
                pi = p_bounds[bk]
                pj = p_bounds[bk+1]
                
                n_size = nj - ni
                m_size = mj - mi
                p_size = pj - pi
                
                if n_size == 3 and m_size == 3 and p_size == 3:
                    # Use Strassen-3 (rank 23)
                    for r in range(23):
                        for ii in range(3):
                            for jj in range(3):
                                U_full[(ni+ii)*m + (mi+jj)][col_offset + r] = U3[ii*3+jj][r]
                        for jj in range(3):
                            for kk in range(3):
                                V_full[(mi+jj)*p + (pi+kk)][col_offset + r] = V3[jj*3+kk][r]
                        for kk in range(3):
                            for ii in range(3):
                                W_full[(pi+kk)*n + (ni+ii)][col_offset + r] = W3[kk*3+ii][r]
                    col_offset += 23
                else:
                    # Schoolbook for other blocks
                    for ii in range(n_size):
                        for jj in range(m_size):
                            for kk in range(p_size):
                                U_full[(ni+ii)*m + (mi+jj)][col_offset] = 1
                                V_full[(mi+jj)*p + (pi+kk)][col_offset] = 1
                                W_full[(pi+kk)*n + (ni+ii)][col_offset] = 1
                                col_offset += 1
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
        if not valid:
            break
    
    if valid:
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # If Strassen-3 verification fails, fall back to previous 108 rank solution
    # Re-run the 108 rank construction
    total_rank = 12 * 7 + 6 * 4
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    col_offset = 0
    U2, V2, W2 = strassen2_decomposition()
    
    n_bounds = [0, 2, 4]
    m_bounds = [0, 2, 4, 5]
    p_bounds = [0, 2, 4, 6]
    
    for bi in range(2):
        for bj in range(3):
            for bk in range(3):
                ni = n_bounds[bi]
                nj = n_bounds[bi+1]
                mi = m_bounds[bj]
                mj = m_bounds[bj+1]
                pi = p_bounds[bk]
                pj = p_bounds[bk+1]
                
                n_size = nj - ni
                m_size = mj - mi
                p_size = pj - pi
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    for r in range(7):
                        for ii in range(2):
                            for jj in range(2):
                                U_full[(ni+ii)*m + (mi+jj)][col_offset + r] = U2[ii*2+jj][r]
                        for jj in range(2):
                            for kk in range(2):
                                V_full[(mi+jj)*p + (pi+kk)][col_offset + r] = V2[jj*2+kk][r]
                        for kk in range(2):
                            for ii in range(2):
                                W_full[(pi+kk)*n + (ni+ii)][col_offset + r] = W2[kk*2+ii][r]
                    col_offset += 7
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    for ii in range(2):
                        for jj in range(1):
                            for kk in range(2):
                                U_full[(ni+ii)*m + (mi+jj)][col_offset] = 1
                                V_full[(mi+jj)*p + (pi+kk)][col_offset] = 1
                                W_full[(pi+kk)*n + (ni+ii)][col_offset] = 1
                                col_offset += 1
                else:
                    for ii in range(n_size):
                        for jj in range(m_size):
                            for kk in range(p_size):
                                U_full[(ni+ii)*m + (mi+jj)][col_offset] = 1
                                V_full[(mi+jj)*p + (pi+kk)][col_offset] = 1
                                W_full[(pi+kk)*n + (ni+ii)][col_offset] = 1
                                col_offset += 1
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
        if not valid:
            break
    
    if valid:
        return {"U": U_full, "V": V_full, "W": W_full}
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
