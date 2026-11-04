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
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        Convention: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # A[0,0]=a->U[0], A[0,1]=b->U[1], A[1,0]=c->U[2], A[1,1]=d->U[3]
        # B[0,0]=e->V[0], B[0,1]=f->V[1], B[1,0]=g->V[2], B[1,1]=h->V[3]
        # C[0,0]->W[0], C[0,1]->W[1], C[1,0]->W[2], C[1,1]->W[3]
        
        # M1 = (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W
    
    # Build full 4x5x6 decomposition using block partition:
    # n=4=2+2, m=5=2+2+1, p=6=2+2+2
    # 2x2x2 blocks: 2*2*3 = 12 blocks, rank 7 each = 84
    # 2x1x2 blocks: 2*1*3 = 6 blocks, rank 1 each = 6
    # Total rank = 90
    
    total_rank = 12 * 7 + 6 * 1  # = 90
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    col_offset = 0
    U2, V2, W2 = strassen2_decomposition()
    
    # Block offsets
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
                    # Use Strassen-2 (rank 7)
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
                    # Rank-1 decomposition for 2x1x2
                    for ii in range(2):
                        U_full[(ni+ii)*m + mi][col_offset] = 1
                    for kk in range(2):
                        V_full[mi*p + (pi+kk)][col_offset] = 1
                        W_full[(pi+kk)*n + ni][col_offset] = 1
                        W_full[(pi+kk)*n + (ni+1)][col_offset] = 1
                    col_offset += 1
                else:
                    # Fallback
                    if n_size == 1 or m_size == 1 or p_size == 1:
                        for ii in range(n_size):
                            for jj in range(m_size):
                                U_full[(ni+ii)*m + (mi+jj)][col_offset] = 1
                        for jj in range(m_size):
                            for kk in range(p_size):
                                V_full[(mi+jj)*p + (pi+kk)][col_offset] = 1
                        for kk in range(p_size):
                            for ii in range(n_size):
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
    
    # Verify the decomposition using numpy for speed
    U_np = np.array(U_full, dtype=float)
    V_np = np.array(V_full, dtype=float)
    W_np = np.array(W_full, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i * m + j]
                v_row = V_np[j * p + k]
                w_row = W_np[k * n + i]
                s = np.dot(u_row, v_row * w_row)
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
