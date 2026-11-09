import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    n, m, p = 3, 4, 6
    
    def make_strassen_2x2():
        """2x2x2 Strassen: U[i*2+j], V[j*2+k], W[k*2+i], rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W
    
    def make_schoolbook_block(ar, ac, bc):
        """Schoolbook for ar x ac x bc block."""
        rank = ar * ac * bc
        U = np.zeros((ar * ac, rank))
        V = np.zeros((ac * bc, rank))
        W = np.zeros((bc * ar, rank))
        col = 0
        for ii in range(ar):
            for jj in range(ac):
                for kk in range(bc):
                    U[ii*ac+jj, col] = 1.0
                    V[jj*bc+kk, col] = 1.0
                    W[kk*ar+ii, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    sU, sV, sW = make_strassen_2x2()
    
    def build_blocked(a_row_parts, a_col_parts, b_col_parts):
        """Build blocked decomposition with Strassen for 2x2x2 blocks."""
        total_rank = 0
        for (ar_s, ar_e) in a_row_parts:
            ar = ar_e - ar_s
            for (ac_s, ac_e) in a_col_parts:
                ac = ac_e - ac_s
                for (bc_s, bc_e) in b_col_parts:
                    bc = bc_e - bc_s
                    if ar == 2 and ac == 2 and bc == 2:
                        total_rank += 7
                    else:
                        total_rank += ar * ac * bc
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        for (ar_s, ar_e) in a_row_parts:
            ar = ar_e - ar_s
            for (ac_s, ac_e) in a_col_parts:
                ac = ac_e - ac_s
                for (bc_s, bc_e) in b_col_parts:
                    bc = bc_e - bc_s
                    if ar == 2 and ac == 2 and bc == 2:
                        for r in range(7):
                            for ii in range(2):
                                for jj in range(2):
                                    U[(ar_s+ii)*m + (ac_s+jj), col] = sU[ii*2+jj, r]
                            for jj in range(2):
                                for kk in range(2):
                                    V[(ac_s+jj)*p + (bc_s+kk), col] = sV[jj*2+kk, r]
                            for kk in range(2):
                                for ii in range(2):
                                    W[(bc_s+kk)*n + (ar_s+ii), col] = sW[kk*2+ii, r]
                            col += 1
                    else:
                        for ii in range(ar):
                            for jj in range(ac):
                                for kk in range(bc):
                                    U[(ar_s+ii)*m + (ac_s+jj), col] = 1.0
                                    V[(ac_s+jj)*p + (bc_s+kk), col] = 1.0
                                    W[(bc_s+kk)*n + (ar_s+ii), col] = 1.0
                                    col += 1
        
        assert col == total_rank, f"col {col} != {total_rank}"
        return U, V, W, total_rank
    
    def verify(U, V, W, rank, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Best known: ar=[2,1], ac=[2,2], bc=[2,2,2] -> rank 66
    best_U, best_V, best_W, best_rank = None, None, None, 72
    
    # Try the 66-rank decomposition first (fast, deterministic)
    result = build_blocked([(0,2),(2,3)], [(0,2),(2,4)], [(0,2),(2,4),(4,6)])
    if result is not None:
        bU, bV, bW, brank = result
        if verify(bU, bV, bW, brank):
            best_U, best_V, best_W, best_rank = bU, bV, bW, brank
    
    # Fallback: schoolbook
    if best_rank == 72:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
