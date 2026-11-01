import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True
    
    def strassen_2x2x2x2(U, V, W, base_col, a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        """Apply Strassen 2x2 x 2x2 at base_col. Returns 7 columns used."""
        # M1 = (a11+a22)(b11+b22) -> c11 += M1, c22 += M1
        col = base_col
        U[a11, col] += 1; U[a22, col] += 1
        V[b11, col] += 1; V[b22, col] += 1
        W[c11, col] += 1; W[c22, col] += 1
        
        # M2 = (a21+a22)b11 -> c21 += M2, c22 -= M2
        col = base_col + 1
        U[a21, col] += 1; U[a22, col] += 1
        V[b11, col] += 1
        W[c21, col] += 1; W[c22, col] -= 1
        
        # M3 = a11(b12-b22) -> c12 += M3, c22 += M3
        col = base_col + 2
        U[a11, col] += 1
        V[b12, col] += 1; V[b22, col] -= 1
        W[c12, col] += 1; W[c22, col] += 1
        
        # M4 = a22(b21-b11) -> c11 += M4, c21 += M4
        col = base_col + 3
        U[a22, col] += 1
        V[b21, col] += 1; V[b11, col] -= 1
        W[c11, col] += 1; W[c21, col] += 1
        
        # M5 = (a11+a12)b22 -> c11 -= M5, c12 += M5
        col = base_col + 4
        U[a11, col] += 1; U[a12, col] += 1
        V[b22, col] += 1
        W[c11, col] -= 1; W[c12, col] += 1
        
        # M6 = (a21-a11)(b11+b12) -> c22 += M6
        col = base_col + 5
        U[a21, col] += 1; U[a11, col] -= 1
        V[b11, col] += 1; V[b12, col] += 1
        W[c22, col] += 1
        
        # M7 = (a12-a22)(b21+b22) -> c11 += M7
        col = base_col + 6
        U[a12, col] += 1; U[a22, col] -= 1
        V[b21, col] += 1; V[b22, col] += 1
        W[c11, col] += 1
        
        return base_col + 7
    
    def strassen_56():
        """8 independent 2x2 x 2x2 Strassen blocks.
        A is 2x4, B is 4x8, C is 2x8.
        Partition: for each pair of A columns (2u, 2u+1) and B rows (2u, 2u+1),
        and for each pair of B columns (2v, 2v+1), do a 2x2 x 2x2 Strassen.
        But this doesn't work directly because the dimensions don't align for standard Strassen.
        
        Actually, let's think of it as:
        C = A * B where A is 2x4, B is 4x8.
        We can write this as 4 independent 2x1 x 1x8 problems? No.
        
        Better: partition B into 4 blocks of 4x2: B = [B0 | B1 | B2 | B3]
        C = [A*B0 | A*B1 | A*B2 | A*B3]
        Each A*Bv is 2x4 x 4x2 = 2x2.
        
        For each 2x4 x 4x2: partition A = [A1 | A2] (2x2 each), Bv = [Bv1; Bv2] (2x2 each)
        Then A*Bv = A1*Bv1 + A2*Bv2, each 2x2 x 2x2, rank 7 each.
        Total per block: 14. Four blocks: 56.
        """
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for v in range(4):  # 4 blocks of B (each 4x2)
            for half in range(2):  # A1 or A2
                # A half: columns 2*half, 2*half+1
                a11 = 0 * m + 2 * half      # A[0, 2*half]
                a12 = 0 * m + 2 * half + 1  # A[0, 2*half+1]
                a21 = 1 * m + 2 * half      # A[1, 2*half]
                a22 = 1 * m + 2 * half + 1  # A[1, 2*half+1]
                
                # B half: rows 2*half, 2*half+1, columns 2*v, 2*v+1
                b11 = (2 * half) * p + 2 * v        # B[2*half, 2*v]
                b12 = (2 * half) * p + 2 * v + 1    # B[2*half, 2*v+1]
                b21 = (2 * half + 1) * p + 2 * v    # B[2*half+1, 2*v]
                b22 = (2 * half + 1) * p + 2 * v + 1  # B[2*half+1, 2*v+1]
                
                # C: columns 2*v, 2*v+1
                c11 = (2 * v) * n + 0      # C[0, 2*v]
                c12 = (2 * v + 1) * n + 0  # C[0, 2*v+1]
                c21 = (2 * v) * n + 1      # C[1, 2*v]
                c22 = (2 * v + 1) * n + 1  # C[1, 2*v+1]
                
                col = strassen_2x2x2x2(U, V, W, col, a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22)
        
        return U, V, W, rank
    
    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_56()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
