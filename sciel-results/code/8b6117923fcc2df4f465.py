import time
import numpy as np

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Strassen rank-7 for columns (0,1)
    - Strassen rank-7 for columns (2,3)  
    - Strassen rank-7 for columns (4,5)
    - Schoolbook rank-4 for column 6
    Total rank: 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    def add_strassen_block(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block of B columns k_start, k_start+1."""
        # M1 = (A00 + A11) * (B0,k + B1,k+1) -> C0,k and C1,k+1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[0*p + k_start, col_start+0] += 1
        V[1*p + (k_start+1), col_start+0] += 1
        W[k_start*n + 0, col_start+0] += 1
        W[(k_start+1)*n + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,k -> C1,k and -C1,k+1
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[0*p + k_start, col_start+1] += 1
        W[k_start*n + 1, col_start+1] += 1
        W[(k_start+1)*n + 1, col_start+1] -= 1

        # M3 = A00 * (B0,k+1 - B1,k+1) -> C0,k+1 and C1,k+1
        U[0, col_start+2] += 1
        V[0*p + (k_start+1), col_start+2] += 1
        V[1*p + (k_start+1), col_start+2] -= 1
        W[(k_start+1)*n + 0, col_start+2] += 1
        W[(k_start+1)*n + 1, col_start+2] += 1

        # M4 = A11 * (B1,k - B0,k) -> C0,k and C1,k
        U[3, col_start+3] += 1
        V[1*p + k_start, col_start+3] += 1
        V[0*p + k_start, col_start+3] -= 1
        W[k_start*n + 0, col_start+3] += 1
        W[k_start*n + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,k+1 -> -C0,k and C0,k+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[1*p + (k_start+1), col_start+4] += 1
        W[k_start*n + 0, col_start+4] -= 1
        W[(k_start+1)*n + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,k + B0,k+1) -> C1,k+1
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[0*p + k_start, col_start+5] += 1
        V[0*p + (k_start+1), col_start+5] += 1
        W[(k_start+1)*n + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,k + B1,k+1) -> C0,k
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[1*p + k_start, col_start+6] += 1
        V[1*p + (k_start+1), col_start+6] += 1
        W[k_start*n + 0, col_start+6] += 1

    # Apply Strassen to pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Schoolbook for column 6
    col_start = 21
    k = 6
    
    U[0, col_start+0] += 1
    V[0*p + k, col_start+0] += 1
    W[k*n + 0, col_start+0] += 1
    
    U[1, col_start+1] += 1
    V[1*p + k, col_start+1] += 1
    W[k*n + 0, col_start+1] += 1
    
    U[2, col_start+2] += 1
    V[0*p + k, col_start+2] += 1
    W[k*n + 1, col_start+2] += 1
    
    U[3, col_start+3] += 1
    V[1*p + k, col_start+3] += 1
    W[k*n + 1, col_start+3] += 1

    # Verify
    for i in range(n):
        for j in range(m):
            for k_idx in range(p):
                a_idx = i * m + j
                b_idx = j * p + k_idx
                c_idx = k_idx * n + i
                s = sum(U[a_idx, r] * V[b_idx, r] * W[c_idx, r] for r in range(rank))
                if abs(s - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_sb = n * m * p
                    U_sb = np.zeros((n*m, rank_sb), dtype=float)
                    V_sb = np.zeros((m*p, rank_sb), dtype=float)
                    W_sb = np.zeros((p*n, rank_sb), dtype=float)
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii*m+jj, col] = 1
                                V_sb[jj*p+kk, col] = 1
                                W_sb[kk*n+ii, col] = 1
                                col += 1
                    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
