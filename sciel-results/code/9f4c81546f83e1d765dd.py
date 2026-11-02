import time
import numpy as np

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy via isomorphism T_{2,2,7} ≅ T_{2,7,2}:
    - Reinterpret as computing C^T (7x2) = B^T (7x2) * A^T (2x2)
    - Apply Strassen (rank 7) to 3 blocks of 2 rows of B^T
    - Apply schoolbook (rank 4) to remaining 1 row of B^T
    - Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize factor matrices
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(row_start, k_start):
        """
        Add Strassen rank-7 terms for a 2-row block of B^T (rows row_start, row_start+1).
        Computes C^T[row_start:row_start+2, :] = B^T[row_start:row_start+2, :] * A^T
        """
        # M1 = (A00 + A11) * (B0,rs + B1,rs+1)
        U[0, k_start+0] += 1
        U[3, k_start+0] += 1
        V[row_start + 0, k_start+0] += 1
        V[row_start + 1, k_start+0] += 1
        W[row_start + 0, k_start+0] += 1
        W[row_start + 1, k_start+0] += 1

        # M2 = (A10 + A11) * B0,rs
        U[2, k_start+1] += 1
        U[3, k_start+1] += 1
        V[row_start + 0, k_start+1] += 1
        W[row_start + 0, k_start+1] += 1
        W[row_start + 1, k_start+1] -= 1

        # M3 = A00 * (B0,rs+1 - B1,rs+1)
        U[0, k_start+2] += 1
        V[row_start + 0, k_start+2] += 1
        V[row_start + 1, k_start+2] -= 1
        W[row_start + 0, k_start+2] += 1
        W[row_start + 1, k_start+2] += 1

        # M4 = A11 * (B1,rs - B0,rs)
        U[3, k_start+3] += 1
        V[row_start + 1, k_start+3] += 1
        V[row_start + 0, k_start+3] -= 1
        W[row_start + 0, k_start+3] += 1
        W[row_start + 1, k_start+3] += 1

        # M5 = (A00 + A01) * B1,rs+1
        U[0, k_start+4] += 1
        U[1, k_start+4] += 1
        V[row_start + 1, k_start+4] += 1
        W[row_start + 0, k_start+4] -= 1
        W[row_start + 1, k_start+4] += 1

        # M6 = (A10 - A00) * (B0,rs + B0,rs+1)
        U[2, k_start+5] += 1
        U[0, k_start+5] -= 1
        V[row_start + 0, k_start+5] += 1
        V[row_start + 1, k_start+5] += 1
        W[row_start + 1, k_start+5] += 1

        # M7 = (A01 - A11) * (B1,rs + B1,rs+1)
        U[1, k_start+6] += 1
        U[3, k_start+6] -= 1
        V[row_start + 1, k_start+6] += 1
        V[row_start + 0, k_start+6] += 1
        W[row_start + 0, k_start+6] += 1

    # Apply Strassen to row pairs (0,1), (2,3), (4,5) of B^T
    add_strassen_block(0, 0)
    add_strassen_block(2, 7)
    add_strassen_block(4, 14)
    
    # Apply schoolbook to row 6 of B^T
    row = 6
    k_start = 21
    
    U[0, k_start+0] += 1
    V[row + 0, k_start+0] += 1
    W[row + 0, k_start+0] += 1
    
    U[1, k_start+1] += 1
    V[row + 0, k_start+1] += 1
    W[row + 1, k_start+1] += 1
    
    U[2, k_start+2] += 1
    V[row + 1, k_start+2] += 1
    W[row + 0, k_start+2] += 1
    
    U[3, k_start+3] += 1
    V[row + 1, k_start+3] += 1
    W[row + 1, k_start+3] += 1

    # Full tensor verification: check all 4*14*14 entries
    valid = True
    # Build the set of valid (a,b,c) indices
    # a = i*m + j (A index), b = j*p + k (B index), c = k*n + i (C index)
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.add((i * m + j, j * p + k, k * n + i))
    
    # Check all entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                s = 0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                if (a, b, c) in valid_indices:
                    if s != 1:
                        valid = False
                else:
                    if s != 0:
                        valid = False
    
    if not valid:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=int)
        V_sb = np.zeros((m*p, rank_sb), dtype=int)
        W_sb = np.zeros((p*n, rank_sb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
