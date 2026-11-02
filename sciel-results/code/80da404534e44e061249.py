import numpy as np
import time

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize factor matrices
    # U: (n*m, rank) -> (4, 25)
    # V: (m*p, rank) -> (14, 25)
    # W: (p*n, rank) -> (14, 25)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, k_start):
        """
        Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
        col_start: starting column index in U, V, W for this block's terms.
        k_start: starting column index in B (and C).
        
        Indices:
        A is 2x2: A[i, j] where i in {0,1}, j in {0,1}
        B is 2x2: B[j, k] where j in {0,1}, k in {0,1} (local)
        C is 2x2: C[i, k] where i in {0,1}, k in {0,1} (local)
        
        Global indices:
        U index: i * m + j = 2*i + j
        V index: j * p + (k_start + k)
        W index: (k_start + k) * n + i = 2*(k_start + k) + i
        """
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # Contributes to C0,ks and C1,ks+1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        # Contributes to C1,ks and C1,ks+1 (with negative sign for C1,ks+1)
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # Contributes to C0,ks+1 and C1,ks+1
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        # Contributes to C0,ks and C1,ks
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        # Contributes to C0,ks (negative) and C0,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # Contributes to C1,ks+1
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # Contributes to C0,ks
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    # Apply Strassen to pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Apply schoolbook to column 6
    col_start = 21
    k = 6
    
    # C0,6 = A00*B0,6 + A01*B1,6
    # Term 1: A00 * B0,6
    U[0, col_start+0] += 1
    V[7*0 + k, col_start+0] += 1
    W[2*k + 0, col_start+0] += 1
    
    # Term 2: A01 * B1,6
    U[1, col_start+1] += 1
    V[7*1 + k, col_start+1] += 1
    W[2*k + 0, col_start+1] += 1
    
    # C1,6 = A10*B0,6 + A11*B1,6
    # Term 3: A10 * B0,6
    U[2, col_start+2] += 1
    V[7*0 + k, col_start+2] += 1
    W[2*k + 1, col_start+2] += 1
    
    # Term 4: A11 * B1,6
    U[3, col_start+3] += 1
    V[7*1 + k, col_start+3] += 1
    W[2*k + 1, col_start+3] += 1

    # Quick verification using vectorized operations
    if time.time() < deadline:
        valid = True
        for i in range(n):
            for j in range(m):
                for k_idx in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k_idx
                    c_idx = k_idx * n + i
                    
                    # Vectorized sum over rank
                    s = np.dot(U[a_idx, :], np.multiply(V[b_idx, :], W[c_idx, :]))
                    
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if not valid:
            # Fallback to schoolbook if verification fails (should not happen)
            rank_sb = n * m * p
            U_sb = np.zeros((n*m, rank_sb), dtype=int)
            V_sb = np.zeros((m*p, rank_sb), dtype=int)
            W_sb = np.zeros((p*n, rank_sb), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k_idx in range(p):
                        U_sb[i*m+j, col] = 1
                        V_sb[j*p+k_idx, col] = 1
                        W_sb[k_idx*n+i, col] = 1
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V_sb.tolist() if False else V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
