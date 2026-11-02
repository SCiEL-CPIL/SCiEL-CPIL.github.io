import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115  # Safety margin before 120s limit

    def verify_decomp(U, V, W, trials=30):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
        Ua = np.array(U, dtype=np.float64) if isinstance(U, list) else U.astype(np.float64)
        Va = np.array(V, dtype=np.float64) if isinstance(V, list) else V.astype(np.float64)
        Wa = np.array(W, dtype=np.float64) if isinstance(W, list) else W.astype(np.float64)
        
        for t in range(trials):
            A = np.random.randint(-10, 11, size=(n, m)).astype(np.float64)
            B = np.random.randint(-10, 11, size=(m, p)).astype(np.float64)
            C_true = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(rank):
                u_val = 0.0
                for i in range(n):
                    for j in range(m):
                        u_val += Ua[i*m+j, r] * A[i, j]
                v_val = 0.0
                for j in range(m):
                    for k in range(p):
                        v_val += Va[j*p+k, r] * B[j, k]
                prod = u_val * v_val
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += Wa[k*n+i, r] * prod

            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    def build_rank25():
        """
        Build the known rank-25 decomposition for 2x2 * 2x7.
        Uses Strassen (rank 7) for three 2x2 blocks (cols 0-1, 2-3, 4-5)
        and schoolbook (rank 4) for the last column (col 6).
        Total rank: 7 + 7 + 7 + 4 = 25.
        """
        rank = 25
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_strassen(col_start, k_start):
            # Strassen for 2x2 * 2x2 block starting at column k_start
            # The 2x2 block of B is B[0:2, k_start:k_start+2]
            # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            # V indices: j*p+k -> 0*7+k, 1*7+k
            # W indices: k*n+i -> k*2+i
            
            # M1 = (A00+A11)*(B00+B11)
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[0*7+k_start, col_start+0] += 1; V[1*7+k_start+1, col_start+0] += 1
            W[k_start*n+0, col_start+0] += 1
            W[k_start*n+1, col_start+0] -= 1
            W[(k_start+1)*n+0, col_start+0] += 1
            W[(k_start+1)*n+1, col_start+0] += 1

            # M2 = (A10+A11)*B00
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[0*7+k_start, col_start+1] += 1
            W[k_start*n+1, col_start+1] += 1
            W[(k_start+1)*n+1, col_start+1] -= 1

            # M3 = A00*(B01-B11)
            U[0, col_start+2] += 1
            V[0*7+k_start+1, col_start+2] += 1; V[1*7+k_start+1, col_start+2] -= 1
            W[(k_start+1)*n+0, col_start+2] += 1
            W[(k_start+1)*n+1, col_start+2] += 1

            # M4 = A11*(B10-B00)
            U[3, col_start+3] += 1
            V[1*7+k_start, col_start+3] += 1; V[0*7+k_start, col_start+3] -= 1
            W[k_start*n+0, col_start+3] += 1
            W[k_start*n+1, col_start+3] += 1

            # M5 = (A00+A01)*B11
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[1*7+k_start+1, col_start+4] += 1
            W[k_start*n+0, col_start+4] -= 1
            W[(k_start+1)*n+0, col_start+4] += 1

            # M6 = (A10-A00)*(B00+B01)
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[0*7+k_start, col_start+5] += 1; V[0*7+k_start+1, col_start+5] += 1
            W[(k_start+1)*n+1, col_start+5] += 1

            # M7 = (A01-A11)*(B10+B11)
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[1*7+k_start, col_start+6] += 1; V[1*7+k_start+1, col_start+6] += 1
            W[k_start*n+0, col_start+6] += 1

        # Apply Strassen to blocks [0:2], [2:4], [4:6]
        add_strassen(0, 0)
        add_strassen(7, 2)
        add_strassen(14, 4)

        # Schoolbook for column 6
        cs = 21
        # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
        U[0, cs+0] += 1; V[0*7+6, cs+0] += 1; W[6*n+0, cs+0] += 1
        U[1, cs+1] += 1; V[1*7+6, cs+1] += 1; W[6*n+0, cs+1] += 1
        # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
        U[2, cs+2] += 1; V[0*7+6, cs+2] += 1; W[6*n+1, cs+2] += 1
        U[3, cs+3] += 1; V[1*7+6, cs+3] += 1; W[6*n+1, cs+3] += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_rank28():
        """Schoolbook rank-28 decomposition (fallback)."""
        rank_sb = 28
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Primary strategy: Rank 25 (Strassen + Schoolbook)
    result = build_rank25()
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result

    # Fallback: Rank 28 (Schoolbook)
    result = build_rank28()
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result

    # Last resort: Return rank 28 without verification (should not happen)
    return result
