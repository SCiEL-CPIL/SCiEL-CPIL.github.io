"""Program to find a rank-minimal exact tensor decomposition for 2x3 by 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    
    # Initialize U, V, W with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_block(u_offset, v_offset, w_offset, start_col):
        """
        Add Strassen 2x2x2 block contributions.
        u_offset: index in U for A rows (2x2)
        v_offset: index in V for B rows (2x2)
        w_offset: index in W for C cols (2x2)
        start_col: starting column in rank
        """
        col = start_col
        # M1 = (A00+A11)(B00+B11)
        U[u_offset, col] = 1; U[u_offset+2, col] = 1
        V[v_offset, col] = 1; V[v_offset+2, col] = 1
        W[w_offset, col] = 1; W[w_offset+2, col] = -1; W[w_offset+4, col] = 1; W[w_offset+5, col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[u_offset+2, col] = 1; U[u_offset+3, col] = 1
        V[v_offset, col] = 1
        W[w_offset+2, col] = 1; W[w_offset+5, col] = -1
        col += 1
        # M3 = A00(B10-B11)
        U[u_offset, col] = 1
        V[v_offset+2, col] = 1; V[v_offset+3, col] = -1
        W[w_offset+1, col] = 1; W[w_offset+5, col] = 1
        col += 1
        # M4 = A11(B01-B00)
        U[u_offset+3, col] = 1
        V[v_offset+1, col] = 1; V[v_offset, col] = -1
        W[w_offset, col] = 1; W[w_offset+2, col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[u_offset, col] = 1; U[u_offset+1, col] = 1
        V[v_offset+3, col] = 1
        W[w_offset, col] = -1; W[w_offset+1, col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[u_offset+2, col] = 1; U[u_offset, col] = -1
        V[v_offset, col] = 1; V[v_offset+1, col] = 1
        W[w_offset+5, col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[u_offset+1, col] = 1; U[u_offset+3, col] = -1
        V[v_offset+2, col] = 1; V[v_offset+3, col] = 1
        W[w_offset, col] = 1; W[w_offset+1, col] = -1
        col += 1
    
    def trivial_block(u_offset, v_offset, w_offset, start_col):
        """
        Add trivial 2x1x1 block contributions (rank 2 for 2x1x1, but we need 2x1x4).
        Actually, for 2x1x4, rank is 4 (each C entry needs its own product).
        A is 2x1, B is 1x4, C is 2x4.
        C[i,k] = A[i,0] * B[0,k]
        So we need 4 multiplications, one for each k.
        """
        for k in range(4):
            col = start_col + k
            U[u_offset, col] = 1; U[u_offset+1, col] = 1
            V[v_offset + k, col] = 1
            W[w_offset + k, col] = 1
    
    # Block 1: A rows 0-1, B rows 0-1, B cols 0-3 (2x2x4)
    # This is 2 Strassen 2x2x2 blocks side by side
    strassen_block(0, 0, 0, 0)  # B cols 0-1
    strassen_block(0, 4, 4, 7)  # B cols 2-3, W offset 4 (cols 4-7 of C)
    
    # Block 2: A row 2, B row 2, B cols 0-3 (2x1x4)
    trivial_block(4, 8, 0, 14)
    
    # Block 3: A rows 0-1, B rows 0-1, B cols 4-7 (2x2x4)
    strassen_block(0, 12, 8, 18)  # B cols 4-5
    strassen_block(0, 16, 12, 25)  # B cols 6-7
    
    # Block 4: A row 2, B row 2, B cols 4-7 (2x1x4)
    trivial_block(4, 16, 8, 32)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def verify_decomp(U, V, W, n, m, p, tol=1e-6):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r_idx in range(r):
                    s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                if abs(s - 1.0) > tol:
                    return False
    return True

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    print(f"Rank: {len(result['U'][0])}")
    valid = verify_decomp(result['U'], result['V'], result['W'], n, m, p)
    print(f"Valid: {valid}")
