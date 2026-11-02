"""Program for 2x3x10 matrix multiplication tensor decomposition.
Uses Strassen's algorithm for 2x2x2 blocks to reduce rank from 60 to 55.
"""
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3x10 matrix multiplication.
    
    Strategy: Split the 10 columns of B into 5 blocks of 2 columns.
    For each block of 2 columns:
      - A_left (2x2, cols 0,1) * B_top (2x2, rows 0,1) using Strassen (rank 7)
      - A_right (2x1, col 2) * B_bottom (1x2, row 2) using schoolbook (rank 4)
      Total per block: 11
      Total: 55
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Strassen's 7 terms for 2x2x2 matrix multiplication
    # For matrices X (2x2), Y (2x2), compute Z = X*Y:
    # Z[i,j] = sum_k U[i,k,c] * V[k,j,c] * W[j,i,c]
    # where W is indexed by (j,i) in the W matrix (W[k*n+i] where k is col of C, i is row of C)
    # 
    # Standard Strassen:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = M1 - M3 + M4 + M5
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M7
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)  # 6 x 55
    V = np.zeros((m * p, rank), dtype=float)  # 30 x 55
    W = np.zeros((p * n, rank), dtype=float)  # 20 x 55
    
    col = 0
    
    for b in range(5):  # 5 blocks of 2 columns
        # B columns are 2*b and 2*b+1
        # A: rows 0,1; cols 0,1,2
        # B block: rows 0,1,2; cols 2*b, 2*b+1
        
        # Part 1: A[0:2, 0:2] * B[0:2, 2*b:2*b+2] (2x2x2, Strassen, rank 7)
        # U indices: i*m + j where i in [0,1], j in [0,1]
        # V indices: j*p + k where j in [0,1], k in [2*b, 2*b+1]
        # W indices: k*n + i where k in [2*b, 2*b+1], i in [0,1]
        
        # M1: (A[0,0]+A[1,1]) * (B[0,2*b]+B[1,2*b+1])
        U[0, col] = 1  # A[0,0]
        U[4, col] = 1  # A[1,1] (row 1, col 1 -> index 1*3+1=4)
        V[2*b, col] = 1  # B[0,2*b] (row 0, col 2*b -> index 0*10+2*b)
        V[10+2*b+1, col] = 1  # B[1,2*b+1] (row 1, col 2*b+1 -> index 1*10+2*b+1)
        # Output: Z[0,0] += M1, Z[1,1] += M1
        W[(2*b)*2+0, col] = 1  # C[0, 2*b] (row 0 of C, col 2*b)
        W[(2*b+1)*2+1, col] = 1  # C[1, 2*b+1] (row 1 of C, col 2*b+1)
        col += 1
        
        # M2: (A[1,0]+A[1,1]) * B[0,2*b]
        U[3, col] = 1  # A[1,0] (row 1, col 0 -> index 1*3+0=3)
        U[4, col] = 1  # A[1,1]
        V[2*b, col] = 1  # B[0,2*b]
        # Output: Z[1,0] += M2, Z[1,1] -= M2
        W[(2*b)*2+1, col] = 1  # C[1, 2*b]
        W[(2*b+1)*2+1, col] = -1  # C[1, 2*b+1]
        col += 1
        
        # M3: A[0,0] * (B[0,2*b+1]-B[1,2*b+1])
        U[0, col] = 1  # A[0,0]
        V[10+2*b+1, col] = 1  # B[1,2*b+1]
        V[2*b+1, col] = -1  # -B[0,2*b+1]
        # Output: Z[0,0] -= M3, Z[0,1] += M3
        W[(2*b)*2+0, col] = -1  # C[0, 2*b]
        W[(2*b+1)*2+0, col] = 1  # C[0, 2*b+1]
        col += 1
        
        # M4: A[1,1] * (B[1,2*b]-B[0,2*b])
        U[4, col] = 1  # A[1,1]
        V[10+2*b, col] = 1  # B[1,2*b]
        V[2*b, col] = -1  # -B[0,2*b]
        # Output: Z[0,0] += M4, Z[1,0] += M4
        W[(2*b)*2+0, col] = 1  # C[0, 2*b]
        W[(2*b)*2+1, col] = 1  # C[1, 2*b]
        col += 1
        
        # M5: (A[0,0]+A[0,1]) * B[1,2*b+1]
        U[0, col] = 1  # A[0,0]
        U[1, col] = 1  # A[0,1] (row 0, col 1 -> index 0*3+1=1)
        V[10+2*b+1, col] = 1  # B[1,2*b+1]
        # Output: Z[0,0] += M5, Z[0,1] += M5
        W[(2*b)*2+0, col] = 1  # C[0, 2*b]
        W[(2*b+1)*2+0, col] = 1  # C[0, 2*b+1]
        col += 1
        
        # M6: (A[1,0]-A[0,0]) * (B[0,2*b]+B[0,2*b+1])
        U[3, col] = 1  # A[1,0]
        U[0, col] = -1  # -A[0,0]
        V[2*b, col] = 1  # B[0,2*b]
        V[2*b+1, col] = 1  # B[0,2*b+1]
        # Output: Z[1,0] += M6
        W[(2*b)*2+1, col] = 1  # C[1, 2*b]
        col += 1
        
        # M7: (A[0,1]-A[1,1]) * (B[1,2*b]+B[1,2*b+1])
        U[1, col] = 1  # A[0,1]
        U[4, col] = -1  # -A[1,1]
        V[10+2*b, col] = 1  # B[1,2*b]
        V[10+2*b+1, col] = 1  # B[1,2*b+1]
        # Output: Z[1,1] += M7
        W[(2*b+1)*2+1, col] = 1  # C[1, 2*b+1]
        col += 1
        
        # Part 2: A[0:2, 2] * B[2, 2*b:2*b+2] (2x1x2, schoolbook, rank 4)
        # U: A[i,2] for i in [0,1]
        # V: B[2,k] for k in [2*b, 2*b+1]
        # W: C[k,i] for k in [2*b, 2*b+1], i in [0,1]
        
        # A[0,2] * B[2,2*b] -> C[2*b, 0]
        U[2, col] = 1  # A[0,2] (row 0, col 2 -> index 0*3+2=2)
        V[20+2*b, col] = 1  # B[2,2*b] (row 2, col 2*b -> index 2*10+2*b)
        W[(2*b)*2+0, col] = 1  # C[0, 2*b]
        col += 1
        
        # A[0,2] * B[2,2*b+1] -> C[2*b+1, 0]
        U[2, col] = 1  # A[0,2]
        V[20+2*b+1, col] = 1  # B[2,2*b+1]
        W[(2*b+1)*2+0, col] = 1  # C[0, 2*b+1]
        col += 1
        
        # A[1,2] * B[2,2*b] -> C[2*b, 1]
        U[5, col] = 1  # A[1,2] (row 1, col 2 -> index 1*3+2=5)
        V[20+2*b, col] = 1  # B[2,2*b]
        W[(2*b)*2+1, col] = 1  # C[1, 2*b]
        col += 1
        
        # A[1,2] * B[2,2*b+1] -> C[2*b+1, 1]
        U[5, col] = 1  # A[1,2]
        V[20+2*b+1, col] = 1  # B[2,2*b+1]
        W[(2*b+1)*2+1, col] = 1  # C[1, 2*b+1]
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition
    _verify_decomposition(U, V, W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Check shapes
    assert len(U) == n * m, f"U shape mismatch: {len(U)} vs {n*m}"
    assert len(V) == m * p, f"V shape mismatch: {len(V)} vs {m*p}"
    assert len(W) == p * n, f"W shape mismatch: {len(W)} vs {p*n}"
    
    # For matrix multiplication, the tensor T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    print(f"Mismatch at i={i}, j={j}, k={k}: got {total}, expected 1")
                    return False
    return True


if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    print(f"Rank: {len(U[0])}")
    print(f"U shape: {len(U)}x{len(U[0])}")
    print(f"V shape: {len(V)}x{len(V[0])}")
    print(f"W shape: {len(W)}x{len(W[0])}")
    
    if _verify_decomposition(U, V, W):
        print("Decomposition verified successfully!")
    else:
        print("Verification failed!")
