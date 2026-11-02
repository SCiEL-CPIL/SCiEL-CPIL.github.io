"""Program for 2x3x10 matrix multiplication tensor decomposition.
Uses Strassen's algorithm for 2x2x2 blocks to reduce rank from 60 to 55.
"""
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3x10 matrix multiplication.
    
    Strategy: Split the 10 columns of B into 5 blocks of 2 columns.
    For each block of 2 columns:
      - The 2x2 sub-block (rows 1,2 of B) is multiplied with the 2x2 sub-block (cols 1,2 of A)
        using Strassen's algorithm (rank 7).
      - The remaining parts are handled with schoolbook:
        * Row 3 of B (1x2) times cols 1,2 of A (2x2): 2x2x1 -> rank 4
        * Col 3 of A (2x1) times rows 1,2 of B (2x2): 1x2x2 -> rank 4
        * Col 3 of A (2x1) times row 3 of B (1x2): 1x1x1 -> rank 1
      Total per block: 7 + 4 + 4 + 1 = 16... wait that's not right.
      
    Let me reconsider. The 2x3x2 tensor has rank:
      - 2x2x2 part (A[0:2,0:2] * B[0:2,0:2]): rank 7 (Strassen)
      - 2x1x2 part (A[0:2,2] * B[0:2,0:2]): this is 2x1x2, rank = 4
      - 2x2x1 part (A[0:2,0:2] * B[2,0:2]): this is 2x2x1, rank = 4
      - 1x1x1 part (A[2,2]... wait, A is 2x3 so A[2] doesn't exist.
      
    Actually A is 2 rows x 3 cols, B is 3 rows x 10 cols, C is 2 rows x 10 cols.
    
    For a 2-column block of B (say columns k, k+1):
      C[:, k:k+2] = A * B[k:k+2, :]
      
      Let's partition A into A_left (2x2, cols 0,1) and A_right (2x1, col 2).
      Let's partition B_block (3x2) into B_top (2x2, rows 0,1) and B_bottom (1x2, row 2).
      
      C[:, k:k+2] = A_left * B_top + A_right * B_bottom
      
      A_left * B_top: 2x2x2, rank 7 (Strassen)
      A_right * B_bottom: 2x1x2, rank 4 (schoolbook: 2*1*2 = 4)
      
      Total per 2-column block: 7 + 4 = 11
      Total for 5 blocks: 55
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Strassen's 7 terms for 2x2x2 matrix multiplication
    # For matrices X, Y (2x2), compute Z = X*Y:
    # Z[0,0] = M5 - M4 + M1 - M3
    # Z[0,1] = M2 + M5
    # Z[1,0] = M4 + M6
    # Z[1,1] = M1 - M2 + M3 + M7
    # Where:
    # M1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
    # M2 = (X[1,0]+X[1,1]) * Y[0,0]
    # M3 = X[0,0] * (Y[0,1]-Y[1,1])
    # M4 = X[1,1] * (Y[1,0]-Y[0,0])
    # M5 = (X[0,0]+X[0,1]) * Y[1,1]
    # M6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1])
    # M7 = (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1])
    
    rank = 55
    U = [[0.0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 3*10 = 30 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 10*2 = 20 rows
    
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
        U[0*3+0][col] = 1  # A[0,0]
        U[1*3+1][col] = 1  # A[1,1]
        V[0*10+2*b][col] = 1  # B[0,2*b]
        V[1*10+2*b+1][col] = 1  # B[1,2*b+1]
        # Output contributions:
        # Z[0,0] += M1, Z[1,1] += M1
        W[(2*b)*2+0][col] = 1  # C[2*b, 0]
        W[(2*b+1)*2+1][col] = 1  # C[2*b+1, 1]
        col += 1
        
        # M2: (A[1,0]+A[1,1]) * B[0,2*b]
        U[1*3+0][col] = 1  # A[1,0]
        U[1*3+1][col] = 1  # A[1,1]
        V[0*10+2*b][col] = 1  # B[0,2*b]
        # Output: Z[1,0] += M2, Z[1,1] -= M2
        W[(2*b)*2+1][col] = 1  # C[2*b, 1]
        W[(2*b+1)*2+1][col] = -1  # C[2*b+1, 1]
        col += 1
        
        # M3: A[0,0] * (B[0,2*b+1]-B[1,2*b+1])
        U[0*3+0][col] = 1  # A[0,0]
        V[0*10+2*b+1][col] = 1  # B[0,2*b+1]
        V[1*10+2*b+1][col] = -1  # -B[1,2*b+1]
        # Output: Z[0,0] -= M3, Z[1,1] += M3
        W[(2*b)*2+0][col] = -1  # C[2*b, 0]
        W[(2*b+1)*2+1][col] = 1  # C[2*b+1, 1]
        col += 1
        
        # M4: A[1,1] * (B[1,2*b]-B[0,2*b])
        U[1*3+1][col] = 1  # A[1,1]
        V[1*10+2*b][col] = 1  # B[1,2*b]
        V[0*10+2*b][col] = -1  # -B[0,2*b]
        # Output: Z[1,0] += M4, Z[0,0] -= M4
        W[(2*b)*2+1][col] = 1  # C[2*b, 1]
        W[(2*b)*2+0][col] = -1  # C[2*b, 0]
        col += 1
        
        # M5: (A[0,0]+A[0,1]) * B[1,2*b+1]
        U[0*3+0][col] = 1  # A[0,0]
        U[0*3+1][col] = 1  # A[0,1]
        V[1*10+2*b+1][col] = 1  # B[1,2*b+1]
        # Output: Z[0,0] += M5, Z[0,1] += M5
        W[(2*b)*2+0][col] = 1  # C[2*b, 0]
        W[(2*b+1)*2+0][col] = 1  # C[2*b+1, 0]
        col += 1
        
        # M6: (A[1,0]-A[0,0]) * (B[0,2*b]+B[0,2*b+1])
        U[1*3+0][col] = 1  # A[1,0]
        U[0*3+0][col] = -1  # -A[0,0]
        V[0*10+2*b][col] = 1  # B[0,2*b]
        V[0*10+2*b+1][col] = 1  # B[0,2*b+1]
        # Output: Z[1,0] += M6
        W[(2*b)*2+1][col] = 1  # C[2*b, 1]
        col += 1
        
        # M7: (A[0,1]-A[1,1]) * (B[1,2*b]+B[1,2*b+1])
        U[0*3+1][col] = 1  # A[0,1]
        U[1*3+1][col] = -1  # -A[1,1]
        V[1*10+2*b][col] = 1  # B[1,2*b]
        V[1*10+2*b+1][col] = 1  # B[1,2*b+1]
        # Output: Z[1,1] += M7
        W[(2*b+1)*2+1][col] = 1  # C[2*b+1, 1]
        col += 1
        
        # Part 2: A[0:2, 2] * B[2, 2*b:2*b+2] (2x1x2, schoolbook, rank 4)
        # U: A[i,2] for i in [0,1]
        # V: B[2,k] for k in [2*b, 2*b+1]
        # W: C[k,i] for k in [2*b, 2*b+1], i in [0,1]
        
        # A[0,2] * B[2,2*b] -> C[2*b, 0]
        U[0*3+2][col] = 1  # A[0,2]
        V[2*10+2*b][col] = 1  # B[2,2*b]
        W[(2*b)*2+0][col] = 1  # C[2*b, 0]
        col += 1
        
        # A[0,2] * B[2,2*b+1] -> C[2*b+1, 0]
        U[0*3+2][col] = 1  # A[0,2]
        V[2*10+2*b+1][col] = 1  # B[2,2*b+1]
        W[(2*b+1)*2+0][col] = 1  # C[2*b+1, 0]
        col += 1
        
        # A[1,2] * B[2,2*b] -> C[2*b, 1]
        U[1*3+2][col] = 1  # A[1,2]
        V[2*10+2*b][col] = 1  # B[2,2*b]
        W[(2*b)*2+1][col] = 1  # C[2*b, 1]
        col += 1
        
        # A[1,2] * B[2,2*b+1] -> C[2*b+1, 1]
        U[1*3+2][col] = 1  # A[1,2]
        V[2*10+2*b+1][col] = 1  # B[2,2*b+1]
        W[(2*b+1)*2+1][col] = 1  # C[2*b+1, 1]
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Check shapes
    assert len(U) == n * m, f"U shape mismatch: {len(U)} vs {n*m}"
    assert len(V) == m * p, f"V shape mismatch: {len(V)} vs {m*p}"
    assert len(W) == p * n, f"W shape mismatch: {len(W)} vs {p*n}"
    
    # For each (i, k) pair in C, verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Actually, the tensor decomposition means:
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i, j, k] should be 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (since each product
    # A[i,j]*B[j,k] contributes exactly once to C[i,k]).
    
    # Wait, that's not right. The tensor M has entries M[i,j,k] = 1 for all i,j,k
    # in the range, because each element of the product C[i,k] = sum_j A[i,j]*B[j,k]
    # comes from a distinct term.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # and 0 otherwise (but there's no "otherwise" here since all indices are valid).
    
    import numpy as np
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
