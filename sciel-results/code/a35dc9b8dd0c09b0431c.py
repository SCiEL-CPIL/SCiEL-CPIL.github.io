import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    
    # We split the multiplication into two parts:
    # 1. A (3x4) * B1 (4x4) -> C1 (3x4)
    # 2. A (3x4) * B2 (4x3) -> C2 (3x3)
    
    # Part 1: 3x4 * 4x4
    # Use Winograd rank 12 decomposition for the 4x4 part.
    # The tensor for 4x4x4 multiplication has rank 12.
    # We need U1 (4x4 x 12), V1 (4x4 x 12), W1 (4x4 x 12)
    # But the dimensions are slightly different.
    # U1 shape: (n*m, R1) = (12, 12)
    # V1 shape: (m*p1, R1) = (16, 12) where p1=4
    # W1 shape: (p1*n, R1) = (16, 12)
    
    # Part 2: 3x4 * 4x3
    # Use schoolbook decomposition.
    # Rank R2 = 3 * 4 * 3 = 36
    # U2 shape: (n*m, R2) = (12, 36)
    # V2 shape: (m*p2, R2) = (12, 36) where p2=3
    # W2 shape: (p2*n, R2) = (9, 36)
    
    # Total Rank R = 12 + 36 = 48
    
    R1 = 12
    R2 = 36
    R = R1 + R2
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Initialize Winograd matrices for 4x4x4
    # Standard Winograd F4 transform
    # U_win (4x12), V_win (4x12), W_win (4x12)
    # Note: The standard Winograd decomposition for 4x4 is:
    # C = W^T (U A V)
    # Where U, V, W are 4x12 matrices.
    # The tensor decomposition is T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # Wait, the indices are i (row), j (col of A), k (col of B).
    # T_{i,j,k} corresponds to A_{i,j} B_{j,k}.
    # In the factorization C = sum_r (U A)_{i,r} (B V)_{r,k} W_{i,k,r}?
    # No, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The decomposition is C_{i,k} = sum_r W_{i,k,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k})?
    # This is not a standard tensor decomposition form.
    # The tensor decomposition form is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}.
    # For matrix multiplication, T_{i,j,k} = 1 if j is the connecting index.
    # Actually, the tensor is T_{i,j,k} = delta_{i,j} delta_{j,k}? No.
    # The tensor is defined by C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # T_{i,j,k} is 1 if the indices match the multiplication structure.
    # Actually, T_{i,j,k} = 1 for all i, j, k is not correct.
    # The tensor is T_{i,j,k} = 1 if we are multiplying A_{i,j} and B_{j,k}.
    # So T_{i,j,k} = 1 for all i, j, k in the range?
    # No, the tensor is a 3-way array where T[i,j,k] = 1 if j is the shared index.
    # But j is always the shared index.
    # So T[i,j,k] = 1 for all i, j, k?
    # No, the tensor is T[i,j,k] = delta_{i,j} delta_{j,k}? No.
    # The tensor is T[i,j,k] = 1 if the indices are valid.
    # Actually, the tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k?
    # No, the rank is the minimum R such that T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r].
    # For matrix multiplication, T[i,j,k] = 1 if j is the connecting index.
    # But j is the connecting index for all terms.
    # So T[i,j,k] = 1 for all i,j,k?
    # No, the tensor is T[i,j,k] = 1 if the indices are in the range.
    # Actually, the tensor is T[i,j,k] = 1 for all i,j,k in the valid range.
    # Wait, if T[i,j,k] = 1 for all i,j,k, then the rank is n*m*p.
    # This is the schoolbook rank.
    # The Winograd decomposition reduces this rank by exploiting structure.
    
    # I will use the known Winograd matrices for 4x4x4.
    # U_win, V_win, W_win are 4x12 matrices.
    # The decomposition is C_{i,k} = sum_r W_{i,k,r} * (U_win A)_{i,r} * (B V_win)_{r,k}?
    # No, the decomposition is C_{i,k} = sum_r W_{i,k,r} * (sum_j U_win_{i,j,r} A_{i,j}) * (sum_j V_win_{j,k,r} B_{j,k})?
    # This is not correct.
    
    # Let's use a simpler approach: just use the schoolbook decomposition for now, but with a smaller rank if possible.
    # Since I don't have the exact Winograd matrices memorized, I will use the schoolbook decomposition.
    # Rank = 84.
    # This is not an improvement.
    
    # Let's try to implement the Winograd decomposition for 4x4x4.
    # The Winograd algorithm for 4x4 matrix multiplication uses 12 multiplications.
    # The matrices are:
    # U = [[1,0,1,0,0,0,0,0,1,0,0,0],
    #      [0,1,0,0,1,0,0,0,0,1,0,0],
    #      [0,0,1,0,0,1,0,0,0,0,1,0],
    #      [0,0,0,1,0,0,1,0,0,0,0,1]]
    # This is not correct.
    
    # I will use the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
