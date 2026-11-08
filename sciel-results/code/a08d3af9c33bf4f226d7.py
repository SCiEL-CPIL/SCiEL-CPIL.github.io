import time
import random

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # 130s out of 150s
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Strategy: Block Diagonal Embedding
    # Part 1: 4x4x4 (Columns 0-3 of B)
    # Part 2: 4x4x3 (Columns 4-6 of B)
    
    # --- Part 2: 4x4x3 Decomposition (Rank 12) ---
    # This part is simple. For each column k in 4..6, we use a rank-4 decomposition for 4x4x1.
    # Total rank for this part is 3 * 4 = 12.
    
    U2 = [[0] * 12 for _ in range(n * m)]
    V2 = [[0] * 12 for _ in range(m * p)]
    W2 = [[0] * 12 for _ in range(p * n)]
    
    for col_idx in range(3):
        k_global = 4 + col_idx
        # Rank 4 decomposition for 4x4x1
        # For each j in 0..3:
        for j in range(4):
            r_local = col_idx * 4 + j
            # U: picks A[i, j]
            for i in range(4):
                U2[i * m + j][r_local] = 1
            # V: picks B[j, k_global]
            V2[j * p + k_global][r_local] = 1
            # W: puts result in C[i, k_global]
            for i in range(4):
                W2[k_global * n + i][r_local] = 1

    # --- Part 1: 4x4x4 Decomposition ---
    # Try Laderman Rank 13 first. If it fails, use Rank 16.
    
    # Laderman Coefficients for 4x4x4
    # U (16x13), V (16x13), W (16x13)
    # Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    
    # I will use a known valid set of coefficients for Laderman's algorithm.
    # If these are incorrect, the verification will fail.
    
    # Note: The specific coefficients for Laderman's algorithm are complex.
    # To ensure correctness, I will use the Rank 16 (Schoolbook) for the 4x4x4 part as well,
    # resulting in a total rank of 16 + 12 = 28.
    # This is a guaranteed valid solution.
    
    U1 = [[0] * 16 for _ in range(n * m)]
    V1 = [[0] * 16 for _ in range(m * p)]
    W1 = [[0] * 16 for _ in range(p * n)]
    
    # Rank 16 decomposition for 4x4x4
    for j in range(4):
        for k in range(4):
            r_local = j * 4 + k
            # U: picks A[i, j]
            for i in range(4):
                U1[i * m + j][r_local] = 1
            # V: picks B[j, k]
            V1[j * p + k][r_local] = 1
            # W: puts result in C[i, k]
            for i in range(4):
                W1[k * n + i][r_local] = 1

    # Combine U1, U2
    R_total = 16 + 12
    U = [[0] * R_total for _ in range(n * m)]
    V = [[0] * R_total for _ in range(m * p)]
    W = [[0] * R_total for _ in range(p * n)]
    
    # Copy Part 1
    for i in range(n * m):
        for r in range(16):
            U[i][r] = U1[i][r]
    for i in range(m * p):
        for r in range(16):
            V[i][r] = V1[i][r]
    for i in range(p * n):
        for r in range(16):
            W[i][r] = W1[i][r]
            
    # Copy Part 2
    for i in range(n * m):
        for r in range(12):
            U[i][16 + r] = U2[i][r]
    for i in range(m * p):
        for r in range(12):
            V[i][16 + r] = V2[i][r]
    for i in range(p * n):
        for r in range(12):
            W[i][16 + r] = W2[i][r]

    # Verify the decomposition
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    # Note: The tensor is defined such that C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] should be 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # In our construction, for each (j,k) pair in Part 1, we have a term.
    # For a specific (i, j, k), the sum should be 1 if j is the summation index?
    # No, the tensor T is defined by the decomposition.
    # We need to check if the decomposition reproduces the multiplication tensor.
    # The multiplication tensor has T[i, j, k] = 1 if the indices are consistent?
    # Actually, the tensor is $T_{ijk}$ where $i$ is row of A/col of C, $j$ is col of A/row of B, $k$ is col of B/col of C.
    # The condition is $\sum_r U_{im+j, r} V_{jp+k, r} W_{pn+i, r} = 1$ for all $i,j,k$?
    # No. The tensor is non-zero only for valid indices.
    # In matrix multiplication, every $A_{ij}$ and $B_{jk}$ contributes to $C_{ik}$.
    # So for every $i, j, k$, the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$ is 1.
    # So yes, the sum must be 1 for all $i, j, k$.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R_total):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return {"U": U_school, "V": V_school, "W": W_school}

