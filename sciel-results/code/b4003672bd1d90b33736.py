"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120  # 120s wall-clock limit

def _time_remaining():
    return DEADLINE - time.time()

def _verify_decomposition(U, V, W, n=2, m=5, p=6):
    """Verify that the tensor decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For random A (n×m) and B (m×p), C[i,k] = sum_j A[i,j]*B[j,k]
    # We check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if i,j,k form a valid triple, 0 otherwise
    # Actually, the tensor is: M[i,j,k] = delta[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition: C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # Wait, that's not right. The standard form is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # No, that's also not standard. Let me think again.
    
    # The tensor decomposition for matrix multiplication:
    # The multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we verify: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Let's verify this directly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6

    # Rank 54 decomposition using Strassen's algorithm for 2x2 blocks
    # A is 2x5, B is 5x6, C is 2x6
    # We partition the 'm' dimension (5) into pairs (0,1), (2,3) and single (4).
    # For each pair (j, j+1) in A and corresponding rows in B, we use Strassen (7 mults).
    # For the single column j=4 in A and row j=4 in B, we use schoolbook (4 mults per 2x2 k-block).
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # i_base: 0 (only one pair of rows in A/C since n=2)
        # j_base: 0 or 2 (pairs of cols in A / rows in B)
        # k_base: 0, 2, or 4 (pairs of cols in B / cols in C)
        
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        # Handles A[i_base..i_base+1, j_base] * B[j_base, k_base..k_base+1] -> C[i_base..i_base+1, k_base..k_base+1]
        # This is a 2x1 * 1x2 multiplication, requiring 4 scalar multiplications.
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Apply Strassen blocks for j-pairs (0,1) and (2,3)
    # k-pairs are (0,1), (2,3), (4,5)
    
    # j_pair 0,1
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # j_pair 2,3
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4
    # k-pairs are (0,1), (2,3), (4,5)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    # Trim if necessary
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        rank = col

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to naive if Strassen decomposition is incorrect
        rank = n * m * p  # 60
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] += 1
                    V[j * p + k][col] += 1
                    W[k * n + i][col] += 1
                    col += 1
        if col < rank:
            U = [row[:col] for row in U]
            V = [row[:col] for row in V]
            W = [row[:col] for row in W]
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
