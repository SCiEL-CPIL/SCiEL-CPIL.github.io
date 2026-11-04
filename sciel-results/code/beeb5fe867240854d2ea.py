"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline with safety margin
    
    # Schoolbook as fallback
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
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a lower rank decomposition using random search
    # The tensor T[i,j,k] = delta (bilinear form)
    # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} for all i,j,k
    
    # Actually, the condition is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = 1 if this corresponds to the multiplication
    # Wait, let me re-read the convention.
    
    # The output factor W[k*n+i, r] suggests the output is indexed by (k, i) where
    # k is the column index of B (0..p-1) and i is the row index of A (0..n-1).
    # So C[i, k] = sum_j A[i,j] * B[j,k]
    # And the tensor entry T_{(i,j),(j,k),(k,i)} = 1 for the multiplication.
    
    # The decomposition condition: for all (i,j), (j,k), (k,i):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is valid, else 0
    # But actually for matrix multiplication, every (i,j,k) triple contributes, and
    # C[i,k] = sum_j A[i,j] B[j,k], so the tensor is T[i,j,k] = 1 for all i,j,k
    # (it's a 3-tensor with all entries 1 in the appropriate indexing).
    
    # Wait no. The matrix multiplication tensor T has T_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    # The bilinear form is: C[i,k] = sum_j T_{i,j,k} * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
    
    # So the condition for the decomposition is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let me verify with schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # for col corresponding to (i,j,k). Then the sum over r gives 1 for each (i,j,k). Correct.
    
    # So we need: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a system of n*m*p equations. We can try to find a sparse solution.
    
    # Random search: try various ranks, use random initialization and check
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W of given rank satisfy the decomposition."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True
    
    # Try ranks from 40 upward (since 72 is schoolbook, 40 is a reasonable lower bound)
    # Actually, the lower bound for 3x3 x 3x8 is at least 3*3*8/2 = 36 (from border rank considerations)
    # Let's try from 36 to 71
    
    for rank in range(36, min(72, 70)):
        if time.time() > deadline - 5:
            break
        
        # Try a few random attempts for this rank
        max_attempts = 50 if rank < 50 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/half-integers
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Fill with random values from {-1, 0, 1} or {-0.5, 0, 0.5, 1}
            for i in range(n * m):
                for r in range(rank):
                    if random.random() < 0.5:
                        U[i][r] = random.choice([-1, 1])
                    else:
                        U[i][r] = 0
            
            for i in range(m * p):
                for r in range(rank):
                    if random.random() < 0.5:
                        V[i][r] = random.choice([-1, 1])
                    else:
                        V[i][r] = 0
            
            for i in range(p * n):
                for r in range(rank):
                    if random.random() < 0.5:
                        W[i][r] = random.choice([-1, 1])
                    else:
                        W[i][r] = 0
            
            if check_decomposition(U, V, W, rank):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                return best_result
        
        # Also try a structured approach: use Strassen-like patterns
        if rank >= 23:
            # Try to use a known 3x3x3 rank-23 decomposition extended to 3x8
            # This is complex, so skip for now
            pass
    
    return best_result
# EVOLVE-BLOCK-END
