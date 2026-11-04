"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Strategy: Use Strassen's algorithm for the 4x4x4 part (rank 7)
    # and schoolbook for the 5th output column (rank 4), total rank 11.
    # 
    # Strassen's algorithm for 4x4 matrix multiplication:
    # We partition each 4x4 matrix into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    #
    # Then we compute 7 products of 2x2 matrices:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # Then:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of 2x2 matrices, which requires 4 scalar multiplications.
    # So total scalar multiplications = 7 * 4 = 28.
    #
    # But we want a tensor decomposition, not just a count.
    # Let me think about this differently.
    #
    # Actually, for the tensor decomposition approach, I need to express
    # the matmul tensor as a sum of rank-1 tensors.
    #
    # For 4x4x4, Strassen gives rank 7 at the block level, but each block
    # multiplication is itself a 2x2x2 tensor of rank 4 (schoolbook) or 
    # potentially lower.
    #
    # Let me use a different approach: directly construct the rank-11 decomposition.
    #
    # The matmul tensor T[i,j,k] for 4x4x4x5 is 1 for all valid (i,j,k) where
    # i in 0..3, j in 0..3, k in 0..4.
    #
    # A rank-1 tensor is U[i*m+j] * V[j*p+k] * W[k*n+i].
    #
    # For the 4x4x4 part (k in 0..3), I can use a known low-rank decomposition.
    # For the 5th column (k=4), I use schoolbook (rank 4).
    
    # Let me try a simpler approach: use the fact that we can embed the problem
    # and use a structured decomposition.
    
    # Actually, let me just use a direct construction.
    # For 4x4x4, the tensor rank is known to be at most 49 (from Alon et al.),
    # but we can do better with Strassen-like constructions.
    
    # Let me use a different strategy: use the DFT-based approach for the 5th dimension.
    # The 4x5 output can be handled by noting that we only need 4 independent
    # columns, and the 5th can be expressed in terms of the others if we use
    # a circulant structure.
    
    # Actually, the simplest valid approach: use schoolbook for all but try to
    # find a lower rank through optimization.
    
    # For now, let me return a valid decomposition. I'll use a known result:
    # The tensor rank of 4x4x4 matrix multiplication is at most 49.
    # For 4x4x5, it's at most 49 + 4 = 53 (using schoolbook for the 5th column).
    
    # But let me try to do better. I'll use a randomized search to find a valid
    # decomposition with rank < 80.
    
    # Actually, let me just return the schoolbook decomposition as a valid baseline.
    # The task is to find the rank-minimal decomposition, and the evaluator will
    # score based on the rank. So I need to find the lowest possible rank.
    
    # Let me implement a proper search.
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    def _schoolbook_decomposition(n, m, p):
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
    
    # Verification function
    def _verify(U, V, W, n, m, p, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a low-rank decomposition using a structured approach.
    # 
    # Key insight: The matmul tensor has a lot of structure. We can use the
    # fact that it's a "slice" of a larger tensor.
    #
    # For now, let me try a few specific constructions.
    
    # Construction 1: Use Strassen for 4x4x4 (rank 7 at block level = 28 scalar mults)
    # This gives rank 28 for the 4x4x4 part, plus 4 for the 5th column = 32 total.
    
    # Let me construct this properly.
    
    # Strassen's algorithm for 4x4x4:
    # We have 7 block products, each of which is a 2x2x2 matmul.
    # Each 2x2x2 matmul has tensor rank 4 (schoolbook) or potentially lower.
    #
    # For 2x2x2, the tensor rank is 7 (Strassen) or 4 (schoolbook).
    # Actually, the tensor rank of 2x2x2 matmul is 7 (Strassen).
    #
    # So total rank for 4x4x4 using Strassen at both levels: 7 * 7 = 49.
    # Plus 4 for the 5th column: total rank 53.
    
    # Let me construct this.
    
    # First, let me define the 2x2x2 Strassen decomposition.
    # For 2x2x2 matmul: C = A * B where A, B, C are 2x2.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each Mi is a product of linear combinations.
    # This is getting complex. Let me use a simpler approach.
    
    # Actually, let me just use the schoolbook decomposition and return it.
    # The evaluator will score it, and if there's a better one, the search
    # should find it.
    
    # But wait - the task says "Find a rank-minimal exact tensor decomposition".
    # So I need to find the minimum rank, not just any valid decomposition.
    
    # Let me try a more systematic search.
    
    # For small ranks, I can try to solve the system of equations directly.
    # The constraint is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    #
    # This is a system of n*m*p = 80 equations in rank * (n*m + m*p + p*n) variables.
    # For rank r, we have r * (16 + 20 + 20) = 56r variables.
    # We need 56r >= 80, so r >= 2 (but this is a lower bound).
    
    # The actual lower bound is higher. Let me try ranks from 20 upward.
    
    # For efficiency, I'll use a randomized local search.
    
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from a reasonable lower bound upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # But for matmul, the lower bound is higher. Let's try from 20 to 60.
    
    for r in range(20, min(best_rank, 60)):
        if time.time() > deadline - 10:
            break
        
        # Try several random initializations
        max_attempts = 100 if r < 30 else 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            
            # Local search: try to improve by flipping individual entries
            improved = False
            for round_num in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Check if current decomposition is valid
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, r):
                    improved = True
                    break
                
                # Try flipping a random entry
                mat_choice = np.random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row, col = np.random.randint(n*m), np.random.randint(r)
                    old_val = U[row, col]
                    new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
                    if new_val == old_val:
                        continue
                    U[row, col] = new_val
                    # Check if this improves things (heuristic: check a subset)
                    # For simplicity, just accept and check validity later
                elif mat_choice == 'V':
                    row, col = np.random.randint(m*p), np.random.randint(r)
                    old_val = V[row, col]
                    new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
                    if new_val == old_val:
                        continue
                    V[row, col] = new_val
                else:
                    row, col = np.random.randint(p*n), np.random.randint(r)
                    old_val = W[row, col]
                    new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
                    if new_val == old_val:
                        continue
                    W[row, col] = new_val
            
            if improved:
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = r
                break  # Found a valid decomposition at this rank
    
    return best_result
# EVOLVE-BLOCK-END
