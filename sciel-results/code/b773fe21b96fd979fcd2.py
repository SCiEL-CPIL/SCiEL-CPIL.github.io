"""Matrix multiplication 3x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook decomposition (rank 72)
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try to find better decompositions using known algorithms
    # Strategy 1: Use Strassen for 2x2x2 blocks
    # For 3x3x8, we can partition A (3x3) and B (3x8)
    
    # Let's try a systematic approach:
    # The tensor T(i,j,k) = 1 if i,j,k are valid indices
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = delta(i,j,k)
    
    # Actually, the decomposition is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    # So for each r, we have three vectors:
    # u_r: (n*m)-vector, v_r: (m*p)-vector, w_r: (p*n)-vector
    # Such that sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = delta(i,j,k) for all valid i,j,k
    # And the sum is 0 for invalid combinations (but all i,j,k are valid here)
    
    # Wait, the tensor is T(i,j,k) = 1 for all valid i,j,k.
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a rank-1 decomposition of the all-ones tensor of shape (n*m, m*p, p*n)?
    # No, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor T has entries T(i,j,k) where:
    # - i indexes rows of A (0 to n-1)
    # - j indexes columns of A / rows of B (0 to m-1)  
    # - k indexes columns of B (0 to p-1)
    # T(i,j,k) = 1
    
    # The factorization is: T(i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need to decompose the all-ones tensor of shape (9, 24, 24) into rank-1 tensors.
    # But wait, that doesn't seem right either. Let me re-read the problem.
    
    # Actually, I think the standard formulation is:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # If we write A as sum_r (sum_j U_{ij}^{(r)} A_{ij}) and B similarly...
    
    # No, the correct formulation for tensor decomposition of matrix multiplication is:
    # The bilinear form sum_j A[i,j] B[j,k] can be written as sum_r f_r(A) g_r(B) h_r
    # where f_r(A) = sum_{i,j} U[i*m+j, r] A[i,j]
    # g_r(B) = sum_{j,k} V[j*p+k, r] B[j,k]
    # h_r = W[k*n+i, r]... but h_r depends on i and k?
    
    # I think I'm confusing myself. Let me look at this differently.
    
    # The standard way: we want to compute C = A * B where A is n x m and B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # A rank-r algorithm computes C as:
    # C[i,k] = sum_{r=1}^{R} alpha_r(i,k) * (sum_{i,j} U[i*m+j, r] A[i,j]) * (sum_{j,k} V[j*p+k, r] B[j,k])
    
    # where alpha_r(i,k) = W[k*n+i, r]
    
    # So the condition is: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i,j,k form a valid "chain"
    # In our case, all i in [0,n), j in [0,m), k in [0,p) are valid, so the RHS is 1.
    
    # Wait, that's not right. The bilinear form is sum_j A[i,j] B[j,k], which is a single term.
    # The decomposition is: sum_j A[i,j] B[j,k] = sum_r (sum_{i',j'} U[i'*m+j', r] A[i',j']) * (sum_{j',k'} V[j'*p+k', r] B[j',k']) * W[k*n+i, r]
    
    # This must hold for all i, k. So expanding the left side:
    # sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] A[i',j']) * (sum_{j',k'} V[j'*p+k', r] B[j',k'])
    
    # For this to equal the left side for all A and B, we need:
    # The coefficient of A[i',j'] B[j'',k'] on the right must match the coefficient on the left.
    # On the left, the coefficient of A[i',j'] B[j'',k'] is 1 if j' = j'' and (i',k') = (i,k), else 0.
    # But this is getting complicated. Let me just use a known result.
    
    # For 2x2x2 matrix multiplication, Strassen's algorithm achieves rank 7.
    # The factors are well-known. Let me implement Strassen for the 2x2 case first.
    
    # Actually, let me try a different approach: use the known Strassen algorithm structure
    # and extend it to our dimensions.
    
    # For 2x2 matrix multiplication (n=m=p=2), Strassen uses 7 multiplications:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our problem (3x3 x 3x8), we can try to apply Strassen to 2x2 blocks.
    # Partition A into blocks: A[0:2,0:2], A[0:2,2:3], A[2:3,0:2], A[2:3,2:3]
    # Partition B into blocks: B[0:2,0:8], B[2:3,0:8]
    
    # The product C = A*B has blocks:
    # C[0:2,0:8] = A[0:2,0:2]*B[0:2,0:8] + A[0:2,2:3]*B[2:3,0:8]
    # C[2:3,0:8] = A[2:3,0:2]*B[0:2,0:8] + A[2:3,2:3]*B[2:3,0:8]
    
    # We can apply Strassen to the 2x2 * 2x8 part, but Strassen is for 2x2 * 2x2.
    # For 2x2 * 2x8, we can view B as 4 columns of 2x2, and apply Strassen to each.
    # But that gives 7*4 = 28 multiplications for the 2x2*2x8 part.
    # Schoolbook would be 4*4 = 16 for 2x2*2x2 per column, so 16*4 = 64.
    # Wait, 2x2 * 2x2 schoolbook is 4 mults. 2x2 * 2x8 is 2x2 times 4 columns of 2x2, so 4*4=16 mults.
    # Strassen for 2x2*2x2 is 7 mults, so for 4 columns: 7*4=28 mults.
    # So Strassen is worse for 2x2*2x8! We should use schoolbook there.
    
    # Let me reconsider. The schoolbook rank for 3x3*3x8 is 3*3*8 = 72.
    # Can we do better?
    
    # One approach: use the fact that R(3,3,3) <= 23 (known result).
    # But I don't have the explicit factors for rank 23.
    
    # Let me try a heuristic search for small ranks.
    
    def validate(U, V, W):
        """Validate that the decomposition is correct."""
        n, m, p = DIMENSIONS
        R = len(U[0])
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def try_random_search(max_rank, iterations, deadline):
        """Try random search for a valid decomposition."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        random.seed(42)
        
        for target_rank in range(30, max_rank + 1):
            if time.time() > deadline:
                break
            for _ in range(iterations):
                if time.time() > deadline:
                    break
                # Generate random sparse factors with small integer entries
                U = [[0] * target_rank for _ in range(nm)]
                V = [[0] * target_rank for _ in range(mp)]
                W = [[0] * target_rank for _ in range(pn)]
                
                for r in range(target_rank):
                    # Each rank-1 component should have some structure
                    # Randomly choose non-zero entries
                    for _ in range(random.randint(1, 4)):
                        idx = random.randint(0, nm - 1)
                        U[idx][r] = random.choice([-1, 1])
                    for _ in range(random.randint(1, 4)):
                        idx = random.randint(0, mp - 1)
                        V[idx][r] = random.choice([-1, 1])
                    for _ in range(random.randint(1, 4)):
                        idx = random.randint(0, pn - 1)
                        W[idx][r] = random.choice([-1, 1])
                
                if validate(U, V, W):
                    return {"U": U, "V": V, "W": W}, target_rank
        
        return None, max_rank + 1
    
    # Try random search with a time limit
    if time.time() < deadline:
        result, rank = try_random_search(60, 1000, deadline)
        if result is not None and rank < best_rank:
            best_result = result
            best_rank = rank
    
    # If we found a better decomposition, return it
    if best_rank < schoolbook_rank:
        return best_result
    
    # Otherwise, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
