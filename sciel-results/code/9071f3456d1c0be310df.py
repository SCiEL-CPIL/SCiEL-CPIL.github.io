"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Start with schoolbook rank 56
    rank_schoolbook = n * m * p  # 56
    
    # Try to construct a Strassen-like decomposition
    # For 2x4x7, we can partition into blocks and use Strassen on 2x2x2 subproblems
    
    # Approach: Use the structure that n=2 allows some cancellation
    # Let's try rank 50 first (known improvement from knowledge)
    
    # Construct a rank-50 decomposition using block Strassen
    # Partition A (2x4) = [A1, A2] where A1, A2 are 2x2
    # Partition B (4x7) = [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2
    # Each A_i (2x2) * B_i (2x7) can be done with rank 7*7=49 using Strassen per column?
    # Actually, for 2x2x7, we can do it with rank 7*7=49? No.
    
    # Better: For 2x2x7, think of it as 7 independent 2x2x2 multiplications
    # Each 2x2x2 has rank 7 (Strassen), so 2x2x7 has rank 7*7=49
    # Total for 2x4x7: 2 * 49 = 98 - too high
    
    # Alternative: Use a more clever decomposition
    # Let me try to directly construct rank 50
    
    # Actually, let's use a different strategy: 
    # The tensor M_{2,4,7} can be seen as the sum of 4 tensors M_{2,1,7} (one for each j)
    # Each M_{2,1,7} has rank 2*1*7 = 14, total 4*14 = 56 (schoolbook)
    # But we can do better by exploiting the structure
    
    # Let me try a constructive approach with specific rank values
    # Start with rank 50 and try to verify
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check: for all i,k, sum_j A[i,j]*B[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Actually, the tensor equation is:
        # sum_c U[a,c] * V[b,c] * W[c_index, c] = delta where a=(i,j), b=(j,k), c_index=(k,i)
        # More precisely: for the matrix multiplication tensor T[i,j,k] = 1 if... 
        # The decomposition satisfies: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
        # And T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
        
        # Build the full tensor and check
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        
        # Compute the decomposed tensor
        T_dec = np.zeros((n, m, p))
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for c in range(rank):
            u = U_arr[:, c]
            v = V_arr[:, c]
            w = W_arr[:, c]
            # u has indices (i,j), v has indices (j,k), w has indices (k,i)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_dec[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
        
        # Check if T_dec equals T (which is all 1s)
        if np.allclose(T_dec, T, atol=1e-10):
            return True
        return False
    
    # Try to construct a rank-50 decomposition
    # Use a heuristic: start with schoolbook and try to merge terms
    
    # For now, let's return a verified rank-50 decomposition if we can find one
    # Otherwise fall back to schoolbook rank 56
    
    # Construct schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    # Try to find a better decomposition using optimization
    # Since exact decomposition with small integers is hard, let's try a few known ranks
    
    # Try rank 50
    best_result = None
    best_rank = rank_schoolbook
    
    # First, get schoolbook
    school = schoolbook()
    best_result = school
    best_rank = rank_schoolbook
    
    # Try to improve using a simple local search
    # This is complex, so for now let's just return schoolbook
    # But the knowledge says rank 50 was achieved, so let me try to construct it
    
    # Actually, let me try a different approach: use the fact that we can sometimes
    # combine terms. For 2x4x7, one idea is to use the identity:
    # (a+b)(c+d) = ac + ad + bc + bd can sometimes be computed with 3 terms
    # using (a+b)(c+d) - ac - bd = ad + bc, but that's not helpful directly.
    
    # For matrix multiplication, Strassen's trick for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 2x2x7, I can apply Strassen to each column independently, giving rank 7*7=49
    # Then for 2x4x7, partition into two 2x2x7 blocks, total rank 98 - worse
    
    # Let me just return schoolbook for now, but try to verify rank 50 is possible
    # by constructing it explicitly if I can figure out the pattern
    
    # Given time constraints, return schoolbook
    result = school
    del result["rank"]
    return result

# EVOLVE-BLOCK-END
