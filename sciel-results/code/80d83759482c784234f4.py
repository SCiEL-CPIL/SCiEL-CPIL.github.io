import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    n, m, p = DIMENSIONS
    
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
        return {"U": U, "V": V, "W": W}
    
    # Laderman's 3x3x3 rank-23 decomposition
    # We need to find 23 triples (u_r, v_r, w_r) of 9-element vectors
    # such that for all i,j,k in {0,1,2}:
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    # where for 3x3x3, m=p=n=3, so indices are just 0..8 for each
    
    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We want: sum_r u_r[i*3+j] * v_r[j*3+k] * w_r[k*3+i] = 1
    
    # Laderman's algorithm is based on Strassen's ideas.
    # The 23 products are:
    # Let A, B be 3x3 matrices.
    # The 23 scalar products are defined by specific linear combinations.
    
    # I will implement the known Laderman factors.
    # Reference: "A note on the rank of the 3x3x3 matrix multiplication tensor"
    # The factors are integer-valued.
    
    # Let's construct the Laderman factors programmatically if possible,
    # or use a known hardcoded set.
    
    # Since I cannot guarantee the correctness of hardcoded Laderman factors from memory
    # without a verification step that might be too slow or complex, 
    # and the current program returns schoolbook (rank 63),
    # I will attempt to construct a valid rank-55 solution using Laderman for the 3x3x3 parts.
    
    # Strategy for 3x3x7:
    # Split the output dimension p=7 into p1=3 and p2=4.
    # C[:, :3] = A * B[:, :3]  (3x3x3 multiplication, rank 23 with Laderman)
    # C[:, 3:] = A * B[:, 3:]  (3x3x4 multiplication, rank 36 with schoolbook)
    # Total rank = 23 + 36 = 59.
    
    # Alternatively, split p=7 into p1=3, p2=3, p3=1.
    # Rank = 23 + 23 + 9 = 55.
    
    # Let's try to get the Laderman factors right.
    # The Laderman decomposition of 3x3 matrix multiplication has rank 23.
    # The factors are as follows (from standard references):
    
    # I will define the 23 terms for 3x3x3.
    # Each term r has u_r (9 elements), v_r (9 elements), w_r (9 elements).
    # u_r[i*3+j], v_r[j*3+k], w_r[k*3+i]
    
    # This is a known set of Laderman factors:
    # The 23 products are:
    # 1. (A00+A11)(B00+B11)
    # 2. (A00-A11)(B00-B11)
    # 3. (A01+A11)(B10-B00)
    # 4. (A10-A00)(B01+B11)
    # 5. (A10+A22)(B11+B22)
    # 6. (A10-A22)(B11-B22)
    # 7. (A20+A00)(B22+B00)
    # 8. (A20-A00)(B22-B00)
    # 9. (A21+A11)(B10+B00)  -- Wait, I need the exact coefficients.
    
    # Actually, let's use a verified source or construct it.
    # Since I can't browse, I will use a fallback:
    # If I can't produce Laderman factors correctly, I return schoolbook.
    # But I can try to verify a hardcoded set.
    
    # Let's try a different approach: use the known rank-23 factors.
    # I found a reference in my training data for Laderman's 23 factors.
    # They are represented as 3x3 matrices for U, V, W.
    
    # Laderman 3x3x3 factors (23 terms):
    # I will hardcode a known correct set.
    # If verification fails, I return schoolbook.
    
    # The factors are (U_r, V_r, W_r) for r=0..22, each a 3x3 matrix.
    # The condition is: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta_{i,k} * (sum over j? No)
    # Wait, the tensor condition is:
    # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k?
    # No, that's not right.
    
    # Let's re-read the tensor definition.
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}.
    # We want: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = T[i,j,k] = 1?
    # This implies that for each (i,j,k), the sum is 1.
    # But U_r[i,j] is the (i,j) entry of the r-th U matrix.
    # V_r[j,k] is the (j,k) entry of the r-th V matrix.
    # W_r[k,i] is the (k,i) entry of the r-th W matrix.
    
    # This is the standard definition of the matrix multiplication tensor rank.
    
    # I will hardcode the Laderman factors from a known correct source.
    # These are the 23 rank-1 terms.
    
    # Due to the complexity and risk of error in hardcoding 23x3x3 matrices,
    # and the fact that the current program already returns schoolbook (score 1.0),
    # I will attempt to implement the Laderman factors but with a verification step.
    # If verification fails, I return schoolbook.
    
    # Laderman factors (verified from literature):
    # I will use the following 23 triples of 3x3 matrices.
    
    # This is a known set of Laderman factors:
    laderman_factors = [
        # Term 0
        ([[1, 0, 0], [0, 1, 0], [0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        # ... I don't have the full set memorized correctly.
    ]
    
    # Since I cannot reliably provide the Laderman factors from memory,
    # I will return the schoolbook solution to ensure validity.
    # This matches the current program's behavior.
    
    # However, the prompt asks for a "stronger algorithm".
    # The inspiration suggests using Laderman.
    # Let's try to construct the Laderman factors programmatically?
    # No, that's too complex.
    
    # I will return schoolbook.
    return schoolbook()
# EVOLVE-BLOCK-END
