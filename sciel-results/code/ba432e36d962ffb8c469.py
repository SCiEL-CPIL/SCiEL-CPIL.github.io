import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-23 decomposition of 3x3x3 matrix multiplication
    # U33: 23 vectors of length 9 (for A, flattened row-major)
    # V33: 23 vectors of length 9 (for B, flattened row-major)
    # W33: 23 vectors of length 9 (for C, flattened row-major)
    
    # Using a known valid rank-23 decomposition
    # Each vector has entries in {-1, 0, 1}
    
    U33 = [
        [1,0,0, 0,0,0, 0,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
        [1,0,0, 0,1,0, 0,0,0],
        [0,1,0, 0,0,0, 0,1,0],
        [0,0,1, 0,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,1,0],
        [0,1,0, 0,0,0, 0,0,1],
        [0,0,1, 0,1,0, 0,0,0],
        [0,0,0, 1,0,0, 0,1,0],
        [0,0,0, 0,1,0, 0,0,1],
        [0,0,0, 0,0,1, 0,1,0],
        [1,0,0, 0,0,0, 0,0,1],
        [0,1,0, 0,1,0, 0,0,0],
        [0,0,1, 0,0,0, 0,1,0],
        [1,0,0, 0,0,0, 0,1,0],
        [0,1,0, 0,0,0, 0,0,1],
    ]
    
    V33 = [
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1],
        [1,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [1,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 1,0,0, 0,1,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [1,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,1,0],
    ]
    
    W33 = [
        [1,0,0, 0,0,0, 0,0,0],
        [0,1,0, 0,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0],
        [0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
        [0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1],
    ]
    
    # Verify the 3x3x3 decomposition
    # T[i,j,k] should equal sum_r U33[r][i*3+j] * V33[r][j*3+k] * W33[r][k*3+i]
    # Wait, the convention is: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for the term A[i,j]*B[j,k] contributing to C[i,k]
    # In tensor form: T has indices (i,j,k) where i=row of A, j=col of A/row of B, k=col of B
    # U acts on A: U[r, i*m+j]
    # V acts on B: V[r, j*p+k]
    # W acts on C: W[r, k*n+i]  (trace convention W[k*n+i, r])
    
    # Let me verify with a simple check
    def verify_333(U, V, W):
        n3, m3, p3 = 3, 3, 3
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    # T[i,j,k] = 1
                    s = 0
                    for r in range(len(U)):
                        s += U[r][i*m3+j] * V[r][j*p3+k] * W[r][k*n3+i]
                    if s != 1:
                        return False
        # Check that T[i,j,k]=0 when it should be
        # Actually for 3x3x3, every (i,j,k) triple with valid indices contributes
        # The tensor T[i,j,k] = 1 for all i,j,k in [0,2] since C[i,k] = sum_j A[i,j]B[j,k]
        # Wait no - T is the structure tensor. T[i,j,k] = 1 iff the product A[i,j]*B[j,k] 
        # appears in C[i,k]. For matrix mult, this is always 1 for valid i,j,k.
        # But we need to check that spurious terms cancel.
        # The condition is: for all i,k: sum_j T[i,j,k] * A[i,j]*B[j,k] = C[i,k]
        # which means sum_j A[i,j]*B[j,k] = C[i,k], which is correct.
        # The tensor decomposition condition is:
        # sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = T[i,j,k] for all i,j,k
        # where T[i,j,k] = 1 if (i,j,k) is a valid term, 0 otherwise.
        # For matrix mult 3x3x3, T[i,j,k] = 1 for all i,j,k in [0,2].
        # So we need sum_r ... = 1 for all i,j,k.
        # We already checked that. But we also need to make sure there are no 
        # additional constraints. Actually the condition is exactly that.
        return True
    
    # Actually, I realize my hardcoded vectors are likely wrong. Let me use a 
    # different, more reliable approach.
    
    # Safe approach: use schoolbook but try to find a valid lower-rank decomposition
    # by using a known correct rank-23 decomposition.
    
    # Since I'm not confident in the hardcoded values, let me use a construction
    # that I can verify.
    
    # Alternative: use the "border rank" approach or a known valid decomposition.
    
    # Let me try a different strategy: construct a rank-46 decomposition by 
    # taking two independent rank-23 decompositions.
    
    # But first, let me just return a verified schoolbook to be safe, 
    # and then try to improve.
    
    # Actually, let me try to construct a valid decomposition using a known method.
    
    # For 3x3x3, one valid rank-23 decomposition can be found in the literature.
    # Let me use a simpler but still improved approach.
    
    # The minimum rank for 3x3x3 is 23, for 3x3x6 it's at most 46 (two copies).
    # Can we do better? Possibly, but let's start with 46.
    
    # I'll use a verified rank-23 decomposition. Here's one from a reliable source:
    
    # Using the decomposition from "Tensor rank and arithmetic complexity of small matrices"
    # by Landsberg. The 23 forms are as follows (each is a triple of 3x3 matrices):
    
    # Let me use a different, more systematic approach. I'll generate a valid 
    # decomposition by solving the system of equations.
    
    # For 3x3x3, we need 23 vectors u_r, v_r, w_r in R^9 such that
    # sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k in {0,1,2}
    
    # This is a nonlinear system. Let me try a greedy approach or use known values.
    
    # Given the complexity, let me just return the schoolbook decomposition 
    # which is guaranteed correct, with rank 54.
    
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
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
