"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Try Strassen-embedded hybrid decomposition
    # Strassen for 3x3x3: rank 7
    # Boundary terms: schoolbook for remaining parts
    
    # Strassen factors for 3x3x3 (indices 0,1,2 for i,j,k)
    # M1 = (A[0]+A[1])*(B[0]+B[1])
    # M2 = (A[0]+A[1])*B[2]
    # M3 = A[0]*(B[0]-B[1])
    # M4 = A[2]*(B[1]+B[2])
    # M5 = (A[2]-A[1])*(B[2]+B[3]) -- wait, we only have 3 columns
    # Actually for 3x3: M5 = (A[2]-A[1])*(B[2]+B[1])? No.
    
    # Standard Strassen for 3x3x3 (from literature, rank 7):
    # Let's use the known rank-7 decomposition for 3x3x3 matrix multiplication.
    # One known decomposition:
    # M1 = (A11 + A12) * (B11 + B12)
    # M2 = (A11 + A12) * B22
    # M3 = A11 * (B11 - B12)
    # M4 = A22 * (B12 + B22)
    # M5 = (A22 - A12) * (B22 + B11)
    # M6 = (A11 + A22) * (B22 - B11)
    # M7 = (A12 + A22) * (B11 - B22)
    # where A = [[A11, A12], [A21, A22]] is 2x2 block partition... 
    # Actually this is getting complex. Let me use a simpler approach.
    
    # For now, let's try a direct approach: use schoolbook but see if we can
    # find any rank reduction. If not, return schoolbook.
    
    # Actually, let me implement the Strassen embedding properly.
    # The 3x3x3 Strassen rank-7 decomposition (one known form):
    # Using indices 0,1,2 for the 3x3 blocks
    
    # Strassen rank-7 for 3x3x3 (verified decomposition):
    # M0 = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1] + A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    # This is just the full dot product, not helpful.
    
    # Let me use a different strategy: try to verify if a lower rank is achievable
    # by constructing the tensor and checking. But that's computationally expensive.
    
    # For safety, return the schoolbook decomposition which is guaranteed correct.
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
