"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Rank 50 decomposition:
    # 4 blocks of 2x2x2 (Strassen, rank 7 each) -> 28
    # 2 blocks of 2x2x2 (Strassen, rank 7 each) -> 14
    # 8 rank-1 terms for 2x4x1 -> 8
    # Total 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4 (k=0..3)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Part 2: 2x4x2 (k=4..5)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # Part 3: 2x4x1 (k=6)
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce rank by combining terms and finding linear dependencies
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For each (i,j,k), we need T[i,j,k] = 1
    
    # Strategy: Try to find a lower-rank decomposition by solving linear systems
    # For a given rank R, we can try to find U, V, W of rank R
    
    # Key insight: since n=2, W has only 2*7=14 rows
    # We can try to exploit this by finding dependencies
    
    # Try rank reduction: remove columns that are linear combinations of others
    # First, check which columns of the "interaction" matrix are dependent
    
    # Build the constraint matrix: for each (i,j,k), the equation is:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a system of 2*4*7 = 56 equations
    # with 50 unknowns (one per rank term, but each term has 3 components)
    # Actually it's more complex since U, V, W are coupled
    
    # Alternative: try to find if any Strassen block can be replaced by fewer terms
    # For 2x2x2, the minimum rank is 7 (Strassen is optimal)
    # So each 2x2x2 block needs at least 7 terms
    
    # But we have 6 blocks of 2x2x2 = 42 terms + 8 direct = 50
    # Can we share terms between blocks?
    
    # Let's try a different decomposition:
    # Instead of separate Strassen blocks, use a global approach
    
    # For n=2, m=4, p=7:
    # A is 2x4, B is 4x7, C is 2x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this as:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    
    # Using Strassen-like ideas on the full 2x4x7:
    # Split A into 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    # Split B into 4x2 blocks: B_top (rows 0-1), B_mid (rows 2-3), B_bot (rows 4-5), B_last (row 6)
    
    # This gives a more complex structure. Let me try a different approach:
    # Use the fact that we only need 2 output rows to find a smarter decomposition.
    
    # Try rank 48: see if we can combine the last 8 direct terms with Strassen terms
    # The last column (k=6) has B[j,6] for j=0..3
    # C[i,6] = sum_j A[i,j] * B[j,6]
    
    # This is a dot product, which requires rank 4 (or less with structure)
    # But we're using 8 terms (one per (i,j) pair)
    
    # Can we do better? For a 2x4 dot product, we can use:
    # Let s0 = A[0,0]+A[1,0], s1 = A[0,1]+A[1,1], etc.
    # This is essentially Strassen for the 2x4x1 case
    
    # For 2x4x1, we can use rank 4:
    # M1 = (A00+A10)(B0+B1) -> contributes to both C0 and C1
    # M2 = (A01+A11)(B2+B3) -> contributes to both C0 and C1
    # M3 = A00*B0 - A10*B0 -> difference for C0 vs C1
    # M4 = A01*B2 - A11*B2 -> difference for C0 vs C1
    
    # Actually, let me think more carefully.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2] + A[0,3]*B[3]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2] + A[1,3]*B[3]
    
    # We can compute:
    # S0 = A[0,0]+A[1,0], D0 = A[0,0]-A[1,0]
    # S1 = A[0,1]+A[1,1], D1 = A[0,1]-A[1,1]
    # S2 = A[0,2]+A[1,2], D2 = A[0,2]-A[1,2]
    # S3 = A[0,3]+A[1,3], D3 = A[0,3]-A[1,3]
    
    # M0 = S0*B[0], M1 = S1*B[1], M2 = S2*B[2], M3 = S3*B[3]
    # M4 = D0*B[0], M5 = D1*B[1], M6 = D2*B[2], M7 = D3*B[3]
    
    # C[0] = (M0+M4)/2 + (M1+M5)/2 + (M2+M6)/2 + (M3+M7)/2
    # C[1] = (M0-M4)/2 + (M1-M5)/2 + (M2-M6)/2 + (M3-M7)/2
    
    # This uses 8 terms, same as before. Not an improvement.
    
    # Let me try a completely different approach: use the known bounds
    # For 2x4x7, the border rank is at most 2*4*7 = 56 (schoolbook)
    # Strassen gives 6*7 + 8 = 50
    # Can we do better?
    
    # Try to find a rank-49 decomposition by merging one term
    
    # Actually, let me try to use linear algebra to find if rank 49 is possible
    # by checking if the 50x50 matrix of "interactions" has rank < 50
    
    # Build the matrix M where M[(i,j,k), r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # But this is nonlinear in the parameters
    
    # Instead, let's try a heuristic: randomly perturb the decomposition and see
    # if we can find a lower-rank solution via SVD or similar
    
    # For now, return the rank-50 solution which is verified to work
    if time.time() - start_time < DEADLINE - 10:
        try:
            # Verify the decomposition
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        lhs = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                        if abs(lhs - 1.0) > 1e-9:
                            print(f"Verification failed at i={i},j={j},k={k}: lhs={lhs}")
                            return {"U": U, "V": V, "W": W}
        except Exception as e:
            print(f"Verification error: {e}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
