"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    schoolbook_rank = n * m * p  # 120
    
    def verify_decomp(U, V, W):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Build schoolbook decomposition as baseline
    U_school = [[0.0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0.0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0.0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Strategy: Use block structure to build a potentially lower-rank decomposition
    # Partition 4x5 = [4x4 | 4x1] and 5x6 = [[4x4],[4x2]]
    # This gives 4 sub-multiplications:
    # C[0:4,0:4] = A[0:4,0:4]*B[0:4,0:4] + A[0:4,4]*B[4,0:4]
    # C[0:4,4:6] = A[0:4,0:4]*B[0:4,4:6] + A[0:4,4]*B[4,4:6]
    # C[4:4,0:4] = A[4,0:4]*B[0:4,0:4]  (but n=4, so no row 4)
    # Actually n=4, so rows are 0-3. Let me redo.
    
    # n=4 rows, m=5 cols for A; m=5 rows, p=6 cols for B
    # Partition A as [A1 (4x4) | A2 (4x1)] and B as [[B1 (4x4)],[B2 (1x4)]]... 
    # Wait, B is 5x6. Partition B as [[B11 (4x4)],[B12 (4x2)]] and [[B21 (1x4)],[B22 (1x2)]]
    
    # C = A*B where A is 4x5, B is 5x6
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [[B1],[B2]] where B1 is 4x6, B2 is 1x6
    # C = A1*B1 + A2*B2
    
    # A1*B1: 4x4 * 4x6 -> 4x6, rank 24 schoolbook, but can use better
    # A2*B2: 4x1 * 1x6 -> 4x6, rank 24 schoolbook (outer product structure)
    
    # For A2*B2: this is rank 1 in the middle dimension, so it's an outer product
    # A2*B2[i,k] = sum_j A2[i,j]*B2[j,k] = A2[i,0]*B2[0,k] (since j only 0)
    # This can be done with rank 4*6=24 terms, but actually it's just 24 scalar multiplications
    # No savings here since it's already minimal for this structure
    
    # For A1*B1: 4x4 * 4x6
    # We can further partition: 4x4 * 4x6 = 4x4*4x4 + 4x4*4x2
    # 4x4*4x4: known rank 21 (optimal)
    # 4x4*4x2: rank 32 schoolbook, but can we do better?
    
    # Let's use Strassen-like for 4x4*4x4 (rank 21) and schoolbook for the rest
    # Total: 21 + 32 + 24 = 77? That's already much better than 120!
    
    # But I need to construct the actual decomposition. Let me use a simpler approach:
    # Just use schoolbook and try greedy elimination.
    
    # Greedy elimination: try to remove terms
    def try_eliminate(U, V, W, max_rounds=5):
        """Try to eliminate redundant terms"""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        eliminated = 0
        
        for _round in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            found = False
            for r in range(rank):
                if time.time() > deadline - 5:
                    break
                
                # Check if column r is needed
                # Zero it out and verify
                U_save = [row[:] for row in U]
                V_save = [row[:] for row in V]
                W_save = [row[:] for row in W]
                
                for row in U:
                    row[r] = 0.0
                for row in V:
                    row[r] = 0.0
                for row in W:
                    row[r] = 0.0
                
                if verify_decomp(U, V, W):
                    # Remove column r
                    U = [row[:r] + row[r+1:] for row in U]
                    V = [row[:r] + row[r+1:] for row in V]
                    W = [row[:r] + row[r+1:] for row in W]
                    rank -= 1
                    eliminated += 1
                    found = True
                    break
                else:
                    U, V, W = U_save, V_save, W_save
            
            if not found:
                break
        
        return U, V, W, eliminated
    
    # Try elimination on schoolbook
    U_elim, V_elim, W_elim, elim_count = try_eliminate(U_school, V_school, W_school)
    
    if elim_count > 0:
        best_U, best_V, best_W = U_elim, V_elim, W_elim
        best_rank = len(U_elim[0])
    
    # Try a structured decomposition using block multiplication
    # 4x5 * 5x6: partition into 2x2 blocks of sizes (2,2)x(2,3) for A and (2,3)x(3,3) for B
    # Actually, let's try: A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    # B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x4, B21 is 3x2, B22 is 3x4
    # C = [A11*B11 + A12*B21, A11*B12 + A12*B22; A21*B11 + A22*B21, A21*B12 + A22*B22]
    
    # This is getting complex. Let me just use the elimination result.
    
    # One more attempt: use the fact that for rectangular matrices, we can sometimes
    # use rank (n+m+p-1)^3 / 8 or similar bounds, but construction is hard.
    
    # Return best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
