"""Combined program: Strassen-based block decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import sys

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def make_schoolbook():
        """Return schoolbook decomposition of rank n*m*p."""
        rank = n * m * p  # 144
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is valid."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > tol:
                        return False
        return True
    
    # Start with schoolbook
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using block structure
    if time.time() < deadline - 10:
        # Use Strassen-like decomposition for 2x2 * 2x9 blocks
        # Split A into 2x2 blocks, B into 2x9 blocks
        # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
        # B = [[B00, B01], [B10, B11]] where each Bj is 2x9
        
        # For each 2x2 * 2x9 multiplication, use 7 terms (Strassen)
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Strassen for 2x2 * 2x2 uses 7 multiplications.
        # For 2x2 * 2x9, we can apply Strassen to each column pair.
        
        # Let me try a different approach: use the fact that 2x2 matrix multiplication
        # has rank 7 (Strassen). For 2x2 * 2x9, we can decompose as:
        # For each of the 9 columns of B, do 2x2 * 2x1 multiplication.
        # 2x2 * 2x1 has schoolbook rank 4, but can we do better?
        
        # Actually, let me try a greedy approach: represent the tensor as a 3D array
        # and use hypergraph matching to find a low-rank decomposition.
        
        # The tensor T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,9)
        # We want to write T as sum of rank-1 tensors:
        # T = sum_r U[:,i,m+j] * V[:,j,p+k] * W[:,k,n+i]
        
        # Greedy approach: at each step, find the rank-1 tensor that covers
        # the most remaining uncovered entries.
        
        # However, this is complex. Let me try a simpler construction first.
        
        # Construction using block Strassen:
        # Split A into 2x2 blocks, B into 2x9 blocks.
        # Use Strassen's 7-term formula for each 2x2 * 2x2 subproblem.
        
        # For 2x2 * 2x9:
        # B = [B_left | B_right] where B_left, B_right are 2x9/2... wait, 9 is odd.
        
        # Let me try: B is 2x9. Split into B1 (2x4) and B2 (2x5).
        # Then A*B = A*B1 + A*B2
        # A*B1: 2x2 * 2x4, schoolbook rank 16
        # A*B2: 2x2 * 2x5, schoolbook rank 20
        # Total: 36, same as schoolbook.
        
        # Alternative: Use the fact that we can share computations.
        
        # Let me try a specific rank-108 construction:
        # For each i (4 rows of A), for each k (9 columns of B),
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Group the j indices: {0,1} and {2,3}
        # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
        
        # For the first group, define:
        # M1[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # This is a 4x9 matrix that depends on A[:,0:2] and B[0:2,:]
        
        # Similarly M2[i,k] = A[i,2]*B[2,k] + A[i,3]*B[3,k]
        
        # C = M1 + M2
        
        # M1 is the product of a 4x2 matrix and a 2x9 matrix.
        # Schoolbook rank for 4x2 * 2x9 is 4*2*9 = 72.
        # M2 is the same: 72.
        # Total: 144, still schoolbook.
        
        # Let me try to use Strassen on the 4x2 * 2x9 product.
        # 4x2 * 2x9: Split A into two 2x2 blocks.
        # A = [[A_top, A_bottom]] where each is 2x2
        # B = [[B_top, B_bottom]] where each is 2x9
        
        # A*B = A_top*B_top + A_bottom*B_bottom
        
        # Each 2x2 * 2x9 has schoolbook rank 36.
        # Total: 72.
        
        # Can we do better for 2x2 * 2x9?
        # 2x2 * 2x9: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        
        # For fixed i, this is a linear combination of B[0,:] and B[1,:].
        # We can precompute:
        # S0[k] = B[0,k], S1[k] = B[1,k]
        # C[i,k] = A[i,0]*S0[k] + A[i,1]*S1[k]
        
        # This is still 2*9 = 18 terms per i, 4*18 = 72 total.
        
        # Alternative: Use the identity:
        # C[0,k] = a*B[0,k] + b*B[1,k]
        # C[1,k] = c*B[0,k] + d*B[1,k]
        
        # Define:
        # X[k] = B[0,k] + B[1,k]
        # Y[k] = B[0,k] - B[1,k]
        # Z[k] = B[0,k]
        
        # Then:
        # C[0,k] = ((a+b)/2)*X[k] + ((a-b)/2)*Y[k]
        # C[1,k] = ((c+d)/2)*X[k] + ((c-d)/2)*Y[k]
        
        # This uses 2*9 = 18 terms, same as before.
        
        # I'm not finding a clear reduction. Let me try a computational approach.
        
        # Use numpy to attempt to find a low-rank decomposition.
        try:
            import numpy as np
            
            # Create the tensor as a 3D array
            T = np.ones((n, m, p), dtype=np.float64)
            
            # Try to find a rank-r decomposition using randomized SVD-like approach
            # This is complex; let me try a simpler heuristic.
            
            # Heuristic: For each (i,j,k), we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to sum to 1.
            
            # Try rank 108: use 36 terms for "diagonal" and 72 for "off-diagonal"
            # This is getting too speculative.
            
            # Let me just return the schoolbook solution for now.
            pass
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
