"""Matrix multiplication 2,2,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, R)
    # V: (m*p, rank) = (12, R)  
    # W: (p*n, rank) = (12, R)
    # Tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try to find a low-rank decomposition
    # Start with a structured ansatz and refine
    
    # For 2x2 x 2x6, try rank 20 with a structured approach
    # Use the fact that we can share structure between the two "halves" of the 6 columns
    
    R = 20
    
    # Initialize with schoolbook-like structure but with sharing
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Strategy: Use a Kronecker-like structure
    # Split the 6 columns into 3 pairs, use shared factors
    
    import numpy as np
    
    # Let's try a direct construction
    # For each (i,j,k), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Try: U[i,j,r] depends on i,j and a "block" index
    # V[j,k,r] depends on j,k and block
    # W[k,i,r] depends on k,i and block
    
    # Simpler: just set up and solve the polynomial system
    # For R=20, we have 28*20=560 variables, 24 equations
    
    # Use a greedy/structured initialization
    # Set most entries to 0 or 1, then adjust
    
    # Try a specific pattern:
    # For r = 0..19, assign each r to handle specific (i,j,k) tuples
    # But with sharing to reduce rank
    
    # Let me try: first 12 terms handle schoolbook for first 3 k values
    # Next 8 terms handle the rest with some sharing
    
    # Actually, let's just try rank 20 with a simple structured solution
    # and verify it works
    
    U = np.zeros((4, R), dtype=float)
    V = np.zeros((12, R), dtype=float)
    W = np.zeros((12, R), dtype=float)
    
    # Schoolbook for all 24 terms would be rank 24
    # We need to save 4 ranks by sharing
    
    # Key insight: For fixed (i,j), the term A[i,j]*B[j,k] contributes to all k
    # So U[i,j,r] can be shared across k if V and W are structured appropriately
    
    # Try: Use 4 terms for the "diagonal" structure, then add correction terms
    
    # Let me try a different approach: use the known result that 
    # 2x2 x 2x6 might have rank 20 or less
    
    # Construct a solution by solving the system
    # Variables: U (4xR), V (12xR), W (12xR)
    # Equations: for each (i,j,k) in range(2)xrange(2)xrange(6):
    #   sum_r U[i*2+j, r] * V[j*6+k, r] * W[k*2+i, r] = 1
    
    # This is a system of 24 polynomial equations in 560 variables
    # Very underdetermined, so we can fix many variables
    
    # Fix strategy:
    # - Set U[i*2+j, r] = 1 for r = i*2+j (first 4 terms), 0 otherwise initially
    # - Then adjust V and W
    
    # Better: use a randomized initialization and Newton's method
    # But that's complex. Let's try a structured ansatz.
    
    # Ansatz: 
    # r = 0..3: handle "main" terms
    # r = 4..19: correction terms
    
    # For r in 0..3: let r = i*2+j for i in 0..1, j in 0..1
    # U[i*2+j, r] = 1 if r == i*2+j else 0
    # V[j*6+k, r] = some value
    # W[k*2+i, r] = some value
    
    # This is getting complex. Let me just try to find ANY valid rank-20 solution
    # by solving the system with a specific structure.
    
    # Simplest valid approach: use rank 24 (schoolbook) as fallback
    # But try rank 20 first
    
    # Let me try a specific construction:
    # Use the fact that for 2x2, we can use rank-2 for the 2x2 part
    
    # For 2x2 x 2x6, think of it as:
    # C[:,0:3] = A @ B[:,0:3]  (2x2 x 2x3)
    # C[:,3:6] = A @ B[:,3:6]  (2x2 x 2x3)
    
    # If 2x2 x 2x3 has rank R3, then total rank <= 2*R3
    # Schoolbook for 2x2 x 2x3 is rank 12, so 2*12=24
    # If R3 < 12, we save ranks
    
    # Known: 2x2 x 2x2 has rank 7 (Winograd)
    # 2x2 x 2x3 might have rank < 12
    
    # Let me try to find rank for 2x2 x 2x3 first, then combine
    
    # For now, let's just output a valid rank-20 solution if we can find one,
    # otherwise fall back to rank 24
    
    # Try a simple structured rank-20:
    # Use 20 terms where some share structure
    
    # Actually, let's just verify the schoolbook works and see if we can
    # find a better one within time budget
    
    # Schoolbook rank 24:
    U_sb = np.zeros((4, 24))
    V_sb = np.zeros((12, 24))
    W_sb = np.zeros((12, 24))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify(U_mat, V_mat, W_mat, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U_sb, V_sb, W_sb, n, m, p):
        # Shouldn't happen, but fallback
        pass
    
    # Now try to find rank < 24
    # Try rank 20 with a structured approach
    
    # For rank R, we need to solve:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Use a specific structure: 
    # Let's say for r < 4, U[i*m+j,r] is nonzero only for specific (i,j)
    # This is a bilinear in U,V,W system
    
    # Try: fix W to be identity-like, solve for U,V
    # Or use a random initialization and iterate
    
    # Given time constraints, let's try a few structured rank-20 candidates
    
    # Candidate 1: Share the last 4 k-values
    # Use 16 terms for k=0,1,2,3 (schoolbook) + 4 shared terms for k=4,5
    
    # Actually, let me just try to solve it numerically
    # For R=20, fix some variables to reduce the system
    
    # Fix: U[i*2+j, r] = 1 for r = i*2+j (r=0..3), 0 for r>=4
    # Then we need: for each (i,j,k):
    #   V[j*6+k, i*2+j] * W[k*2+i, i*2+j] + sum_{r=4}^{19} 0*... = 1
    #   => V[j*6+k, i*2+j] * W[k*2+i, i*2+j] = 1
    # This requires V[j*6+k, i*2+j] = 1 and W[k*2+i, i*2+j] = 1 for all j,k
    # But W[k*2+i, i*2+j] depends on j, which shouldn't happen
    
    # So that doesn't work. Let me try a different structure.
    
    # Given the complexity, let's just return the schoolbook rank-24 solution
    # unless we can quickly find a better one
    
    # For now, return schoolbook
    U_out = U_sb.tolist()
    V_out = V_sb.tolist()
    W_out = W_sb.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
