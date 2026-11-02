"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Return a low-rank tensor decomposition for 2x3x9 matrix multiplication.
    
    Tries to find a decomposition with rank < 54 using structured patterns.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Try to build a rank-50 decomposition using a structured approach
    # Split k into 3 groups of 3: {0,1,2}, {3,4,5}, {6,7,8}
    # For each group, use a 2x3x3 decomposition
    # Schoolbook for 2x3x3 is rank 18, total 54
    # Try to save 2 ranks per group for the first two groups
    
    # Strategy: For each group of 3 k-values, use 16 terms instead of 18
    # This gives 16+16+18 = 50 total
    
    def build_2x3x3_decomp(k_offset, save_ranks=2):
        """Build a 2x3x3 decomposition. Returns (U_list, V_list, W_list) of terms.
        Each term is a tuple (u_vec, v_vec, w_vec) where:
        u_vec has 6 entries (i*m+j for i in 0..1, j in 0..2)
        v_vec has 9 entries (j*p+k for j in 0..2, k in 0..8) but we only use 3 k's
        w_vec has 6 entries (k*n+i for k in 0..8, i in 0..1) but we only use 3 k's
        
        Actually, let's think in local indices for the 2x3x3 subproblem:
        i in 0..1, j in 0..2, k_local in 0..2 (k_global = k_offset + k_local)
        """
        terms = []
        # Schoolbook: 18 terms
        for i in range(2):
            for j in range(3):
                for kl in range(3):
                    u = [0.0]*6
                    v = [0.0]*27  # full V size
                    w = [0.0]*18  # full W size
                    u[i*3+j] = 1.0
                    v[j*9 + (k_offset + kl)] = 1.0
                    w[(k_offset + kl)*2 + i] = 1.0
                    terms.append((u, v, w))
        
        if save_ranks == 0:
            return terms
        
        # Try to merge some terms to save ranks
        # Idea: For fixed j, combine the two i-terms for different k's
        # This is tricky. Let's try a different approach.
        # 
        # Alternative: Use the fact that for fixed i and k, the sum over j can be
        # computed with fewer products if we use Strassen-like tricks.
        #
        # For 2x3x3, consider: C[i,kl] = A[i,0]*B[0,kl] + A[i,1]*B[1,kl] + A[i,2]*B[2,kl]
        # 
        # One known approach: use 16 multiplications for 2x3x3
        # Let's try: 
        # For each pair of j's, use a 2x2x3 decomposition with rank 12 (schoolbook)
        # and share across the 3 j-pairs... this doesn't directly work.
        
        # Let me try a concrete rank-16 construction for 2x3x3:
        # This is a known result. The construction:
        # 
        # Let a_i = A[i,j] for i=0,1 and b_jk = B[j,k] for j=0,1,2, k=0,1,2
        # 
        # We'll use the following 16 products:
        # For k=0,1,2 and j=0,1,2: A[i,j]*B[j,k] gives 18 terms
        # To save 2, we need to combine 2 pairs.
        
        # Actually, let me try a different structured approach.
        # Use the following identity for 2x3x3:
        # 
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        
        # Group by j:
        # j=0: contributes A[i,0]*B[0,k] for i=0,1; k=0,1,2 -> 6 terms
        # j=1: contributes A[i,1]*B[1,k] for i=0,1; k=0,1,2 -> 6 terms  
        # j=2: contributes A[i,2]*B[2,k] for i=0,1; k=0,1,2 -> 6 terms
        
        # For each j, we have a 2x1x3 subproblem: compute [A[i,j]*B[j,k] for k=0,1,2] for i=0,1
        # That's 2 outputs per i... wait, it's 3 outputs per i (one per k).
        # So for fixed j, we compute a 2x3 vector: [A[0,j]*B[j,0], A[0,j]*B[j,1], A[0,j]*B[j,2],
        #                                                  A[1,j]*B[j,0], A[1,j]*B[j,1], A[1,j]*B[j,2]]
        # This requires 6 scalar products (schoolbook).
        
        # Can we do 2x1x3 in fewer than 6? The tensor is essentially two independent 
        # 1x1x3 problems (one for each i), so minimum rank is 3+3=6. No savings here.
        
        # So the 2x3x3 problem decomposes into three 2x1x3 problems, each of rank 6, total 18.
        # This suggests rank 18 is optimal for 2x3x3 when viewed this way.
        
        # But wait - we can share across j! The A[i,j] values are different for different j,
        # but maybe we can use linear combinations.
        
        # Let me try: compute (A[0,0]+A[0,1])*B[0,k] and (A[0,0]+A[0,1])*B[1,k]...
        # This doesn't directly help because we need A[0,0]*B[0,k] and A[0,1]*B[1,k] separately.
        
        # I think for 2x3x3, the rank might actually be 18 (optimal).
        # Let me just use schoolbook for all groups: 18*3 = 54.
        
        # Alternative approach: don't split into groups. Use the full 2x3x9 structure.
        # Maybe there's a global optimization.
        
        return terms  # return schoolbook for now
    
    # Since structured savings seem hard, let me try a randomized search
    # for a rank-50 decomposition using integer/half-integer entries.
    
    # Actually, let me try a specific known technique:
    # For the 2x3x9 problem, use the following construction:
    # 
    # View the computation as: for each of the 3 j-values, compute a 2x9 vector
    # A[i,j] * B[j,:] (which is a scalar times a row vector). That's 3 scalar-vector
    # products per i, so 6 scalar-vector products total, each of length 9 = 54 scalars.
    #
    # To save: if we can compute some linear combinations of the A[i,j] values first.
    # For example, compute S_k = A[0,k]*B[0,:] + A[1,k]*B[1,:] ... no, that mixes i.
    #
    # Let me try: for each k in 0..8, compute:
    #   P_k = A[0,0]*B[0,k] + A[1,0]*B[0,k]  (1 product)
    #   Q_k = A[0,0]*B[0,k] - A[1,0]*B[0,k]  (1 product)
    # Then A[0,0]*B[0,k] = (P_k+Q_k)/2, A[1,0]*B[0,k] = (P_k-Q_k)/2
    # But that's 2 products per k for j=0, same as schoolbook's 2 products per k.
    
    # I'll fall back to schoolbook rank 54 but verify correctness.
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]  # 6 x 54
    V = [[0] * rank for _ in range(m * p)]  # 27 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give
        # the coefficient for A[i,j]*B[j,k] which is 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        # Also check that cross terms are zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i,j,k) == (i2,j2,k2):
                                    continue
                                # The tensor entry T[i,j,k] for output (i,k) from input (j)
                                # Actually, the tensor is T_{i,j,k} where the decomposition is:
                                # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                                # We need T[i,j,k] = 1 if this is the right term, else 0
                                # Wait, the tensor for matrix mult is:
                                # T[i,j,k] = delta (it's 1 for all i,j,k in the schoolbook sense)
                                # No - the tensor is T[i,j,k] = 1 for all valid i,j,k
                                # because C[i,k] = sum_j A[i,j] B[j,k]
                                # So the tensor has T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
                                pass
        return True
    
    # The verification above is incomplete. Let me do a proper check.
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k
    # (since C[i,k] = sum_j A[i,j]*B[j,k], the tensor coefficient is 1 for each (i,j,k))
    # 
    # A rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is indexed by (i,j) -> row i*m+j, V by (j,k) -> row j*p+k, W by (k,i) -> row k*n+i
    # 
    # So we need: for all i,j,k: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    verified = False
                    break
        if not verified:
            break
    
    if not verified:
        # Should not happen with schoolbook, but safety check
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
