"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if j matches in A and B
    # T shape: (n*m, m*p, p*n) -> T[i*m+j1, j2*p+k, k*n+i] = delta(j1,j2)
    
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    # Start with schoolbook rank 18
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # The tensor has 18 "basis" terms. We need to find if any subset can be represented
    # by fewer terms.
    
    # Strategy: Try rank 16, 17, 18 in order, using a structured search
    # For each candidate rank, try to find a valid decomposition
    
    # First, let's try a known rank-17 decomposition (Strassen + schoolbook hybrid)
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen for 2x2x2 block (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2])
    # A indices: 0,1,3,4; B indices: 0,1,3,4; C indices: 0,1,2,3
    U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
    
    # Schoolbook for remaining terms
    # A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # A[0:2,0:3] * B[0:3,2] -> C[0:2,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    if not _verify(U, V, W, n, m, p, rank):
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find rank 16
    if time.time() < deadline - 10.0:
        # Approach: Try to merge pairs of columns from the rank-17 solution
        # by finding linear combinations that preserve the tensor
        # 
        # Key idea: If we can find coefficients c1, c2 such that
        # c1*col_a + c2*col_b = 0 in the tensor sense, we can eliminate one
        #
        # More practically: try all pairs of columns and see if one can be
        # expressed as a linear combination of others with the remaining columns
        #
        # Alternative: use a more structured search with the specific pattern
        
        # Let's try a different approach: direct search for rank 16
        # with the constraint that entries are in {-1, -0.5, 0, 0.5, 1}
        # and the solution is "sparse" (each column affects few terms)
        
        np.random.seed(42)
        found_16 = False
        
        # Precompute the tensor for fast verification
        # T[i*m+j1, j2*p+k, k*n+i] = 1 if j1==j2 else 0
        # We need: sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] = delta(j1,j2)
        
        # For rank 16, we have 16 columns. Each column (u, v, w) contributes
        # u[i,j] * v[j,k] * w[k,i] to the tensor.
        
        # Strategy: enumerate sparse patterns
        # Each column can be characterized by which (i,j) pairs in U, (j,k) in V, (k,i) in W are nonzero
        # For efficiency, try patterns where each column affects at most 2-3 terms
        
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # Generate candidate column patterns
        # U has 6 entries (i=0,1; j=0,1,2)
        # V has 9 entries (j=0,1,2; k=0,1,2)  
        # W has 6 entries (k=0,1,2; i=0,1)
        
        max_attempts = 20000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            U_cand = np.zeros((n*m, 16))
            V_cand = np.zeros((m*p, 16))
            W_cand = np.zeros((p*n, 16))
            
            for c in range(16):
                # Randomly choose a sparse pattern
                # Pick 1-3 nonzero entries in each of U, V, W
                for _ in range(np.random.randint(1, 4)):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    U_cand[i*m+j, c] = np.random.choice(values)
                
                for _ in range(np.random.randint(1, 4)):
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    V_cand[j*p+k, c] = np.random.choice(values)
                
                for _ in range(np.random.randint(1, 4)):
                    k = np.random.randint(p)
                    i = np.random.randint(n)
                    W_cand[k*n+i, c] = np.random.choice(values)
            
            # Quick check: verify a subset of constraints first
            valid = True
            # Check diagonal terms
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        s = 0.0
                        for r in range(16):
                            s += W_cand[k*n+i, r] * U_cand[i*m+j, r] * V_cand[j*p+k, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check off-diagonal terms
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            if j1 == j2:
                                continue
                            s = 0.0
                            for r in range(16):
                                s += W_cand[k*n+i, r] * U_cand[i*m+j1, r] * V_cand[j2*p+k, r]
                            if abs(s) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                found_16 = True
                U = U_cand
                V = V_cand
                W = W_cand
                rank = 16
                break
        
        if not found_16 and time.time() < deadline - 5.0:
            # Try a more targeted approach: start from rank 17 and try to
            # perturb it to find rank 16
            # 
            # Idea: take the rank-17 solution, set one column to zero, and
            # see if we can adjust the other columns to compensate
            #
            # This is equivalent to checking if the rank-17 columns are linearly
            # independent in the tensor sense
            
            pass
    
    # Final verification
    if rank == 16:
        if not _verify(U, V, W, n, m, p, rank):
            # Revert to rank 17
            rank = 17
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
            U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
            U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
            U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
            U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
            U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
            U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
            U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
            U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
            U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
            U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
            U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
            U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
            U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
            U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
            U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
            U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
