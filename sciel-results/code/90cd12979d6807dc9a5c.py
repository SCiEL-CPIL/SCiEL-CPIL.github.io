import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = 4, 4, 9
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Best known: schoolbook rank 144
    best_rank = 144
    best_result = None
    
    # Try to find better decompositions via random sparse support search
    # Strategy: for a given rank R, randomly assign sparse supports to U, V, W
    # then solve the linear system to find coefficients.
    
    # The tensor equation: for all (i,j,k):
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a system of n*m*p = 4*4*9 = 144 equations.
    # Unknowns: entries of U, V, W with fixed support pattern.
    
    # We'll try ranks from 72 up to 143 (below schoolbook)
    # For each rank, try several random sparse patterns
    
    nU = n * m  # 16
    nV = m * p  # 36
    nW = p * n  # 36
    num_eqs = n * m * p  # 144
    
    # Precompute the equation indices
    # Equation (i,j,k) corresponds to index e = i*m*p + j*p + k
    # U index: u = i*m + j
    # V index: v = j*p + k
    # W index: w = k*n + i
    
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                e = i * m * p + j * p + k
                u = i * m + j
                v = j * p + k
                w = k * n + i
                eq_indices.append((e, u, v, w))
    
    # Try various rank values
    # Start with lower ranks and go up
    ranks_to_try = []
    for r in range(72, 144):
        ranks_to_try.append(r)
    
    random.shuffle(ranks_to_try)
    # But prioritize lower ranks
    ranks_to_try.sort()
    
    # For each rank, try multiple random sparse patterns
    max_attempts_per_rank = 5
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        if rank >= best_rank:
            continue
            
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline - 20:
                break
                
            # Generate random sparse support pattern
            # Each column r of U, V, W gets a small number of non-zero entries
            # We want the total number of free variables to be >= num_eqs
            
            # Strategy: for each rank index r, choose how many non-zeros in U[:,r], V[:,r], W[:,r]
            # Total unknowns = sum_r (nnz_U[r] + nnz_V[r] + nnz_W[r])
            # We need total_unknowns >= num_eqs = 144
            
            # Let's try: each column has ~1-3 non-zeros in each matrix
            # With rank R, total unknowns ≈ R * (avg_nnz_U + avg_nnz_V + avg_nnz_W)
            # For R=80, need avg total ~1.8 per column per matrix → ~1 per matrix per column
            
            # Simpler approach: randomly assign supports
            U_support = []
            V_support = []
            W_support = []
            
            for r in range(rank):
                # Choose 1-3 random rows for U
                nu = random.randint(1, 3)
                U_support.append(random.sample(range(nU), nu))
                # Choose 1-3 random rows for V
                nv = random.randint(1, 3)
                V_support.append(random.sample(range(nV), nv))
                # Choose 1-3 random rows for W
                nw = random.randint(1, 3)
                W_support.append(random.sample(range(nW), nw))
            
            total_unknowns = sum(len(s) for s in U_support) + sum(len(s) for s in V_support) + sum(len(s) for s in W_support)
            
            if total_unknowns < num_eqs:
                continue
            
            # Build the linear system
            # Each equation (i,j,k): sum_r U[u,r]*V[v,r]*W[w,r] = 1
            # This is cubic in the unknowns, not linear!
            # 
            # Alternative: fix two of the three matrices randomly, solve for the third.
            # Or: use a different approach.
            #
            # Better approach: Use the fact that if we fix U and V, then W is determined
            # by: for each (k,i) and rank r: W[k*n+i, r] appears in equations
            # where that (k,i) pair is involved.
            #
            # Actually, the system is: for each (i,j,k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # If we fix U and V, then for each (k,i), the unknowns are W[k*n+i, r] for all r.
            # But W[k*n+i, r] appears in multiple equations (for different j).
            # Specifically, for fixed (k,i), W[k*n+i, r] appears in equations where
            # that (k,i) is the W-index, i.e., for all j: equation (i,j,k).
            #
            # So if we fix U and V, we get a linear system for W.
            # Similarly, we could fix U and W, solve for V, etc.
            #
            # Let's try: randomly generate U and V (with sparse support), then solve for W.
            
            # Generate random sparse U and V
            U_mat = np.zeros((nU, rank))
            V_mat = np.zeros((nV, rank))
            
            for r in range(rank):
                for u_idx in U_support[r]:
                    U_mat[u_idx, r] = random.choice([-1, 1])
                for v_idx in V_support[r]:
                    V_mat[v_idx, r] = random.choice([-1, 1])
            
            # Now solve for W: for each (i,j,k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # Group by W index: w = k*n + i
            # For each w = k*n+i, the unknowns are W[w, r] for r in 0..rank-1.
            # These appear in equations where (i,j,k) has this (k,i) pair, i.e., for all j.
            # There are m=4 such equations per (k,i) pair.
            #
            # So we have p*n = 36 groups of m=4 equations each, with rank unknowns per group.
            # This is an underdetermined system (rank > 4 typically), so we can find a solution
            # using least squares or by picking a subset.
            #
            # Actually, we can set up: for each (k,i), we have 4 equations (one per j)
            # and 'rank' unknowns. We can pick 4 of the rank unknowns to be free and solve.
            # But this is getting complex. Let's just use least squares.
            
            # Build the system: A * W_vec = b
            # Where W_vec is the flattened W matrix (nW * rank variables)
            # Each equation (i,j,k) gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is linear in W!
            A = np.zeros((num_eqs, nW * rank))
            b = np.ones(num_eqs)
            
            for eq_idx, (e, u, v, w) in enumerate(eq_indices):
                # Equation e involves W[w, r] for all r
                for r in range(rank):
                    coeff = U_mat[u, r] * V_mat[v, r]
                    if coeff != 0:
                        A[eq_idx, w * rank + r] = coeff
            
            # Solve using least squares
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if the solution is exact (residual ~ 0)
            if len(residuals) > 0 and residuals[0] > 1e-10:
                continue
            
            # Verify the solution
            W_mat = W_vec.reshape(nW, rank)
            
            # Verify all equations
            valid = True
            for e, u, v, w in eq_indices:
                val = 0.0
                for r in range(rank):
                    val += U_mat[u, r] * V_mat[v, r] * W_mat[w, r]
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # We found a valid decomposition!
            best_rank = rank
            best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
            break  # Move to next (lower) rank
    
    # If we didn't find anything better, return schoolbook
    if best_result is None:
        # Return schoolbook rank-144 decomposition
        U = np.zeros((nU, 144))
        V = np.zeros((nV, 144))
        W = np.zeros((nW, 144))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, r] = 1
                    V[v, r] = 1
                    W[w, r] = 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
