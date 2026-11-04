"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Build the target tensor: T[a, b, c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1, all else 0
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor entry: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # And = 0 for all other (a,b,c) combinations
    
    # The key constraint: for each (i,j,k), the sum over r must equal 1
    # For all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i), sum must be 0
    
    # Schoolbook solution (rank 120)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb
    
    def verify(U, V, W, rank):
        """Verify exact decomposition using the tensor convention."""
        # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        # Also check that non-matching entries are zero
        # For any (a, b, c) not corresponding to valid (i,j,k), sum should be 0
        # But this is automatically satisfied if the above holds for the structure
        # Actually, we need to verify: for a=i*m+j, b=j2*p+k, c=k2*n+i2
        # if j != j2 or i != i2 or k != k2, then sum should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    for j2 in range(m):
                        for k2 in range(p):
                            for i2 in range(n):
                                b = j2 * p + k
                                c = k2 * n + i
                                if j != j2 or i != i2 or k != k2:
                                    s = 0.0
                                    for r in range(rank):
                                        s += U[a, r] * V[b, r] * W[c, r]
                                    if abs(s) > 1e-8:
                                        return False
        return True
    
    # Verify schoolbook
    if not verify(best_U, best_V, best_W, best_rank):
        # Should not happen, but fallback
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find lower rank using structured search
    # Strategy: for rank r < 120, we need to find U, V, W such that
    # the bilinear form constraints are satisfied exactly.
    
    # Use a greedy/iterative approach: start with schoolbook and try to merge terms
    # by finding linear dependencies in the tensor structure.
    
    # Alternative: use the fact that for rectangular matrix multiplication,
    # there are known results. For 4x5x6, the rank is not known to be less than 120
    # in simple constructions, but let's try a few structured reductions.
    
    # Try rank 119: can we merge two schoolbook terms?
    # Term (i1,j1,k1) and term (i2,j2,k2) can be merged if there exist
    # scalars alpha, beta such that the combined term satisfies all constraints.
    
    # For a merge of terms c1 and c2 into a single term c:
    # We need U_new, V_new, W_new such that:
    # U_new[a] * V_new[b] * W_new[c] replaces U1[a]*V1[b]*W1[c] + U2[a]*V2[b]*W2[c]
    # This is generally not possible unless the two terms have special structure.
    
    # Let's try a different approach: use numerical optimization for small rank reductions
    # Start with rank 119 and try to find a valid solution.
    
    # For rank r, we have r * (nm + mp + pn) = r * 74 unknowns
    # and nm * mp * pn = 14400 constraints (but many are redundant due to structure)
    
    # The actual independent constraints: for each (i,j,k), we need the sum = 1
    # That's 120 constraints. Plus the "zero" constraints for non-matching indices.
    # Total independent constraints is much less than 14400.
    
    # Let's count: the tensor has 120 non-zero entries. The bilinear form gives
    # 120 equations (one per non-zero entry) plus the zero equations.
    # But the zero equations are automatically satisfied if we maintain the structure.
    
    # Actually, the key insight: the multiplication tensor is determined by its 120
    # non-zero entries. A rank-r decomposition must satisfy:
    # sum_{r'=1}^{r} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1 for all (i,j,k)
    
    # This is 120 equations with r*74 unknowns. For r=119, we have 8806 unknowns
    # and 120 equations - highly underdetermined. So solutions should exist!
    
    # Let's use a randomized approach: initialize randomly and solve the linear system
    # by fixing V and W, then solving for U, or using alternating least squares.
    
    # Better: use the fact that we can express this as a linear system.
    # Fix V and W, then for each (i,j,k): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
    # This is linear in U! So we can solve for U given V and W.
    
    def try_rank_reduction(target_rank, max_attempts=10):
        """Try to find a valid decomposition of the given rank."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random initialization of V and W
            V_try = np.random.randn(mp, target_rank) * 0.5
            W_try = np.random.randn(pn, target_rank) * 0.5
            
            # For each (i,j,k), we need:
            # sum_r U[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r] = 1
            # Let's define M[(i,j,k), r] = V_try[j*p+k, r] * W_try[k*n+i, r]
            # Then we need: sum_r U[i*m+j, r] * M[(i,j,k), r] = 1
            
            # Group by (i,j): for fixed (i,j), we have p equations (one per k)
            # with target_rank unknowns (U[i*m+j, r] for r=0..target_rank-1)
            # So for each (i,j), we solve a p x target_rank system:
            # M_ij * U_ij^T = 1 (vector of ones)
            
            U_try = np.zeros((nm, target_rank))
            valid = True
            
            for i in range(n):
                for j in range(m):
                    # Build the p x target_rank matrix M_ij
                    M_ij = np.zeros((p, target_rank))
                    for k in range(p):
                        M_ij[k, :] = V_try[j * p + k, :] * W_try[k * n + i, :]
                    
                    # Solve M_ij * U_ij = ones(p)
                    # This is an underdetermined system (p equations, target_rank unknowns)
                    # Use least squares or pseudoinverse
                    try:
                        # We want U_ij such that M_ij @ U_ij = ones(p)
                        # Using pseudoinverse: U_ij = pinv(M_ij) @ ones(p)
                        rhs = np.ones(p)
                        U_ij = np.linalg.lstsq(M_ij, rhs, rcond=None)[0]
                        U_try[i * m + j, :] = U_ij
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Now verify the full decomposition
            if verify(U_try, V_try, W_try, target_rank):
                return U_try, V_try, W_try
        
        return None
    
    # Try ranks from 119 down to, say, 100
    # Due to time constraints, try a limited number
    for target_rank in range(best_rank - 1, max(90, best_rank - 30), -1):
        if time.time() > deadline - 5:
            break
        
        result = try_rank_reduction(target_rank, max_attempts=5)
        if result is not None:
            U_new, V_new, W_new = result
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, target_rank
            break  # Found a lower rank, stop
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
