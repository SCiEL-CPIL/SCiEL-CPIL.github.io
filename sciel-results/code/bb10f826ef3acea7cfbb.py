"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
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
    
    result = {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return result
    
    # Verify schoolbook
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build target tensor
    T_target = np.ones((n, m, p))
    
    # Verify
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
    
    if not np.allclose(T_recon, T_target, atol=1e-10):
        return result
    
    # Try to reduce rank using linear algebra
    # Strategy: The tensor equation T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # is multilinear. We can try to find a smaller set of terms.
    
    # Approach: Try rank r from 95 down to some limit, check feasibility
    # This is NP-hard in general, so we use heuristics.
    
    # Heuristic 1: Look for pairs of terms that can be merged
    # Two terms (i1,j1,k1) and (i2,j2,k2) share the same U-factor if i1*m+j1 == i2*m+j2
    # share V-factor if j1*p+k1 == j2*p+k2, share W-factor if k1*n+i1 == k2*n+i2
    
    # For merging, we need the tensor contributions to be linearly dependent.
    
    # Better approach: Use the fact that the problem is small enough to try
    # iterative elimination with linear solves.
    
    # Represent the problem as: find vectors u_c in R^12, v_c in R^32, w_c in R^24
    # such that for all (i,j,k): sum_c u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i] = 1
    
    # This is a system of n*m*p = 96 equations in r*(12+32+24) = 68r unknowns (but nonlinear).
    
    # Alternative: Fix U and V, solve for W, or use alternating minimization.
    
    # Given time constraints, try a structured approach:
    # Group by i (3 groups of 4*8=32 terms each)
    # For each i, the sub-tensor is T[i,j,k] = 1 for all j,k
    # This is a 4x8 all-ones tensor, which has rank 1 (as a matrix) but as part of the 3D tensor...
    
    # Actually, let's try a direct rank reduction search.
    # Start with schoolbook and try to eliminate one term at a time.
    
    # To eliminate term c0, we need to adjust other terms such that:
    # sum_{c != c0} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)
    # where previously term c0 contributed 1 to exactly one (i,j,k) entry.
    
    # This is equivalent to: can we find vectors u_c, v_c, w_c for c != c0
    # such that the sum equals T - e_{c0} (where e_{c0} is the basis vector for term c0)?
    
    # This is still nonlinear. Let's try a simpler heuristic.
    
    # Heuristic: Try to find a decomposition of rank 95, 94, etc.
    # by randomly perturbing and checking.
    
    # Actually, for this specific problem, let's try a known result:
    # The rank of 3x4x8 matrix multiplication.
    
    # Known bounds: 
    # - Trivial: 96
    # - Winograd-like improvements may exist
    
    # Let's try a greedy approach: 
    # 1. Start with all 96 terms
    # 2. For each term, check if removing it and solving for adjustments is feasible
    # 3. Use least-squares as a relaxation
    
    if time.time() > deadline - 30:
        return result
    
    # Try to find a rank-95 decomposition
    # Remove one term and see if we can compensate
    
    # The key insight: if two terms share structure, we might merge them.
    # For example, terms (i,j,k1) and (i,j,k2) for same i,j but different k
    # have the same U-factor and same V-row structure.
    
    # Let's try a different approach: use the fact that the all-ones tensor
    # in each "slice" might allow compression.
    
    # For fixed i, the sub-tensor T[i, :, :] is a 4x8 all-ones matrix.
    # This can be written as u_i * v_i^T where u_i is 4-dim all-ones and v_i is 8-dim all-ones.
    # But in the tensor decomposition, we need 3D factors.
    
    # Let's try: for each i, use a single term that covers all j,k for that i.
    # Term for i: U has 1 at position i*m+j for all j, V has 1 at position j*p+k for all j,k,
    # W has 1 at position k*n+i for all k.
    # But this doesn't work because U, V, W are shared across terms.
    
    # Let's just try a computational search for rank reduction.
    
    # Use scipy for linear algebra
    try:
        from scipy.linalg import lstsq
        from scipy.optimize import minimize
    except ImportError:
        return result
    
    if time.time() > deadline - 20:
        return result
    
    # Try to find a lower-rank decomposition using alternating least squares
    # Start with schoolbook and try to merge terms
    
    # Actually, let's try a simpler approach: check if rank 95 is achievable
    # by removing one term and solving a linear system for the remaining terms'
    # W-factors (with U and V fixed to schoolbook values except for adjustments).
    
    # This is getting complex. Let's try a random search approach.
    
    # Randomly perturb the schoolbook decomposition and check if it still works.
    # If we can find a perturbation that makes one term redundant, we reduce rank.
    
    np.random.seed(42)
    
    best_rank = rank
    best_result = result
    
    # Try a few random rank-reduction attempts
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        
        # Try to remove one random term
        c_remove = np.random.randint(rank)
        
        # Create modified U, V, W with term c_remove zeroed out
        U_mod = U_np.copy()
        V_mod = V_np.copy()
        W_mod = W_np.copy()
        U_mod[:, c_remove] = 0
        V_mod[:, c_remove] = 0
        W_mod[:, c_remove] = 0
        
        # Check if the remaining terms can still produce the target tensor
        # This requires solving for adjusted U, V, W
        # For simplicity, check if the residual can be expressed as a rank-1 tensor
        # (which would mean we just need to adjust one term)
        
        T_current = np.zeros((n, m, p))
        for c in range(rank):
            if c == c_remove:
                continue
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_current[i, j, k] += U_mod[i*m+j, c] * V_mod[j*p+k, c] * W_mod[k*n+i, c]
        
        residual = T_target - T_current
        
        # Check if residual is a rank-1 tensor
        # A tensor is rank-1 if it can be written as a ⊗ b ⊗ c
        # This is equivalent to the tensor having a specific structure
        
        # For a 3D tensor A, it's rank-1 if A[i,j,k] = a[i]*b[j]*c[k]
        # We can check this by looking at the ratios
        
        # Reshape residual to check rank-1 structure
        # If residual[i,j,k] = a[i]*b[j]*c[k], then for any i1,i2,j1,j2,k1,k2:
        # residual[i1,j1,k1]*residual[i2,j2,k2] = residual[i1,j2,k1]*residual[i2,j1,k2]
        
        # Check a few entries
        is_rank1 = True
        nonzero_entries = np.argwhere(residual != 0)
        if len(nonzero_entries) == 0:
            # Residual is zero, so we can remove the term
            best_rank = rank - 1
            best_result = {
                "U": [U_mod[:, c] for c in range(rank) if c != c_remove],
                "V": [V_mod[:, c] for c in range(rank) if c != c_remove],
                "W": [W_mod[:, c] for c in range(rank) if c != c_remove]
            }
            break
        elif len(nonzero_entries) < 2:
            # Only one nonzero entry, can't be rank-1 unless it's zero
            is_rank1 = False
        else:
            # Check rank-1 condition on a few entries
            e1, e2 = nonzero_entries[0], nonzero_entries[1]
            i1, j1, k1 = e1
            i2, j2, k2 = e2
            
            # Check if residual[i1,j1,k1]*residual[i2,j2,k2] == residual[i1,j2,k1]*residual[i2,j1,k2]
            val1 = residual[i1, j1, k1] * residual[i2, j2, k2]
            val2 = residual[i1, j2, k1] * residual[i2, j1, k2]
            if abs(val1 - val2) > 1e-10:
                is_rank1 = False
        
        if is_rank1 and len(nonzero_entries) > 0:
            # Try to find a, b, c such that residual[i,j,k] = a[i]*b[j]*c[k]
            # Use the first nonzero entry to normalize
            i0, j0, k0 = nonzero_entries[0]
            if residual[i0, j0, k0] != 0:
                a = np.zeros(n)
                b = np.zeros(m)
                c = np.zeros(p)
                a[i0] = 1
                b[j0] = 1
                c[k0] = 1
                
                # For all nonzero entries, determine a, b, c
                # This is underdetermined, so we need to be careful
                # For simplicity, check if it works
                valid = True
                for i, j, k in nonzero_entries:
                    expected = a[i] * b[j] * c[k]
                    if abs(residual[i, j, k] - expected) > 1e-10:
                        valid = False
                        break
                
                if valid:
                    # We found a rank-1 residual, so we can add it back as one term
                    # This means we can remove the original term and add one new term
                    # Net rank change: 0 (removed one, added one)
                    # So no improvement
                    pass
    
    # Try a more systematic approach: look for known decompositions
    # For 3x4x8, check if there's a known lower bound
    
    # Return the best result found
    if best_rank < rank:
        return best_result
    
    return result
# EVOLVE-BLOCK-END
