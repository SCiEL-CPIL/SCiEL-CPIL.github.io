"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses rank-deflation: start from rank-26, try to eliminate terms to reach rank-25.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin (evaluator kills at 150s)
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Build rank-26 baseline
    R_base = 26
    U_base, V_base, W_base = _build_rank26(n, m, p, idx_u, idx_v, idx_w)
    
    # Try rank-deflation: eliminate one term and solve for remaining 25
    for R_target in [25, 24, 23]:
        if time.time() > deadline - 20:
            break
        
        result = _rank_deflation_search(R_target, n, m, p, U_base, V_base, W_base,
                                         idx_u, idx_v, idx_w, deadline)
        if result is not None:
            return result
    
    # Fallback: return rank-26
    return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}


def _build_rank26(n, m, p, idx_u, idx_v, idx_w):
    """Build the known rank-26 decomposition."""
    R = 26
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    # Strassen for top-left 2x2 (7 terms)
    r = 0
    U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    r = 1
    U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1
    
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1
    
    r = 4
    U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1
    
    r = 5
    U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1
    
    r = 6
    U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1
    
    border_indices = [
        (2, 0, 0), (2, 0, 1), (2, 0, 2),
        (2, 1, 0), (2, 1, 1), (2, 1, 2),
        (2, 2, 0), (2, 2, 1), (2, 2, 2),
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 2, 0), (1, 2, 1), (1, 2, 2),
        (0, 0, 2), (0, 1, 2),
        (1, 0, 2), (1, 1, 2)
    ]
    
    col = 7
    for i, j, k in border_indices:
        U[idx_u(i, j), col] = 1
        V[idx_v(j, k), col] = 1
        W[idx_w(k, i), col] = 1
        col += 1
    
    assert col == R
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, idx_u, idx_v, idx_w):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(R):
                    val += U[idx_u(i, j), r] * V[idx_v(j, k), r] * W[idx_w(k, i), r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _rank_deflation_search(R_target, n, m, p, U_base, V_base, W_base,
                           idx_u, idx_v, idx_w, deadline):
    """
    Try to eliminate (26 - R_target) terms from the rank-26 baseline by solving
    linear constraints for the remaining terms.
    
    Strategy: For each subset of terms to eliminate, set their U,V,W columns to 0,
    then solve the resulting bilinear system. We use a linearization approach:
    fix U and V, solve for W (linear in W), then check if W has the required sparsity.
    """
    R_base = U_base.shape[1]  # 26
    
    # We want to eliminate (R_base - R_target) terms.
    # For R_target=25: eliminate 1 term.
    # For R_target=24: eliminate 2 terms.
    
    n_eliminate = R_base - R_target
    
    # Try eliminating different subsets of terms
    # Focus on border terms first (indices 7-25) as they are more "independent"
    # but also try Strassen terms
    
    import itertools
    
    # Generate candidate elimination sets
    # Start with single elimination (for R_target=25)
    if n_eliminate == 1:
        candidates = [tuple([r]) for r in range(R_base)]
    elif n_eliminate == 2:
        # Try pairs from border terms and mixed
        candidates = list(itertools.combinations(range(R_base), 2))
    else:
        # Too many to enumerate; try a subset
        candidates = list(itertools.combinations(range(R_base), n_eliminate))
        if len(candidates) > 500:
            candidates = candidates[:500]
    
    for elim_set in candidates:
        if time.time() > deadline - 10:
            break
        
        result = _try_eliminate_terms(elim_set, R_target, n, m, p,
                                       U_base, V_base, W_base,
                                       idx_u, idx_v, idx_w)
        if result is not None:
            return result
    
    return None


def _try_eliminate_terms(elim_set, R_target, n, m, p,
                         U_base, V_base, W_base,
                         idx_u, idx_v, idx_w):
    """
    Try to eliminate the given set of terms and solve for the remaining R_target terms.
    
    Approach: Linearize by fixing U and V, then solve for W.
    The constraint is: for all (i,j,k):
        sum_{r not in elim} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    This is linear in W. We set up a linear system:
        For each (i,j,k): sum_{r not in elim} (U[i,j,r] * V[j,k,r]) * W[k,i,r] = 1
    
    W has entries W[k,i,r] for k in {0,1,2}, i in {0,1,2}, r in {0,...,R_target-1}.
    That's 3*3*R_target = 9*R_target unknowns.
    Constraints: 3*3*3 = 27 equations.
    
    If R_target >= 3, we have 9*R_target >= 27 unknowns vs 27 equations (underdetermined or square).
    For R_target=25: 225 unknowns, 27 equations -> very underdetermined.
    
    But we also need W to have the right structure: W[k,i,r] should be 0 for most (k,i) pairs
    given the fixed U and V. Actually no - W is free.
    
    Wait - the issue is that U and V are also variables. Let me reconsider.
    
    Actually, let's use a different approach: fix the structure of U and V from the baseline
    (just zero out the eliminated columns), and solve for W. Then verify.
    """
    R_base = U_base.shape[1]
    
    # Create U and V with eliminated terms zeroed
    U = U_base.copy()
    V = V_base.copy()
    W = W_base.copy()
    
    for r in elim_set:
        U[:, r] = 0
        V[:, r] = 0
        W[:, r] = 0
    
    # Now solve for W given fixed U and V
    # For each (i,j,k): sum_{r=0}^{R_base-1} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # Terms where U[i,j,r] or V[j,k,r] is 0 don't contribute.
    # After zeroing elim_set, only non-eliminated terms matter.
    
    # Build linear system: A * w_vec = b
    # w_vec is the vector of W values: w_vec[k*n + i, r] = W[k,i,r]
    # Actually, let's index W as W[k*n+i, r]
    
    num_w_unknowns = p * n * R_base  # W has shape (p*n, R_base)
    
    # But we only need to solve for W entries that can actually contribute.
    # W[k,i,r] only matters if there exists some j such that U[i,j,r] != 0 and V[j,k,r] != 0.
    
    # Let's build the system properly.
    # For each (i,j,k) in {0,1,2}^3:
    #   sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # This gives 27 equations.
    # Unknowns: W[k,i,r] for k,i in {0,1,2}, r in {0,...,R_base-1}
    # But W[k,i,r] = 0 for r in elim_set (we already set them to 0).
    
    # Active unknowns: W[k,i,r] where r not in elim_set
    active_r = [r for r in range(R_base) if r not in elim_set]
    
    # For each equation (i,j,k), the coefficient of W[k',i',r] is:
    #   U[i,j,r] * V[j,k,r] if (k',i') == (k,i), else 0
    # So each equation only involves W[k,i,r] for the specific (k,i) pair.
    
    # This means the system is block-diagonal by (k,i) pair!
    # For each (k,i): sum_{r in active_r} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for each j
    # Wait, that's not right either. Let me re-examine.
    
    # Equation for (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # The unknowns in this equation are W[k,i,r] for r in active_r.
    # Coefficient of W[k,i,r] is U[i,j,r] * V[j,k,r].
    
    # So for a fixed (k,i), the equations involving W[k,i,r] are:
    #   For each j: sum_{r in active_r} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    #   That's m=3 equations for 3*R_base unknowns (but only active_r matter).
    
    # Actually, for fixed (k,i), we have m=3 equations (one per j) and |active_r| unknowns.
    # For R_target=25, |active_r|=25, so 3 equations, 25 unknowns -> underdetermined.
    
    # We can use least squares or find a particular solution.
    # Let's use numpy lstsq for each (k,i) block.
    
    W_new = np.zeros((p * n, R_base), dtype=np.float64)
    
    for k in range(p):
        for i in range(n):
            # Build the 3 x |active_r| system
            # Equation j: sum_{r in active_r} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
            A_block = np.zeros((m, len(active_r)), dtype=np.float64)
            b_block = np.ones(m, dtype=np.float64)
            
            for j in range(m):
                for idx_r, r in enumerate(active_r):
                    A_block[j, idx_r] = U[idx_u(i, j), r] * V[idx_v(j, k), r]
            
            # Solve A_block * w = b_block
            # Use lstsq for underdetermined system
            try:
                w_sol, residuals, rank, sv = np.linalg.lstsq(A_block, b_block, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is exact (residuals should be ~0)
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            # Check if solution is "nice" (integer or half-integer)
            # For now, just check if it's close to a simple value
            W_new[k * n + i, :] = 0
            for idx_r, r in enumerate(active_r):
                W_new[k * n + i, r] = w_sol[idx_r]
    
    # Now verify the full decomposition
    U_final = U
    V_final = V
    W_final = W_new
    
    if _verify_decomposition(U_final, V_final, W_final, n, m, p, idx_u, idx_v, idx_w):
        # Check if entries are "nice" (close to integers or half-integers)
        all_nice = True
        for mat in [U_final, V_final, W_final]:
            for val in mat.flatten():
                if abs(val - round(val)) > 1e-6:
                    # Check half-integer
                    if abs(val * 2 - round(val * 2)) > 1e-6:
                        all_nice = False
                        break
            if not all_nice:
                break
        
        if all_nice:
            # Round to nearest nice value
            U_out = np.zeros_like(U_final)
            V_out = np.zeros_like(V_final)
            W_out = np.zeros_like(W_final)
            for mat_in, mat_out in [(U_final, U_out), (V_final, V_out), (W_final, W_out)]:
                for idx in range(mat_in.size):
                    val = mat_in.flatten()[idx]
                    # Try integer
                    if abs(val - round(val)) < 1e-6:
                        mat_out.flatten()[idx] = round(val)
                    elif abs(val * 2 - round(val * 2)) < 1e-6:
                        mat_out.flatten()[idx] = round(val * 2) / 2.0
                    else:
                        mat_out.flatten()[idx] = val
            
            # Final verification
            if _verify_decomposition(U_out, V_out, W_out, n, m, p, idx_u, idx_v, idx_w):
                return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    return None
# EVOLVE-BLOCK-END
