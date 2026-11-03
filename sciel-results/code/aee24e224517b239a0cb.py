"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Use the known rank-44 solution as a base and attempt structured
    rank reduction by combining terms. Also try numerical optimization with
    better initialization.
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    size_U = n * m  # 6
    size_V = m * p  # 24
    size_W = p * n  # 16
    
    def error_func(params, R):
        """Compute residuals for given parameters."""
        U = params[0 : R*size_U].reshape(R, size_U)
        V = params[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
        W = params[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_col = W[:, k*n + i]
                    u_row = U[:, i*m + j]
                    v_row = V[:, j*p + k]
                    val = np.sum(w_col * u_row * v_row)
                    target = 1.0 if j == k else 0.0
                    residuals.append(val - target)
                    
        return np.array(residuals)
    
    def verify_solution(U, V, W, tol=1e-9):
        """Verify that the solution exactly decomposes the tensor."""
        R = U.shape[0]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_col = W[:, k*n + i]
                    u_row = U[:, i*m + j]
                    v_row = V[:, j*p + k]
                    val = np.sum(w_col * u_row * v_row)
                    target = 1.0 if j == k else 0.0
                    if abs(val - target) > tol:
                        return False
        return True
    
    def snap_to_half_integers(mat, tol=1e-6):
        """Snap values to nearest 0.5 and verify."""
        snapped = np.round(mat * 2) / 2
        # Check if snapping was close enough
        if np.max(np.abs(snapped - mat)) > tol:
            return None
        return snapped
    
    def get_rank44_solution():
        """Return the known rank-44 solution as numpy arrays (R x size)."""
        rank = 44
        U = np.zeros((rank, size_U))
        V = np.zeros((rank, size_V))
        W = np.zeros((rank, size_W))
        col = 0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[col, u_idx] = 1
                V[col, v_idx] = 1
                W[col, w_idx] = 1
                col += 1
        
        for block_idx in range(4):
            k_start = block_idx * 2
            steps = [
                {'U': {0*m + 1: 1, 1*m + 2: 1}, 'V': {1*p + k_start: 1, 2*p + (k_start+1): 1}, 'W': {k_start*n + 0: 1, (k_start+1)*n + 1: 1}},
                {'U': {1*m + 1: 1, 1*m + 2: 1}, 'V': {1*p + k_start: 1}, 'W': {k_start*n + 1: 1, (k_start+1)*n + 1: -1}},
                {'U': {0*m + 1: 1}, 'V': {1*p + (k_start+1): 1, 2*p + (k_start+1): -1}, 'W': {(k_start+1)*n + 0: 1, (k_start+1)*n + 1: 1}},
                {'U': {1*m + 2: 1}, 'V': {2*p + k_start: 1, 1*p + k_start: -1}, 'W': {k_start*n + 0: 1, k_start*n + 1: 1}},
                {'U': {0*m + 1: 1, 0*m + 2: 1}, 'V': {2*p + (k_start+1): 1}, 'W': {k_start*n + 0: -1, (k_start+1)*n + 0: 1}},
                {'U': {1*m + 1: 1, 0*m + 1: -1}, 'V': {1*p + k_start: 1, 1*p + (k_start+1): 1}, 'W': {(k_start+1)*n + 1: 1}},
                {'U': {0*m + 2: 1, 1*m + 2: -1}, 'V': {2*p + k_start: 1, 2*p + (k_start+1): 1}, 'W': {k_start*n + 0: 1}}
            ]
            for step in steps:
                for u_idx, val in step['U'].items():
                    U[col, u_idx] += val
                for v_idx, val in step['V'].items():
                    V[col, v_idx] += val
                for w_idx, val in step['W'].items():
                    W[col, w_idx] += val
                col += 1
        
        return U, V, W
    
    def to_output_format(U, V, W):
        """Convert numpy arrays (R x size) to list format (size x R)."""
        U_list = U.T.tolist()
        V_list = V.T.tolist()
        W_list = W.T.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    def try_rank_optimization(R, max_restarts=3, seed_base=1000):
        """Try to find rank-R solution using numerical optimization."""
        total_size = R * (size_U + size_V + size_W)
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                return None
            
            # Use structured initialization: start from rank-44 and try to collapse
            if R < 44:
                # Initialize with random but scaled values
                np.random.seed(seed_base + restart)
                params = np.random.randn(total_size) * 0.5
            else:
                np.random.seed(seed_base + restart)
                params = np.random.randn(total_size) * 0.3
            
            try:
                res = least_squares(error_func, params, args=(R,), method='lm', max_nfev=3000)
            except Exception:
                continue
            
            if res.cost < 1e-14:
                U = res.x[0 : R*size_U].reshape(R, size_U)
                V = res.x[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
                W = res.x[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
                
                # Try to snap to half-integers
                U_snapped = snap_to_half_integers(U)
                V_snapped = snap_to_half_integers(V)
                W_snapped = snap_to_half_integers(W)
                
                if U_snapped is not None and V_snapped is not None and W_snapped is not None:
                    if verify_solution(U_snapped, V_snapped, W_snapped):
                        return to_output_format(U_snapped, V_snapped, W_snapped)
        
        return None
    
    def try_structured_reduction():
        """
        Try to reduce rank by finding linear dependencies in the rank-44 solution.
        We systematically search for coefficient vectors that make terms redundant.
        """
        U44, V44, W44 = get_rank44_solution()
        rank44 = 44
        
        # Build the tensor decomposition matrix
        # For each term r and each (i,j,k) entry, compute the contribution
        # T[r, (i,j,k)] = U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
        
        num_entries = n * m * p  # 48 entries
        T = np.zeros((rank44, num_entries))
        
        entry_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank44):
                        T[r, entry_idx] = U44[r, i*m+j] * V44[r, j*p+k] * W44[r, k*n+i]
                    entry_idx += 1
        
        # Target tensor
        target = np.zeros(num_entries)
        entry_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[entry_idx] = 1.0 if j == k else 0.0
                    entry_idx += 1
        
        # Check that rank-44 solution works
        recon = T.T @ np.ones(rank44)
        if not np.allclose(recon, target, atol=1e-10):
            return None
        
        def try_subset(selected_terms):
            """Try to solve for a subset of terms."""
            if len(selected_terms) >= rank44:
                return None
            T_rem = T[selected_terms, :]
            try:
                c, residuals, rank, sv = np.linalg.lstsq(T_rem, target, rcond=None)
                recon = T_rem.T @ c
                if np.allclose(recon, target, atol=1e-10):
                    c_snapped = np.round(c * 2) / 2
                    if np.allclose(c_snapped, c, atol=1e-8):
                        U_new = U44[selected_terms, :].copy()
                        V_new = V44[selected_terms, :].copy()
                        W_new = W44[selected_terms, :].copy()
                        
                        U_new = U_new * c_snapped[:, np.newaxis]
                        V_new = V_new * c_snapped[:, np.newaxis]
                        W_new = W_new * c_snapped[:, np.newaxis]
                        
                        if verify_solution(U_new, V_new, W_new):
                            return to_output_format(U_new, V_new, W_new)
            except Exception:
                pass
            return None
        
        # Strategy 1: Try removing single terms
        for term_to_remove in range(rank44):
            if time.time() - start_time > timeout:
                break
            remaining = [r for r in range(rank44) if r != term_to_remove]
            result = try_subset(remaining)
            if result is not None:
                return result
        
        # Strategy 2: Try removing pairs of terms
        for term1 in range(rank44):
            if time.time() - start_time > timeout:
                break
            for term2 in range(term1 + 1, rank44):
                if time.time() - start_time > timeout:
                    break
                remaining = [r for r in range(rank44) if r != term1 and r != term2]
                result = try_subset(remaining)
                if result is not None:
                    return result
        
        # Strategy 3: Try removing triples of terms
        for term1 in range(rank44):
            if time.time() - start_time > timeout:
                break
            for term2 in range(term1 + 1, rank44):
                if time.time() - start_time > timeout:
                    break
                for term3 in range(term2 + 1, rank44):
                    if time.time() - start_time > timeout:
                        break
                    remaining = [r for r in range(rank44) if r != term1 and r != term2 and r != term3]
                    result = try_subset(remaining)
                    if result is not None:
                        return result
        
        # Strategy 4: Try random subsets of size R for R in 41, 42, 43
        for R in [41, 42, 43]:
            if time.time() - start_time > timeout:
                break
            max_trials = 50 if R >= 42 else 30
            for trial in range(max_trials):
                if time.time() - start_time > timeout:
                    break
                np.random.seed(5000 + R * 100 + trial)
                selected = np.random.choice(rank44, R, replace=False)
                result = try_subset(list(selected))
                if result is not None:
                    return result
        
        return None
    
    # First, verify the rank-44 solution works
    U44, V44, W44 = get_rank44_solution()
    if verify_solution(U44, V44, W44):
        fallback_result = to_output_format(U44, V44, W44)
    else:
        # Should not happen, but provide a default
        fallback_result = to_output_format(U44, V44, W44)
    
    # Try structured reduction first (faster and more likely to succeed)
    result = try_structured_reduction()
    if result is not None:
        return result
    
    # Try numerical optimization for lower ranks (less likely to find exact solutions)
    for R in range(41, 44):
        if time.time() - start_time > timeout:
            break
        result = try_rank_optimization(R, max_restarts=2)
        if result is not None:
            return result
    
    # Return the rank-44 fallback
    return fallback_result
# EVOLVE-BLOCK-END
