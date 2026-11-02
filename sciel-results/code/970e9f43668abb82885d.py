import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    total_size = nm + mp + pn
    
    def compute_residual_tensor(U, V, W, R):
        """
        U: shape (nm, R)
        V: shape (mp, R)
        W: shape (pn, R)
        Returns: (n, m, p) tensor of residuals
        """
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We want T[i,j,k] == 1 for all i,j,k
        T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
        return T - 1.0

    def residual_function(params, R):
        """
        params: flat vector of length R * (nm + mp + pn)
        Returns: flat vector of residuals
        """
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        res = compute_residual_tensor(U, V, W, R)
        return res.flatten()

    def snap_to_half_integer(arr):
        """Snap array to nearest half-integers (multiples of 0.5)."""
        return np.round(arr * 2) / 2.0

    def verify_exact(U, V, W, R):
        """
        Verify if U, V, W (assumed to be half-integers) exactly decompose the tensor.
        Returns True if valid.
        """
        # Scale by 2 to work with integers for exactness check
        U_int = (U * 2).astype(np.int64)
        V_int = (V * 2).astype(np.int64)
        W_int = (W * 2).astype(np.int64)
        
        U3 = U_int.reshape(n, m, R)
        V3 = V_int.reshape(m, p, R)
        W3 = W_int.reshape(p, n, R)
        
        # Product in integer domain: sum (2U)(2V)(2W) = 4 * (sum UVW)
        # We want sum UVW = 1, so sum (2U)(2V)(2W) should be 4
        T_int = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
        
        # Check if all entries are exactly 4
        return np.all(T_int == 4)

    def try_strassen_based_construction(R):
        """
        Try to construct a rank-R decomposition using Strassen-like algebraic structure.
        For 2x3 * 3x5, we can try to exploit the structure.
        Returns (U, V, W, success) or None
        """
        # Strategy: Use the fact that we can decompose the 2x3 matrix into blocks
        # and use algebraic identities to reduce rank.
        
        # For rank 28, try a structured approach:
        # Start with schoolbook (30 terms) and try to merge/cancel pairs.
        
        if R < 27 or R > 30:
            return None
        
        # Initialize with schoolbook
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Fill first min(R, 30) columns with schoolbook
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if r < R:
                        U[i*m+j, r] = 1.0
                        V[j*p+k, r] = 1.0
                        W[k*n+i, r] = 1.0
                        r += 1
        
        if r < R:
            # Remaining columns are zero, which is fine
            pass
            
        return U, V, W, False

    def lattice_local_search(U, V, W, R, max_iters=200, timeout=None, start_time=0):
        """
        Perform greedy local search on the half-integer lattice.
        U, V, W are arrays of half-integers.
        Returns (U, V, W, success)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        def get_res_sq():
            res = compute_residual_tensor(U, V, W, R)
            return np.sum(res**2)

        current_energy = get_res_sq()
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_energy = current_energy
        
        n_vars_U = nm * R
        n_vars_V = mp * R
        n_vars_W = pn * R
        total_vars = n_vars_U + n_vars_V + n_vars_W
        
        # Precompute indices
        # We will iterate through variables and try +/- 0.5
        
        for it in range(max_iters):
            if timeout and time.time() - start_time > timeout:
                break
                
            improved = False
            
            # Shuffle variable order to avoid bias
            indices = np.random.permutation(total_vars)
            
            for idx in indices:
                if timeout and time.time() - start_time > timeout:
                    break
                    
                orig_val = None
                # Identify which array and index
                if idx < n_vars_U:
                    u_idx = idx
                    orig_val = U[u_idx]
                elif idx < n_vars_U + n_vars_V:
                    v_idx = idx - n_vars_U
                    orig_val = V[v_idx]
                else:
                    w_idx = idx - n_vars_U - n_vars_V
                    orig_val = W[w_idx]
                
                # Try +0.5
                if idx < n_vars_U:
                    U[u_idx] = orig_val + 0.5
                elif idx < n_vars_U + n_vars_V:
                    V[v_idx] = orig_val + 0.5
                else:
                    W[w_idx] = orig_val + 0.5
                    
                new_energy = get_res_sq()
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                    # Keep the change
                else:
                    # Try -0.5
                    if idx < n_vars_U:
                        U[u_idx] = orig_val - 0.5
                    elif idx < n_vars_U + n_vars_V:
                        V[v_idx] = orig_val - 0.5
                    else:
                        W[w_idx] = orig_val - 0.5
                        
                    new_energy = get_res_sq()
                    
                    if new_energy < current_energy:
                        current_energy = new_energy
                        improved = True
                    else:
                        # Revert
                        if idx < n_vars_U:
                            U[u_idx] = orig_val
                        elif idx < n_vars_U + n_vars_V:
                            V[v_idx] = orig_val
                        else:
                            W[w_idx] = orig_val
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_energy < 1e-12:
                        return best_U, best_V, best_W, True
                        
            if not improved:
                # If no improvement in a full pass, try a random perturbation
                # to escape local minima
                if it < max_iters - 1:
                    # Perturb a few random variables
                    n_perturb = min(5, total_vars)
                    for _ in range(n_perturb):
                        idx = np.random.randint(total_vars)
                        delta = np.random.choice([-1.0, 1.0]) * 0.5
                        
                        if idx < n_vars_U:
                            U[idx] += delta
                        elif idx < n_vars_U + n_vars_V:
                            V[idx - n_vars_U] += delta
                        else:
                            W[idx - n_vars_U - n_vars_V] += delta
                    
                    new_energy = get_res_sq()
                    if new_energy < best_energy + 0.1:  # Allow some uphill moves
                        current_energy = new_energy
                        # Continue searching
                    else:
                        # Revert perturbation
                        break

        if best_energy < 1e-12:
            return best_U, best_V, best_W, True
            
        return U, V, W, False

    def solve_for_rank(R, start_time, deadline):
        """
        Try to find a rank-R decomposition.
        Returns (U, V, W, success)
        """
        n_attempts = 15
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
                
            # Multiple initialization strategies
            if attempt == 0:
                # Start from schoolbook-like structure
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                
                # Fill with schoolbook pattern where possible
                r = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if r < R:
                                U[i*m+j, r] = 1.0
                                V[j*p+k, r] = 1.0
                                W[k*n+i, r] = 1.0
                                r += 1
                                
                # Add small random perturbation to break symmetry
                U += np.random.normal(0, 0.1, U.shape)
                V += np.random.normal(0, 0.1, V.shape)
                W += np.random.normal(0, 0.1, W.shape)
                
            elif attempt == 1:
                # Pure random initialization
                U = np.random.normal(0, 1.0, (nm, R))
                V = np.random.normal(0, 1.0, (mp, R))
                W = np.random.normal(0, 1.0, (pn, R))
                
            elif attempt == 2:
                # Sparse initialization
                U = np.random.choice([0, 0.5, -0.5, 1, -1], size=(nm, R))
                V = np.random.choice([0, 0.5, -0.5, 1, -1], size=(mp, R))
                W = np.random.choice([0, 0.5, -0.5, 1, -1], size=(pn, R))
                
            else:
                # Random with different scales
                scale = 0.5 + 0.5 * attempt
                U = np.random.normal(0, scale, (nm, R))
                V = np.random.normal(0, scale, (mp, R))
                W = np.random.normal(0, scale, (pn, R))
            
            try:
                # Convert to flat vector for least_squares
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                # Use least_squares with Levenberg-Marquardt
                result = least_squares(residual_function, params, args=(R,), 
                                       method='lm', max_nfev=500, ftol=1e-12)
                
                if result.cost < 1e-3:
                    U_flat = result.x[:nm*R]
                    V_flat = result.x[nm*R:nm*R+mp*R]
                    W_flat = result.x[nm*R+mp*R:]
                    
                    U0 = U_flat.reshape(nm, R)
                    V0 = V_flat.reshape(mp, R)
                    W0 = W_flat.reshape(pn, R)
                    
                    # Snap to half-integers
                    U_s = snap_to_half_integer(U0)
                    V_s = snap_to_half_integer(V0)
                    W_s = snap_to_half_integer(W0)
                    
                    # Check if snapped is already valid
                    if verify_exact(U_s, V_s, W_s, R):
                        return U_s, V_s, W_s, True
                        
                    # Run local search on the lattice
                    if time.time() < deadline - 10:
                        U_ls, V_ls, W_ls, success = lattice_local_search(
                            U_s, V_s, W_s, R, 
                            max_iters=300, 
                            timeout=deadline, 
                            start_time=start_time
                        )
                        
                        if success or verify_exact(U_ls, V_ls, W_ls, R):
                            return U_ls, V_ls, W_ls, True
                            
                    # If close but not exact, try more aggressive local search
                    # from the unsnapped solution
                    if result.cost < 0.01 and time.time() < deadline - 15:
                        U_ls, V_ls, W_ls, success = lattice_local_search(
                            U0, V0, W0, R, 
                            max_iters=500, 
                            timeout=deadline, 
                            start_time=start_time
                        )
                        
                        if success or verify_exact(U_ls, V_ls, W_ls, R):
                            return U_ls, V_ls, W_ls, True
                            
            except Exception as e:
                continue
                
        return None, None, None, False

    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for return overhead
    
    # Try ranks from 28 down to 27 (29 is known to work, 30 is schoolbook)
    for R in [28, 27]:
        if time.time() > deadline:
            break
            
        remaining_time = deadline - time.time()
        if remaining_time < 15:
            break
            
        U, V, W, success = solve_for_rank(R, start_time, deadline)
        
        if success:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback: Schoolbook decomposition (Rank = n*m*p = 30)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
