import numpy as np
import time

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    dim = 1 << n
    psi0 = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute cost values (number of cut edges) for each basis state
    # H_C = sum (1 - Z_u Z_v) / 2. Eigenvalue is number of cut edges.
    costs = np.zeros(dim, dtype=float)
    for state in range(dim):
        cut_count = 0
        for u, v in edges:
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut_count += 1
        costs[state] = cut_count
    
    # Identify the optimal subspace (states with maximum cut)
    c_max = np.max(costs)
    opt_indices = np.where(costs == c_max)[0]
    n_opt = len(opt_indices)
    
    def _apply_mixer(psi, beta, n):
        """Apply exp(-i beta * sum X_i) efficiently."""
        c = np.cos(beta)
        s = np.sin(beta)
        # X_i commute, so we can apply single-qubit rotations sequentially
        for i in range(n):
            bit = 1 << i
            # Iterate over states where bit i is 0
            # Step by 2 * bit to jump over pairs
            for state in range(dim):
                if (state & bit) == 0:
                    partner = state | bit
                    a0 = psi[state]
                    a1 = psi[partner]
                    psi[state] = c * a0 - 1j * s * a1
                    psi[partner] = -1j * s * a0 + c * a1
        return psi
    
    def _qaoa_state(gammas, betas):
        """Compute QAOA state for given angles."""
        psi = psi0.copy()
        for k in range(p):
            # Apply cost unitary: exp(-i gamma_k * H_C)
            # H_C is diagonal with eigenvalues costs[state]
            psi = psi * np.exp(-1j * gammas[k] * costs)
            # Apply mixer unitary
            psi = _apply_mixer(psi, betas[k], n)
        return psi
    
    def _fidelity(psi):
        """Compute fidelity with the optimal subspace state."""
        # The target state is the uniform superposition of all max-cut states.
        # Fidelity = |<psi_opt|psi>|^2 = (1/N_opt) * sum(|psi_i|^2) for i in opt_indices
        return np.sum(np.abs(psi[opt_indices])**2) / n_opt
    
    def objective(params):
        gammas = params[0:p]
        betas = params[p:2*p]
        psi = _qaoa_state(gammas, betas)
        return -_fidelity(psi)
    
    # Simple Nelder-Mead implementation to avoid scipy overhead and allow strict time control
    def nelder_mead(func, x0, max_iter=200, timeout=10.0):
        x = np.array(x0, dtype=float)
        n = len(x)
        simplex = [x.copy()]
        for i in range(n):
            new_x = x.copy()
            new_x[i] = new_x[i] + 0.1 if new_x[i] != 0 else 0.1
            simplex.append(new_x)
        
        fvals = [func(s) for s in simplex]
        best_val = min(fvals)
        best_x = simplex[fvals.index(best_val)]
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        for _ in range(max_iter):
            if time.time() > start_time + timeout:
                break
            
            # Sort simplex
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + sigma * (xr - centroid)
                    fc = func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        break # Shrink
                else:
                    # Inside contraction
                    xc = centroid - sigma * (centroid - simplex[-1])
                    fc = func(xc)
                    if fc <= fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        break # Shrink
            
            if time.time() > start_time + timeout:
                break
            
            if min(fvals) < best_val:
                best_val = min(fvals)
                best_x = simplex[fvals.index(best_val)]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < 1e-8:
                break
                
            # Shrink if needed (simplified: just break if we stalled)
            # For speed, we rely on max_iter and timeout
            
        return best_x, best_val

    # Initial guesses
    initial_guesses = [
        [0.5, 0.5, 0.5, 0.5],
        [0.3, 0.3, 0.3, 0.3],
        [0.7, 0.4, 0.7, 0.4],
        [0.2, 0.6, 0.2, 0.6],
        [1.0, 0.5, 1.0, 0.5],
        [0.4, 0.2, 0.4, 0.2],
        [0.1, 0.1, 0.1, 0.1],
        [0.8, 0.8, 0.8, 0.8],
    ]
    
    best_params = None
    best_val = np.inf
    
    # Allocate time per restart
    time_per_restart = (deadline - time.time()) / len(initial_guesses)
    
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time()
        # Limit each restart to a fraction of remaining time
        current_timeout = min(remaining_time * 0.8, 10.0)
        
        try:
            res_x, res_val = nelder_mead(objective, x0, max_iter=100, timeout=current_timeout)
            if res_val < best_val:
                best_val = res_val
                best_params = res_x
        except Exception:
            continue
    
    if best_params is None:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Final polish if time remains
    if time.time() < deadline - 2:
        try:
            res_x, res_val = nelder_mead(objective, best_params, max_iter=50, timeout=deadline - time.time() - 1)
            if res_val < best_val:
                best_val = res_val
                best_params = res_x
        except Exception:
            pass
    
    gammas = best_params[0:p]
    betas = best_params[p:2*p]
    return np.array(list(zip(gammas, betas)))
