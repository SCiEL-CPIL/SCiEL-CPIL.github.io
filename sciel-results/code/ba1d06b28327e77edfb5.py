"""QAOA angle design: graph-dependent optimization with deadline guard."""
import numpy as np
import time
from itertools import product

# Precompute optimal MaxCut values for small graphs if needed
# The evaluator will call angles(n, edges, p) for each graph.

def _exhaustive_maxcut(n, edges):
    """Compute exact MaxCut for small n via brute force."""
    if n <= 20:
        best = 0
        # Use bit representation
        for mask in range(1 << n):
            cut = 0
            for (u, v) in edges:
                if ((mask >> u) & 1) != ((mask >> v) & 1):
                    cut += 1
            if cut > best:
                best = cut
        return best
    else:
        # Fallback: greedy heuristic
        # For n > 20, use a simple greedy
        return _greedy_maxcut(n, edges)

def _greedy_maxcut(n, edges):
    """Simple greedy MaxCut heuristic."""
    assignment = np.zeros(n, dtype=int)
    # Order vertices by degree
    deg = np.zeros(n, dtype=int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    order = np.argsort(-deg)
    for v in order:
        # Assign to maximize cut given neighbors already assigned
        cut0 = 0
        cut1 = 0
        for u, w in edges:
            if u == v:
                other = w
            elif w == v:
                other = u
            else:
                continue
            if assignment[other] == 0:
                cut1 += 1
            else:
                cut0 += 1
        assignment[v] = 1 if cut1 >= cut0 else 0
    cut = 0
    for u, v in edges:
        if assignment[u] != assignment[v]:
            cut += 1
    return cut

def _expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut size for a 2-layer QAOA circuit.
    State: |+>^n, then for each layer: exp(-i gamma C) exp(-i beta sum X)
    C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2 = |E|/2 - 1/2 sum Z_u Z_v
    So exp(-i gamma C) = exp(-i gamma |E|/2) * exp(i gamma/2 sum Z_u Z_v)
    The global phase doesn't matter for expectation values.
    
    We need <Z_u Z_v> after the full circuit for each edge (u,v).
    
    For p=2, the circuit is:
    U = exp(-i beta2 sum X) exp(-i gamma2 C) exp(-i beta1 sum X) exp(-i gamma1 C)
    
    This is complex to compute exactly. Let's use the known formula for 2-layer QAOA.
    
    For a 2-layer QAOA, the expected cut can be computed by tracking the state.
    Since n is small (6, 8, 10), we can actually simulate the full state vector.
    
    State vector has dimension 2^n. For n=10, that's 1024, which is fine.
    """
    dim = 1 << n
    # Start in |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute X and C operators? That's expensive for n=10.
    # Instead, apply gates one by one.
    
    # exp(-i gamma C) where C = sum_{(u,v)} (1 - Z_u Z_v)/2
    # = exp(-i gamma |E|/2) * product over edges of exp(i gamma/2 Z_u Z_v)
    # Since Z_u Z_v is diagonal, exp(i gamma/2 Z_u Z_v) is diagonal with entries
    # exp(i gamma/2 * z_u z_v) where z_u, z_v are +/-1.
    
    # Actually, let's just build the C operator as a diagonal and apply it.
    
    def apply_gamma(gamma):
        # C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2
        # Diagonal of C: for basis state |b>, C|b> = sum_{(u,v)} (1 - (-1)^{b_u} (-1)^{b_v})/2 |b>
        # = sum_{(u,v)} (1 - (-1)^{b_u + b_v})/2
        # If b_u == b_v: (1-1)/2 = 0
        # If b_u != b_v: (1-(-1))/2 = 1
        # So C|b> = (number of cut edges in partition b) |b>
        diag = np.zeros(dim, dtype=np.complex128)
        for b in range(dim):
            cut_count = 0
            for u, v in edges:
                bu = (b >> u) & 1
                bv = (b >> v) & 1
                if bu != bv:
                    cut_count += 1
            diag[b] = np.exp(-1j * gamma * cut_count)
        psi *= diag
        return psi
    
    def apply_beta(beta):
        # exp(-i beta sum X) = product over i of exp(-i beta X_i)
        # Each exp(-i beta X_i) is a rotation on qubit i.
        # X = [[0,1],[1,0]], exp(-i beta X) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        # Apply to each qubit.
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        # For each qubit i, apply the 2x2 rotation.
        for i in range(n):
            # Split psi into blocks based on qubit i
            block_size = 1 << i
            # Reshape to (dim // (2*block_size), 2, block_size)
            psi_reshaped = psi.reshape(-1, 2, block_size)
            # Apply rotation: new[0] = c*old[0] + s*old[1], new[1] = s*old[0] + c*old[1]
            psi_reshaped[:, 0, :] = c * psi_reshaped[:, 0, :] + s * psi_reshaped[:, 1, :]
            psi_reshaped[:, 1, :] = s * psi_reshaped[:, 0, :] + c * psi_reshaped[:, 1, :]
            # Wait, this is wrong because we're modifying in place. Let me redo.
            # Actually, let's use a copy.
            psi_reshaped = psi.reshape(-1, 2, block_size)
            new_psi_reshaped = np.empty_like(psi_reshaped)
            new_psi_reshaped[:, 0, :] = c * psi_reshaped[:, 0, :] + s * psi_reshaped[:, 1, :]
            new_psi_reshaped[:, 1, :] = s * psi_reshaped[:, 0, :] + c * psi_reshaped[:, 1, :]
            psi = new_psi_reshaped.reshape(dim)
        return psi
    
    # Apply layers: gamma1, beta1, gamma2, beta2
    psi = apply_gamma(gammas[0])
    psi = apply_beta(betas[0])
    psi = apply_gamma(gammas[1])
    psi = apply_beta(betas[1])
    
    # Compute expected cut: sum_{(u,v) in edges} <Z_u Z_v>/2 ... wait
    # Expected cut = sum_{(u,v) in edges} P(Z_u != Z_v) = sum_{(u,v)} (1 - <Z_u Z_v>)/2
    # <Z_u Z_v> = sum_b |psi_b|^2 * (-1)^{b_u + b_v}
    expected_cut = 0.0
    for u, v in edges:
        zz = 0.0
        for b in range(dim):
            bu = (b >> u) & 1
            bv = (b >> v) & 1
            zz += (np.abs(psi[b])**2) * ((-1)**(bu + bv))
        expected_cut += (1 - zz) / 2.0
    return expected_cut

def _expected_cut_vectorized(n, edges, gammas, betas):
    """Vectorized version for speed."""
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cut counts for each basis state
    cut_counts = np.zeros(dim, dtype=np.int32)
    for b in range(dim):
        cc = 0
        for u, v in edges:
            if ((b >> u) & 1) != ((b >> v) & 1):
                cc += 1
        cut_counts[b] = cc
    
    # Precompute Z_u Z_v values for each edge
    zz_vals = {}
    for u, v in edges:
        zz = np.zeros(dim, dtype=np.int32)
        for b in range(dim):
            bu = (b >> u) & 1
            bv = (b >> v) & 1
            zz[b] = 1 if bu == bv else -1
        zz_vals[(u, v)] = zz
    
    def apply_gamma(gamma):
        nonlocal psi
        psi *= np.exp(-1j * gamma * cut_counts)
    
    def apply_beta(beta):
        nonlocal psi
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        for i in range(n):
            block_size = 1 << i
            psi_r = psi.reshape(-1, 2, block_size)
            new_psi_r = np.empty_like(psi_r)
            new_psi_r[:, 0, :] = c * psi_r[:, 0, :] + s * psi_r[:, 1, :]
            new_psi_r[:, 1, :] = s * psi_r[:, 0, :] + c * psi_r[:, 1, :]
            psi = new_psi_r.reshape(dim)
    
    apply_gamma(gammas[0])
    apply_beta(betas[0])
    apply_gamma(gammas[1])
    apply_beta(betas[1])
    
    prob = np.abs(psi)**2
    expected_cut = 0.0
    for u, v in edges:
        zz = zz_vals[(u, v)]
        expected_cut += np.sum(prob * zz) * (-1) / 2.0 + 0.5
    return expected_cut

def _objective(gammas, betas, n, edges, opt_cut):
    """Objective: expected_cut / opt_cut. Higher is better."""
    try:
        ec = _expected_cut_vectorized(n, edges, gammas, betas)
        if opt_cut == 0:
            return 0.0
        return ec / opt_cut
    except:
        return 0.0

def _optimize_angles(n, edges, p, opt_cut, deadline):
    """Optimize angles using a combination of random search and local refinement."""
    start_time = time.time()
    
    # Initial guess
    best_params = np.array([0.5, 0.3, 0.4, 0.2])  # gamma1, beta1, gamma2, beta2
    best_score = _objective(best_params[:2], best_params[2:], n, edges, opt_cut)
    
    # Random restarts
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        # Random initialization
        params = np.random.uniform(0, np.pi, 4)
        score = _objective(params[:2], params[2:], n, edges, opt_cut)
        if score > best_score:
            best_params = params.copy()
            best_score = score
    
    # Local refinement using Nelder-Mead
    from scipy.optimize import minimize
    
    def neg_obj(x):
        g1, b1, g2, b2 = x
        return -_objective([g1, g2], [b1, b2], n, edges, opt_cut)
    
    # Try multiple local optimizations from different starting points
    for r in range(5):
        if time.time() > deadline - 5:
            break
        if r == 0:
            x0 = best_params.copy()
        else:
            x0 = np.random.uniform(0, np.pi, 4)
        
        result = minimize(neg_obj, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < -best_score:
            best_params = result.x.copy()
            best_score = -result.fun
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 2:
        def neg_obj_tight(x):
            g1, b1, g2, b2 = x
            return -_objective([g1, g2], [b1, b2], n, edges, opt_cut)
        
        result = minimize(neg_obj_tight, best_params, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < -best_score:
            best_params = result.x.copy()
            best_score = -result.fun
    
    # Extract gammas and betas
    gammas = [best_params[0], best_params[2]]
    betas = [best_params[1], best_params[3]]
    
    # Build output array
    angles_out = np.zeros((p, 2))
    angles_out[0, 0] = gammas[0]
    angles_out[0, 1] = betas[0]
    angles_out[1, 0] = gammas[1]
    angles_out[1, 1] = betas[1]
    
    return angles_out

# Cache for optimal cuts
_opt_cut_cache = {}

def angles(n, edges, p):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    # Get optimal cut
    cache_key = (n, tuple(sorted(edges)))
    if cache_key not in _opt_cut_cache:
        _opt_cut_cache[cache_key] = _exhaustive_maxcut(n, edges)
    opt_cut = _opt_cut_cache[cache_key]
    
    # Optimize angles
    result = _optimize_angles(n, edges, p, opt_cut, deadline)
    
    return result
