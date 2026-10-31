import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50
RHO_ADM = 10.0  # ADMM penalty parameter

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def compute_conv_matrix_indices():
    """Precompute the mapping for autoconvolution as quadratic form in z.
    For h_i = |z_i|^2, the autoconvolution at lag k is sum_i h_i * h_{i+k}.
    We can compute this efficiently using the structure."""
    pass

def admm_optimize(h_init, deadline, max_iter=500):
    """ADMM optimization using complex vector parameterization h_i = |z_i|^2.
    
    We maximize h2_sq / conv_inf subject to sum(h) = 1, h >= 0.
    Using h_i = |z_i|^2, we optimize over z in C^N.
    
    The objective becomes: sum(|z_i|^4) / max_k sum_i |z_i|^2 * |z_{i+k}|^2
    """
    h_init = np.asarray(h_init, dtype=np.float64)
    h_init = np.maximum(h_init, 0.0)
    s = np.sum(h_init)
    if s < 1e-15:
        return h_init
    
    h_init = h_init / s
    
    # Initialize z: z_i = sqrt(h_i) (real positive)
    z = np.sqrt(h_init).astype(np.complex128)
    
    # ADMM variables
    # We want to maximize f(z) = sum(|z_i|^4) / max_k (sum_i |z_i|^2 * |z_{i+k}|^2)
    # This is equivalent to minimizing -f(z)
    # 
    # ADMM formulation:
    # min_z -sum(|z_i|^4) + t * rho/2 * (max_k conv_k(z) - t)^2 + ...
    # 
    # Simpler approach: alternate between:
    # 1. z-update: minimize -sum(|z_i|^4) + (rho/2)*|||z|^2 - u||^2
    # 2. u-update: project u onto {u >= 0, sum(u) = 1, max_k conv_k(u) <= t}
    # 3. t-update: t = max_k conv_k(u)
    
    u = h_init.copy()  # u = |z|^2
    t = np.max(fftconvolve(u, u, mode='full'))
    
    best_z = z.copy()
    best_score = compute_ratio(h_init)
    
    for it in range(max_iter):
        if time.time() > deadline - 15.0:
            break
        
        # z-update: minimize -sum(|z_i|^4) + (rho/2)*sum((|z_i|^2 - u_i)^2)
        # For each i: minimize -|z_i|^4 + (rho/2)*(|z_i|^2 - u_i)^2
        # Let w = |z_i|^2 >= 0: minimize -w^2 + (rho/2)*(w - u_i)^2
        # d/dw: -2w + rho*(w - u_i) = 0 => w*(rho - 2) = rho*u_i => w = rho*u_i/(rho-2) for rho > 2
        # But we need w >= 0, and the minimum is at w = max(0, rho*u_i/(rho-2))
        # However, this is a simplification. The actual z-update should preserve phase.
        # For real positive z, this works. Let's use a simpler proximal step.
        
        # Actually, let's use a different approach: 
        # z-update: z_i = sqrt(max(0, u_i)) * exp(i*arg(z_i)) (keep phase, update magnitude)
        # But this doesn't optimize the objective. Let's use gradient descent on z instead.
        
        # Gradient of -sum(|z_i|^4) wrt z_i: -2*|z_i|^2 * z_i
        # Gradient of (rho/2)*sum((|z_i|^2 - u_i)^2) wrt z_i: rho*(|z_i|^2 - u_i)*z_i
        
        # So: z_i <- z_i - alpha * (-2*|z_i|^2 * z_i + rho*(|z_i|^2 - u_i)*z_i)
        #    = z_i * (1 - alpha*(-2*|z_i|^2 + rho*(|z_i|^2 - u_i)))
        
        # This is getting complex. Let's use a simpler alternating scheme:
        # 1. Given u, update z to minimize -sum(|z_i|^4) + (rho/2)*sum((|z_i|^2 - u_i)^2)
        #    For each i independently: minimize -w^2 + (rho/2)*(w-u_i)^2 where w = |z_i|^2
        #    Optimal: w* = max(0, u_i * rho / (rho - 2)) if rho > 2, else 0
        #    But this ignores that we want to keep z complex. For real z, this works.
        
        # Let's try a different ADMM formulation:
        # Minimize g(z) = -sum(|z_i|^4) subject to |z|^2 = u, sum(u)=1, u>=0
        # Lagrangian: L = -sum(|z_i|^4) + (rho/2)*sum((|z_i|^2 - u_i)^2) + sum(lambda_i*(|z_i|^2 - u_i))
        
        # Simpler: just do coordinate descent on h directly with better initialization.
        
        # Let's abandon pure ADMM and use a hybrid: ADMM-like updates with gradient steps.
        
        # Compute current h from z
        h_current = np.abs(z)**2
        h_current = np.maximum(h_current, 0)
        s_cur = np.sum(h_current)
        if s_cur < 1e-15:
            break
        h_current = h_current / s_cur
        
        # Compute score
        score_cur = compute_ratio(h_current)
        if score_cur > best_score:
            best_score = score_cur
            best_z = z.copy()
        
        # Check convergence
        if it > 10 and abs(score_cur - best_score) < 1e-10:
            if it > 50:
                break
        
        # u-update: project h_current onto constraint set {sum(u)=1, u>=0, max_conv(u) <= t}
        # This is a quadratic constraint. Use simple projection:
        # 1. Normalize: u = h_current / sum(h_current)
        # 2. Clip: u = max(u, 0)
        # 3. Renormalize
        u = np.maximum(h_current, 0)
        s_u = np.sum(u)
        if s_u > 1e-15:
            u = u / s_u
        
        # t-update
        conv_u = fftconvolve(u, u, mode='full')
        t = np.max(conv_u)
        
        # z-update: z_i = sqrt(u_i) * phase(z_i)
        # But this doesn't improve the objective. Let's add a gradient step.
        
        # Gradient of -sum(|z_i|^4) wrt z: -2*|z_i|^2*z_i
        # Step: z -= alpha * grad
        alpha = 0.1
        grad = -2 * np.abs(z)**2 * z
        z = z - alpha * grad
        
        # Keep z in a reasonable range
        z_mag = np.abs(z)
        if np.max(z_mag) > 10:
            z = z * (10 / np.max(z_mag))
    
    # Final h from best_z
    h_final = np.abs(best_z)**2
    h_final = np.maximum(h_final, 0)
    s_final = np.sum(h_final)
    if s_final > 1e-15:
        h_final = h_final / s_final
    
    return h_final

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def greedy_sidon_search(indices, deadline):
    """Greedy heuristic with random restarts to find a large Sidon subset of indices.
    Returns the best Sidon set found."""
    best_sidon = set()
    for restart in range(50):
        if time.time() > deadline - 20.0:
            break
        # Random ordering of candidates
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        current = set()
        for idx in shuffled:
            # Check if adding idx preserves Sidon property
            test_set = list(current) + [idx]
            if is_sidon_set(test_set):
                current.add(idx)
        if len(current) > len(best_sidon):
            best_sidon = set(current)
    return list(best_sidon)

def local_search_polish(h, deadline):
    """Local search: try swapping in/out indices and adjusting weights.
    Recomputes nonzero set each iteration to avoid stale-state bugs."""
    h = h.copy()
    improved = True
    max_iter = 200
    for _ in range(max_iter):
        if time.time() > deadline - 12.0:
            break
        if not improved:
            break
        improved = False
        current_score = compute_ratio(h)
        
        # Recompute nonzero set each iteration
        nonzero_idx = set(np.where(h > 1e-10)[0])
        zero_idx = [i for i in range(N) if i not in nonzero_idx]
        
        # Try adding a new index with small weight
        for idx in zero_idx:
            if time.time() > deadline - 12.0:
                break
            h_test = h.copy()
            h_test[idx] = 0.01
            sc = compute_ratio(h_test)
            if sc > current_score + 1e-10:
                h = h_test
                current_score = sc
                improved = True
                break
        
        # Try removing an index
        if not improved:
            nonzero_list = list(nonzero_idx)
            for idx in nonzero_list:
                if time.time() > deadline - 12.0:
                    break
                h_test = h.copy()
                h_test[idx] = 0.0
                sc = compute_ratio(h_test)
                if sc > current_score + 1e-10:
                    h = h_test
                    current_score = sc
                    improved = True
                    break
        
        # Try adjusting weights of existing indices
        if not improved:
            nonzero_list = list(nonzero_idx)
            for idx in nonzero_list:
                if time.time() > deadline - 12.0:
                    break
                for factor in [0.5, 0.8, 1.2, 1.5, 2.0]:
                    h_test = h.copy()
                    h_test[idx] *= factor
                    sc = compute_ratio(h_test)
                    if sc > current_score + 1e-10:
                        h = h_test
                        current_score = sc
                        improved = True
                        break
                if improved:
                    break
                    
    return h

def gradient_polish(h, deadline):
    """Smooth gradient-based polish using L-BFGS-B on the full vector.
    Uses log-sum-exp smoothing for the max in the autoconvolution."""
    h = h.copy()
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    def neg_ratio_smooth(x, beta=50.0):
        x = np.maximum(x, 0.0)
        s = np.sum(x)
        if s < 1e-15:
            return 1e10
        x = x / s
        h2 = np.sum(x**2)
        conv = fftconvolve(x, x, mode='full')
        # Smooth max via log-sum-exp
        conv_clipped = np.clip(conv, 1e-15, None)
        logsumexp_val = np.log(np.sum(np.exp(beta * conv_clipped))) / beta
        if logsumexp_val < 1e-15:
            return 1e10
        return -h2 / logsumexp_val
    
    def neg_ratio_exact(x):
        x = np.maximum(x, 0.0)
        s = np.sum(x)
        if s < 1e-15:
            return 1e10
        x = x / s
        h2 = np.sum(x**2)
        conv = fftconvolve(x, x, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf
    
    best_h = h.copy()
    best_sc = compute_ratio(h)
    
    # Try multiple beta values for smoothing
    for beta in [50.0, 20.0, 10.0]:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(neg_ratio_smooth, h, args=(beta,),
                          method='L-BFGS-B',
                          bounds=[(0.0, None)] * N,
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            h_try = np.maximum(res.x, 0.0)
            s_try = np.sum(h_try)
            if s_try > 1e-15:
                h_try = h_try / s_try
                sc_try = compute_ratio(h_try)
                if sc_try > best_sc:
                    best_sc = sc_try
                    best_h = h_try.copy()
        except Exception:
            pass
    
    # Final exact polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            res = minimize(neg_ratio_exact, best_h,
                          method='L-BFGS-B',
                          bounds=[(0.0, None)] * N,
                          options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            h_try = np.maximum(res.x, 0.0)
            s_try = np.sum(h_try)
            if s_try > 1e-15:
                h_try = h_try / s_try
                sc_try = compute_ratio(h_try)
                if sc_try > best_sc:
                    best_sc = sc_try
                    best_h = h_try.copy()
        except Exception:
            pass
    
    return best_h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin: kill at 150s, return by 140s

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []

    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))

    np.random.seed(42)

    # Phase 1: Addition Graph Maximum Independent Set (Sidon Set) Search
    # Find the largest subset of [0, N) that forms a Sidon set
    all_indices = list(range(N))
    
    # Multiple random restarts for greedy Sidon search
    best_sidon = []
    for restart in range(20):
        if time.time() > deadline - 60.0:
            break
        sidon_set = greedy_sidon_search(all_indices, deadline)
        if len(sidon_set) > len(best_sidon):
            best_sidon = sidon_set
    
    # Also try some known good Sidon constructions
    known_sidon_patterns = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 3, 7, 15, 31],
        [0, 2, 5, 11, 23],
        [0, 1, 4, 9, 16, 25, 36, 49],
    ]
    
    all_sidon_candidates = [best_sidon] + known_sidon_patterns
    
    # Evaluate all Sidon candidates
    for sidon_set in all_sidon_candidates:
        if time.time() > deadline - 50.0:
            break
        if len(sidon_set) < 2:
            continue
        h = np.zeros(N)
        for idx in sidon_set:
            if 0 <= idx < N:
                h[idx] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Phase 2: Random sparse search for additional candidates
    num_sparse_tries = 500
    for _ in range(num_sparse_tries):
        if time.time() > deadline - 40.0:
            break
        
        k = np.random.randint(3, 14)
        indices = np.random.choice(N, k, replace=False)
        weights = np.random.rand(k)
        h = np.zeros(N)
        h[indices] = weights
        
        sc = compute_ratio(h)
        update_best(h, sc)
    
    # Phase 2.5: ADMM optimization from top candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_for_admm = candidates[:10]
    
    for i, (sc, h_start) in enumerate(top_for_admm):
        if time.time() > deadline - 25.0:
            break
        
        h_admm = admm_optimize(h_start, deadline, max_iter=200)
        sc_admm = compute_ratio(h_admm)
        update_best(h_admm, sc_admm)

    # Sort candidates by score and keep top K for local refinement
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = candidates[:25]
    
    # Phase 3: Local Optimization from Top Candidates
    def objective_neg_sparse(h_sparse, indices):
        h = np.zeros(N)
        h[indices] = h_sparse
        h = np.maximum(h, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        h2 = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf

    for i, (sc, h_start) in enumerate(top_k):
        if time.time() > deadline - 12.0:
            break
        
        indices = np.where(h_start > 1e-10)[0]
        if len(indices) == 0:
            continue
            
        h_sparse_start = h_start[indices]
        
        # Optimize weights for this fixed support
        maxiter_nm = 3000 if i == 0 else 1500
        res = minimize(objective_neg_sparse, h_sparse_start, args=(indices,), 
                       method='Nelder-Mead', 
                       options={'maxiter': maxiter_nm, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        
        h_opt_sparse = np.maximum(res.x, 0.0)
        h_opt = np.zeros(N)
        h_opt[indices] = h_opt_sparse
        
        s = np.sum(h_opt)
        if s > 1e-15:
            h_opt = h_opt / s
            sc_opt = compute_ratio(h_opt)
            update_best(h_opt, sc_opt)
            
        # Apply gradient polish to top candidates
        if i < 3:
            h_grad = gradient_polish(h_opt, deadline)
            sc_grad = compute_ratio(h_grad)
            update_best(h_grad, sc_grad)
        
        # Apply local search polish to the best candidate
        if i == 0:
            h_polished = local_search_polish(h_opt, deadline)
            sc_polished = compute_ratio(h_polished)
            update_best(h_polished, sc_polished)

    # Final gradient polish on the overall best
    if time.time() < deadline - 3.0:
        h_final = gradient_polish(best_h, deadline)
        sc_final = compute_ratio(h_final)
        if sc_final > best_score:
            best_h = h_final
    
    # Final ADMM polish
    if time.time() < deadline - 2.0:
        h_admm_final = admm_optimize(best_h, deadline, max_iter=100)
        sc_admm_final = compute_ratio(h_admm_final)
        if sc_admm_final > best_score:
            best_h = h_admm_final
    
    return list(best_h)
