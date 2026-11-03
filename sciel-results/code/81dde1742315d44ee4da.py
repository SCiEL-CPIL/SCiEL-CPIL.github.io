"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

Structural change:
  - Treat the problem as a combinatorial support-selection problem first.
  - Use a Sidon-like (B_2) greedy construction to find sparse supports with minimal
    convolution collisions.
  - Then apply a continuous epigraph-based optimization (SLSQP) to refine weights
    on the discovered supports, and a block-coordinate descent for final polish.
  - This combines the discrete structural search (from the current program's greedy
    Sidon idea) with the continuous optimization machinery (from the inspiration's
    epigraph formulation).

The key insight: for a sparse support S with |S| = k, the autoconvolution has
at most k(k+1)/2 distinct sum positions (if S is a Sidon set, all pairwise sums
are distinct). The peak of the autoconvolution is minimized when the support
has few collisions. We first find such supports, then optimize weights.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max <= 0:
        return 0.0
    return l2_sq / (l1 * autoconv_max)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)

    def update_best(h):
        nonlocal best_h, best_score
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # === Phase 1: Combinatorial Support Search via Sidon-like Greedy ===
    # Build sparse supports where pairwise index sums are as distinct as possible.
    # For a support S, the autoconvolution at lag k is sum_{i,j: i+j=k} h_i h_j.
    # If all pairwise sums are distinct (Sidon), each convolution bin has at most
    # one term (or two for symmetric pairs), minimizing the max.
    
    def greedy_sidon_support(max_size=12, seed=0):
        """Greedy construction of a Sidon-like support."""
        rng = np.random.RandomState(seed)
        h = np.zeros(N)
        # Start with a random anchor
        anchor = rng.randint(N)
        h[anchor] = 1.0
        support = [anchor]
        
        for _ in range(max_size - 1):
            if time.time() > deadline - 30:
                break
            best_idx = -1
            best_metric = -np.inf
            
            # Score: minimize the increase in max convolution when adding this index
            # Also prefer indices that create new distinct sum pairs
            for idx in range(N):
                if idx in support:
                    continue
                h_try = h.copy()
                h_try[idx] = 1.0
                conv = np.convolve(h_try, h_try, mode='full')
                max_conv = np.max(conv)
                # We want to minimize max_conv, so score = -max_conv
                # Also add a small bonus for distinct sums
                metric = -max_conv
                if metric > best_metric:
                    best_metric = metric
                    best_idx = idx
            
            if best_idx != -1:
                h[best_idx] = 1.0
                support.append(best_idx)
            else:
                break
        
        return h, support

    # Generate multiple Sidon-like supports with different seeds and sizes
    sidon_supports = []
    for seed in range(20):
        if time.time() > deadline - 40:
            break
        for max_size in [5, 8, 10, 12]:
            h_s, support = greedy_sidon_support(max_size, seed)
            if len(support) >= 3:
                sidon_supports.append((h_s, support, seed, max_size))

    # Also add some random sparse supports
    rng = np.random.RandomState(42)
    for _ in range(30):
        if time.time() > deadline - 40:
            break
        k = rng.randint(3, 13)
        idx = rng.choice(N, k, replace=False)
        h_r = np.zeros(N)
        h_r[idx] = 1.0
        sidon_supports.append((h_r, idx, -1, k))

    # === Phase 2: Epigraph Optimization on Discovered Supports ===
    # For each support, run SLSQP to optimize the weights.
    # Objective: minimize M (upper bound on max autoconvolution)
    # Subject to: sum(h) = 1, h >= 0, M >= conv_k for all k
    
    num_conv = 2 * N - 1

    def run_epigraph_on_support(support, h_init, maxiter=300, ftol=1e-12):
        """Run epigraph optimization for a fixed support."""
        if len(support) < 2:
            return None
        
        # Work in the subspace of the support
        # Variables: weights on support (len(support)) + M
        n_vars = len(support) + 1
        
        def objective(x):
            return x[-1]  # Minimize M
        
        def objective_grad(x):
            grad = np.zeros(n_vars)
            grad[-1] = 1.0
            return grad
        
        def constraint_eq_fun(x):
            return np.sum(x[:-1]) - 1.0
        
        def constraint_eq_jac(x):
            jac = np.zeros(n_vars)
            jac[:-1] = 1.0
            return jac
        
        def constraint_ineq_fun(x):
            w = x[:-1]
            M = x[-1]
            h_full = np.zeros(N)
            h_full[support] = w
            conv = np.convolve(h_full, h_full, mode='full')
            return M - conv  # M >= conv_k for all k
        
        def constraint_ineq_jac(x):
            w = x[:-1]
            jac = np.zeros((num_conv, n_vars))
            jac[:, -1] = 1.0  # dM/dM = 1
            # d(conv_k)/dw_i for each k, i
            for i, s_idx in enumerate(support):
                # conv_k = sum_{j: j + s_idx = k} w_j * w_{k - s_idx} ... 
                # Actually conv_k = sum_{a+b=k} h_a h_b
                # d(conv_k)/dw_i = sum_{b: i+b=k} h_b + sum_{a: a+i=k} h_a
                # = h_{k-i} + h_{k-i} if i != k-i, or 2*h_i if i == k-i
                # More carefully: conv_k = sum_{a=0}^{N-1} h_a * h_{k-a} (where defined)
                # d(conv_k)/dw_i = h_{k-i} + h_{k-i} if i != k-i else 2*h_i
                # But w_i = h_{support[i]}, so d(conv_k)/dw_i = d(conv_k)/dh_{support[i]}
                for k in range(num_conv):
                    j1 = k - s_idx
                    j2 = k - s_idx  # same
                    if 0 <= j1 < N:
                        jac[k, i] += h_full[j1]
                    if 0 <= j2 < N and j2 != j1:
                        # This double counts when j1 == j2
                        pass
                    # Simpler: just compute numerically or use the formula
                    # conv_k = sum_{a} h_a * h_{k-a}
                    # d/dh_i = h_{k-i} + h_{k-i} if i != k-i, 2*h_i if i == k-i
                    # But since we're summing over all a, and h appears twice:
                    # d(conv_k)/dh_i = h_{k-i} (from a=i term) + h_{k-i} (from a=k-i term) if i != k-i
                    # = 2 * h_i if i == k-i
                    # Let's just compute it properly:
                    pass
                # Recompute properly
            # For efficiency, compute the Jacobian columns properly
            for i, s_idx in enumerate(support):
                for k in range(num_conv):
                    # conv_k = sum_{a=0}^{N-1} h[a] * h[k-a] where 0<=k-a<N
                    # d(conv_k)/dh[s_idx] = h[k-s_idx] if 0<=k-s_idx<N 
                    #                       + h[k-s_idx] if 0<=k-s_idx<N (from the other term)
                    # Wait, conv is symmetric: conv_k = sum_{a} h_a h_{k-a}
                    # The derivative w.r.t. h_i is:
                    # sum over a where a=i: h_{k-i}
                    # sum over a where k-a=i i.e. a=k-i: h_i * 1 = h_i... 
                    # No: d/dh_i [h_a * h_{k-a}] = delta_{a,i} h_{k-a} + h_a delta_{k-a,i}
                    # = delta_{a,i} h_{k-a} + h_a delta_{a,k-i}
                    # Summing over a: h_{k-i} + h_{k-i} if i != k-i, else 2*h_i
                    # So d(conv_k)/dh_i = h_{k-i} + (h_i if k-i == i else h_{k-i})... 
                    # Actually: = h_{k-i} + h_{k-i} if i != k-i, 2*h_i if i == k-i
                    # But h_{k-i} when i == k-i is h_i, so it's 2*h_i. Consistent.
                    val = 0.0
                    j = k - s_idx
                    if 0 <= j < N:
                        val += h_full[j]
                    # The second term: when a = k - i, we get h_a * delta = h_{k-i}
                    # But this is the same as above unless k-i == i
                    if 0 <= j < N and j != s_idx:
                        val += h_full[j]
                    elif 0 <= j < N and j == s_idx:
                        val += h_full[j]  # Already counted once, add again for 2*h_i
                    # Hmm, this is getting confused. Let me just do it simply:
                    # d(conv_k)/dh_i = h_{k-i} if 0 <= k-i < N else 0
                    # PLUS h_{k-i} if 0 <= k-i < N and k-i != i (to avoid double counting)
                    # Actually the cleanest: 
                    # conv_k = sum_a h_a h_{k-a}
                    # d/dh_i = h_{k-i} (from a=i) + h_{k-i} (from a=k-i) if i != k-i
                    #        = 2 h_i (from a=i=k-i)
                    # So:
                    if 0 <= j < N:
                        if j == s_idx:
                            val = 2.0 * h_full[j]
                        else:
                            val = 2.0 * h_full[j]
                    # Wait no. If i != k-i, then a=i gives h_{k-i}, and a=k-i gives h_i * 1 = h_i... 
                    # No! d/dh_i [h_a h_{k-a}] with a fixed: 
                    # If a = i: derivative is h_{k-i}
                    # If a = k-i: derivative is h_a = h_{k-i}
                    # So total = 2 h_{k-i} if i != k-i, and 2 h_i if i == k-i.
                    # In both cases it's 2 * h_{k-i} (since when i=k-i, h_{k-i}=h_i).
                    # So d(conv_k)/dh_i = 2 * h_{k-i} if 0 <= k-i < N, else 0.
                    # But wait, that's not right either because conv is defined as:
                    # (f*f)(t) = integral f(s) f(t-s) ds
                    # In discrete: conv_k = sum_{a} h_a h_{k-a} where both indices in [0,N-1]
                    # d(conv_k)/dh_i = h_{k-i} (setting a=i) + h_{k-i} (setting k-a=i, i.e., a=k-i)
                    # = 2 h_{k-i} if i != k-i, 2 h_i if i == k-i.
                    # So yes, 2 * h_{k-i} if valid.
                    # Let me just recompute cleanly.
                    pass
            
            # Clean recomputation of Jacobian
            for i, s_idx in enumerate(support):
                for k in range(num_conv):
                    j = k - s_idx
                    if 0 <= j < N:
                        jac[k, i] = 2.0 * h_full[j]
                    else:
                        jac[k, i] = 0.0
            # But this double counts when s_idx == j (i.e., k = 2*s_idx)
            # In that case, the term h_{s_idx} * h_{s_idx} appears once in the sum,
            # and its derivative is 2*h_{s_idx}. So the formula 2*h_j is correct.
            # Actually no: in the sum sum_a h_a h_{k-a}, when a = s_idx, we get h_{s_idx} h_{k-s_idx}.
            # When k-s_idx = s_idx, this is h_{s_idx}^2, and d/dh_{s_idx} = 2 h_{s_idx}.
            # When k-s_idx != s_idx, the term h_{s_idx} h_{k-s_idx} appears once (at a=s_idx),
            # and also the term h_{k-s_idx} h_{s_idx} appears at a=k-s_idx.
            # So we have two terms: h_{s_idx} h_j and h_j h_{s_idx}, both giving h_j when differentiated.
            # Total: 2 h_j. So the formula is correct.
            return -jac  # Constraint is M - conv >= 0, so gradient is [dM/dx - dconv/dx]
            # But we defined constraint_ineq_fun as M - conv, so:
            # d/dx (M - conv) = [dM/dx] - [dconv/dx]
            # dM/dx = [0,0,...,0,1]
            # dconv/dx = [dconv/dw_1, ..., dconv/dw_n]
            # So the Jacobian row for constraint k is: [-dconv_k/dw_1, ..., -dconv_k/dw_n, 1]
            # We set jac[:, -1] = 1 above, and jac[:, i] should be -dconv_k/dw_i = -2*h_{k-s_i}
            # Let me fix this.
        
        # Let me rewrite the Jacobian properly
        def constraint_ineq_jac(x):
            w = x[:-1]
            h_full = np.zeros(N)
            h_full[support] = w
            jac = np.zeros((num_conv, n_vars))
            jac[:, -1] = 1.0  # dM/dM
            for i, s_idx in enumerate(support):
                for k in range(num_conv):
                    j = k - s_idx
                    if 0 <= j < N:
                        jac[k, i] = -2.0 * h_full[j]
                    else:
                        jac[k, i] = 0.0
            return jac
        
        constraints = [
            {'type': 'eq', 'fun': constraint_eq_fun, 'jac': constraint_eq_jac},
            {'type': 'ineq', 'fun': constraint_ineq_fun, 'jac': constraint_ineq_jac}
        ]
        
        # Initial guess
        w_init = h_init[support].copy()
        if np.sum(w_init) < 1e-15:
            w_init = np.ones(len(support))
        w_init = w_init / np.sum(w_init)
        h_full_init = np.zeros(N)
        h_full_init[support] = w_init
        conv_init = np.convolve(h_full_init, h_full_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
        
        x0 = np.concatenate((w_init, [M_init]))
        bounds = [(0.0, 1.0)] * len(support) + [(0.0, None)]
        
        try:
            res = opt.minimize(
                objective, x0, method='SLSQP', bounds=bounds,
                constraints=constraints, jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            if res.success:
                w_opt = res.x[:-1]
                w_opt = np.clip(w_opt, 0, None)
                if np.sum(w_opt) > 1e-15:
                    w_opt /= np.sum(w_opt)
                    h_out = np.zeros(N)
                    h_out[support] = w_opt
                    return h_out
        except Exception:
            pass
        return None

    # Run epigraph on the best Sidon-like supports
    # Sort by initial score (with uniform weights)
    sidon_supports.sort(key=lambda x: _score_raw(x[0]), reverse=True)
    
    max_epigraph_runs = min(len(sidon_supports), 25)
    for i, (h_s, support, seed, max_size) in enumerate(sidon_supports[:max_epigraph_runs]):
        if time.time() > deadline - 20:
            break
        h_candidate = run_epigraph_on_support(support, h_s, maxiter=200, ftol=1e-10)
        if h_candidate is not None:
            update_best(h_candidate)

    # === Phase 3: Block-Coordinate Descent Refinement ===
    def block_coordinate_refine(h_start, max_iter=20):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 5:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            
            # For each pair (i, k_star - i), try to shift weight
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                
                hi = h[i]
                hj = h[j]
                if hi + hj < 1e-15:
                    continue
                
                # Try shifting t from i to j
                best_t = 0.0
                best_local = _score_raw(h)
                
                for t in np.linspace(-hi * 0.9, hj * 0.9, 20):
                    h_test = h.copy()
                    h_test[i] += t
                    h_test[j] -= t
                    if np.any(h_test < -1e-12):
                        continue
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > best_local:
                        best_local = s_test
                        best_t = t
                
                if best_t != 0.0:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    h /= np.sum(h)
                    improved = True
        
        return h

    # Refine the best solution
    if best_score > 0:
        h_refined = block_coordinate_refine(best_h, max_iter=15)
        update_best(h_refined)

    # === Phase 4: Final local search - try adding/removing/moving peaks ===
    if best_score > 0:
        h_curr = best_h.copy()
        support_curr = np.where(h_curr > 1e-6)[0]
        
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Try adding a new peak
            zero_idx = np.where(h_curr < 1e-6)[0]
            if len(zero_idx) > 0:
                best_add_idx = -1
                best_add_s = best_score
                for idx in zero_idx:
                    h_try = h_curr.copy()
                    h_try[idx] = 0.05
                    s = _score_raw(h_try)
                    if s > best_add_s:
                        best_add_s = s
                        best_add_idx = idx
                
                if best_add_idx != -1:
                    h_curr[best_add_idx] = 0.05
                    # Refine
                    h_refined = block_coordinate_refine(h_curr, max_iter=10)
                    if _score_raw(h_refined) > best_score:
                        best_score = _score_raw(h_refined)
                        best_h = h_refined.copy()
                        h_curr = h_refined.copy()
                        improved = True
            
            # Try removing a peak
            if len(support_curr) > 3:
                best_remove_idx = -1
                best_remove_s = best_score
                for idx in support_curr:
                    h_try = h_curr.copy()
                    h_try[idx] = 0.0
                    s = _score_raw(h_try)
                    if s > best_remove_s:
                        best_remove_s = s
                        best_remove_idx = idx
                
                if best_remove_idx != -1:
                    h_curr[best_remove_idx] = 0.0
                    h_curr = h_curr / np.sum(h_curr)
                    h_refined = block_coordinate_refine(h_curr, max_iter=10)
                    if _score_raw(h_refined) > best_score:
                        best_score = _score_raw(h_refined)
                        best_h = h_refined.copy()
                        h_curr = h_refined.copy()
                        improved = True
            
            # Try moving a peak
            support_curr = np.where(h_curr > 1e-6)[0]
            for idx in support_curr:
                for offset in [-1, 1, -2, 2]:
                    new_idx = idx + offset
                    if 0 <= new_idx < N and h_curr[new_idx] < 1e-6:
                        h_try = h_curr.copy()
                        h_try[new_idx] = h_try[idx]
                        h_try[idx] = 0.0
                        s = _score_raw(h_try)
                        if s > best_score:
                            best_score = s
                            best_h = h_try.copy()
                            h_curr = h_try.copy()
                            improved = True
                            break
            
            if not improved:
                break

    return list(best_h)
