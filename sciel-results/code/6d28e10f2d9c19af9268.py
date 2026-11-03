import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
NUM_CONV = 2 * N - 1

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def _generate_sidon_set(n_max, target_size):
    """Generate a Sidon set of given size within [0, n_max)."""
    # Greedy construction
    S = []
    for i in range(n_max):
        # Check if i can be added
        valid = True
        for j in S:
            diff = abs(i - j)
            # Check if this difference already exists
            # Simple check: if diff is already a difference between two elements in S
            # This is O(K^2) per insertion, which is fine for small K
            for k in range(len(S)):
                for l in range(k+1, len(S)):
                    if abs(S[k] - S[l]) == diff:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if valid:
            S.append(i)
            if len(S) == target_size:
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)

    x_idx = np.arange(N)

    # --- Phase 0: Try Sidon Sets ---
    # Sidon sets guarantee M = ||h||_2^2, so Score = 1/DELTA.
    # This is the theoretical maximum.
    for k in range(10, 3, -1): # Try larger sets first? No, larger sets are harder to find.
        S = _generate_sidon_set(N, k)
        if len(S) == k:
            h_sidon = np.zeros(N)
            h_sidon[S] = 1.0 / k
            s_sidon = _score_raw(h_sidon)
            if s_sidon > best_score:
                best_score = s_sidon
                best_h = h_sidon.copy()
    
    # If we found a perfect Sidon set, we are at the theoretical max.
    # We can return early if we are confident.
    if best_score >= 1.0 / DELTA * 0.999:
         # Still run a quick polish to ensure no numerical error
         pass

    # --- Phase 1: Generate diverse initializations ---
    inits = []

    # 1. Uniform
    inits.append(np.ones(N) / N)

    # 2. Sparse structured (Sidon-like intuition)
    np.random.seed(42)
    for trial in range(30):
        k = np.random.randint(3, 12)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)

    # 3. Concentrated Gaussians
    for c in np.linspace(3, N-3, 10):
        for w in np.linspace(0.5, N/3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)

    # 4. Triangular
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)

    # 5. Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)

    # 6. Alternating patterns (helps reduce auto-correlation peaks)
    for start in range(N):
        for gap in [2, 3, 4, 5]:
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                inits.append(h)

    # --- Phase 2: Epigraph Formulation Optimization ---
    # Variables: x = [h_0, ..., h_{N-1}, M]
    # Objective: Minimize M (x[N])
    # Constraints:
    # 1. sum(h) - 1 == 0
    # 2. h_i >= 0 (via bounds)
    # 3. M - conv_k >= 0 for all k

    def objective(x):
        return x[N]

    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad

    def constraint_eq_fun(x):
        return np.sum(x[:N]) - 1.0

    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac

    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv

    def constraint_ineq_jac(x):
        h = x[:N]
        jac = np.zeros((NUM_CONV, N + 1))
        jac[:, N] = 1.0
        for j in range(N):
            # Part 1: k-j is in [0, N-1] => k in [j, j+N-1]
            k_start = j
            k_end = min(j + N, NUM_CONV)
            if k_start < NUM_CONV and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] -= h[indices_h]

            # Part 2: j-k is in [0, N-1] => k in [j-N+1, j]
            k_start2 = max(0, j - N + 1)
            k_end2 = min(j, NUM_CONV)
            if k_start2 < NUM_CONV and k_end2 >= 0:
                indices_k = np.arange(k_start2, k_end2 + 1)
                indices_h = j - indices_k
                jac[indices_k, j] -= h[indices_h]

        return jac

    constraints = [
        {
            'type': 'eq',
            'fun': constraint_eq_fun,
            'jac': constraint_eq_jac
        },
        {
            'type': 'ineq',
            'fun': constraint_ineq_fun,
            'jac': constraint_ineq_jac
        }
    ]

    bounds = [(0.0, 1.0)] * N + [(0.0, None)]

    def run_epigraph(h_init, maxiter=300, ftol=1e-12):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
        x0 = np.concatenate((h_init, [M_init]))

        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )

            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        return None

    # Sort inits by initial score (descending) to try best ones first
    scored_inits = []
    for h_init in inits:
        s = _score_raw(h_init)
        scored_inits.append((s, h_init))
    scored_inits.sort(key=lambda x: -x[0])

    # Run epigraph from top candidates
    max_epigraph_runs = min(len(scored_inits), 30)
    for i in range(max_epigraph_runs):
        if time.time() > deadline - 40:
            break
        s_init, h_init = scored_inits[i]
        h_candidate = run_epigraph(h_init, maxiter=200, ftol=1e-12)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()

    # --- Phase 3: Local Refinement (Coordinate Descent) ---
    # The epigraph SLSQP finds a good global structure, but might leave small
    # improvements on the table due to numerical tolerance.
    # We perform a coordinate descent where we adjust pairs of bins to reduce
    # the peak convolution while maintaining L1 and maximizing L2.

    def refine_local(h_start, max_iter=50):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)

        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 10:
                break
            improved = False
            iter_count += 1

            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            current_score = l2_sq / (l1 * DELTA * M_val)

            # Try moving mass from high-impact positions to low-impact ones
            # or adjusting individual coordinates to reduce peak convolution
            # while preserving L1

            # Simple coordinate ascent on score
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                h_old = h[i]
                # Try small perturbations
                for delta_val in [0.01, -0.01, 0.001, -0.001, 0.05, -0.05]:
                    h_new = h.copy()
                    h_new[i] = max(0, h_old + delta_val)
                    # Renormalize to maintain L1=1
                    if np.sum(h_new) < 1e-15:
                        continue
                    h_new /= np.sum(h_new)
                    s_new = _score_raw(h_new)
                    if s_new > current_score:
                        h = h_new
                        current_score = s_new
                        improved = True

        return h

    # Apply local refinement to the best solution
    if time.time() < deadline - 10:
        h_refined = refine_local(best_h, max_iter=30)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined

    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        conv_init = np.convolve(h_start, h_start, mode='full')
        M_init = np.max(conv_init) + 1e-6
        x0 = np.concatenate((h_start, [M_init]))

        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass

    # Return the optimized shape (normalized to L1=1)
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)

    return list(best_h)
