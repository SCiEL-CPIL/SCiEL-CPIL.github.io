import numpy as np
import time

N = 600
DEADLINE = 142  # safety margin before 150s hard limit

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def fast_metric(h):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    def soft_max(c, beta):
        """Soft max approximation using log-sum-exp for smooth gradient."""
        c_max = np.max(c)
        if beta < 1e-10:
            return c_max, np.argmax(c)
        # Numerically stable soft-max
        shifted = c - c_max
        exp_shifted = np.exp(beta * shifted)
        log_sum = np.log(np.sum(exp_shifted))
        soft_max_val = c_max + np.log(np.sum(exp_shifted)) / beta
        # Probabilities for gradient
        probs = exp_shifted / np.sum(exp_shifted)
        return soft_max_val, probs

    # Build Legendre basis: h(n) = sum_j a_j * P_j(x_n), x_n in [-1, 1]
    # We use first M Legendre polynomials. h must be nonnegative.
    M = 40  # number of Legendre modes
    nn = np.arange(N)
    # Map interval indices to [-1, 1]
    x = 2.0 * nn / (N - 1) - 1.0

    # Compute Legendre polynomials P_0..P_{M-1} at x
    # Using recurrence: P_0=1, P_1=x, P_{k+1} = ((2k+1)x P_k - k P_{k-1})/(k+1)
    P = np.zeros((M, N))
    P[0, :] = 1.0
    if M > 1:
        P[1, :] = x
    for k in range(1, M - 1):
        P[k+1, :] = ((2*k + 1) * x * P[k, :] - k * P[k-1, :]) / (k + 1)

    # h = P.T @ a, where a is coefficient vector of length M
    # We need h >= 0, so we'll use a soft penalty for negative values.

    # Initial guess: constant function (a_0 = 1, rest = 0)
    a = np.zeros(M)
    a[0] = 1.0

    def h_from_a(a):
        return P.T @ a

    def metric_and_grad(a, beta=50.0):
        """Compute metric value and gradient w.r.t. a using soft-max."""
        h = h_from_a(a)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(M)
        
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        
        # Use soft-max for smooth gradient
        soft_max_val, probs = soft_max(c, beta)
        val = soft_max_val / (s * s)
        
        # Gradient using soft-max: dc_soft/da = sum_j probs[j] * dc[j]/da
        # dc[j]/da_m = sum_{n=0}^{N-1-j} [P[m,n]*h[n+j] + h[n]*P[m,n+j]]
        # For j=0: dc[0]/da_m = 2 * P[m,:] @ h
        # For j>0: dc[j]/da_m = P[m,:N-j] @ h[j:] + P[m,j:] @ h[:N-j]
        
        # Vectorized computation of dc[j]/da for all j with non-negligible probability
        significant = np.where(probs > 1e-10)[0]
        dc_da = np.zeros(M)
        
        for j in significant:
            if j == 0:
                dc_j = 2.0 * P @ h
            else:
                dc_j = P[:, :N-j] @ h[j:] + P[:, j:] @ h[:N-j]
            dc_da += probs[j] * dc_j
        
        ds_da = P.sum(axis=1)  # sum over n of P[m, n]
        
        grad = dc_da / (s * s) - 2.0 * soft_max_val * ds_da / (s * s * s)
        return val, grad

    # Soft penalty for non-negativity: penalty = beta * sum(max(0, -h)^2)
    # This encourages h >= 0

    def penalized_objective(a, beta, soft_beta=50.0):
        h = h_from_a(a)
        neg = np.maximum(0.0, -h)
        penalty = beta * np.sum(neg * neg)
        val, grad_base = metric_and_grad(a, soft_beta)
        # Gradient of penalty w.r.t. a: beta * sum(2 * (-h) * (-dh/da) for h < 0)
        # = beta * 2 * sum(max(0, -h) * (-P[:, :].T) for negative h)
        # d(penalty)/da_m = beta * 2 * sum_n max(0, -h[n]) * (-P[m, n])
        neg_mask = np.maximum(0.0, -h)
        penalty_grad = -2.0 * beta * P @ neg_mask
        return val + penalty, grad_base + penalty_grad

    # Start with constant
    a = np.zeros(M)
    a[0] = 1.0
    h = h_from_a(a)
    best_val = fast_metric(h)
    best_a = a.copy()

    # Projected gradient descent with adaptive step size and annealing soft-beta
    beta = 10.0  # penalty weight for non-negativity
    lr = 0.02
    n_iter = 8000
    soft_beta_start = 20.0
    soft_beta_end = 200.0
    
    for it in range(n_iter):
        if elapsed() > DEADLINE - 30:
            break
        
        # Anneal soft-beta: start smooth, end sharper
        progress = it / n_iter
        soft_beta = soft_beta_start * (soft_beta_end / soft_beta_start) ** progress
        
        val, grad = penalized_objective(a, beta, soft_beta)
        
        # Gradient step with momentum
        if not hasattr(construct, '_momentum'):
            construct._momentum = np.zeros(M)
        momentum = 0.9
        construct._momentum = momentum * construct._momentum + (1 - momentum) * grad
        a_new = a - lr * construct._momentum
        
        # Soft projection: keep a in a reasonable range
        a_new = np.clip(a_new, -5, 5)
        
        h_new = h_from_a(a_new)
        if np.any(h_new < -0.05):  # too many negatives, shrink step
            lr *= 0.5
            continue
        
        v_new = fast_metric(np.maximum(h_new, 0))
        if v_new < best_val:
            best_val = v_new
            best_a = a_new.copy()
        
        # Adaptive step size
        if v_new < val:
            lr = min(lr * 1.02, 0.05)
        else:
            lr *= 0.8
        
        a = a_new
        
        # Periodically increase beta to enforce non-negativity more strongly
        if it % 500 == 499 and beta < 1000:
            beta *= 2.0

    # Reconstruct h from best_a and clip to nonnegative
    h_best = h_from_a(best_a)
    h_best = np.maximum(h_best, 0)
    
    if np.sum(h_best) < 1e-15:
        h_best = np.ones(N)
    else:
        # Scale to convenient magnitude
        h_best = h_best / np.sum(h_best) * N
    
    # Additional local optimization in Legendre coefficient space
    if elapsed() < DEADLINE - 20:
        # Try small perturbations in coefficient space
        np.random.seed(42)
        best_a_local = best_a.copy()
        best_val_local = best_val
        lr_local = 0.01
        for _ in range(3000):
            if elapsed() > DEADLINE - 15:
                break
            # Random direction in coefficient space
            direction = np.random.randn(M)
            # Try both positive and negative steps
            for sign in [1, -1]:
                a_trial = best_a_local + sign * lr_local * direction
                a_trial = np.clip(a_trial, -5, 5)
                h_trial = h_from_a(a_trial)
                if np.min(h_trial) < -0.01:
                    continue
                h_trial = np.maximum(h_trial, 0)
                v_trial = fast_metric(h_trial)
                if v_trial < best_val_local:
                    best_val_local = v_trial
                    best_a_local = a_trial.copy()
                    if best_val_local < best_val:
                        best_val = best_val_local
                        h_best = h_trial.copy()
            # Adaptive step
            if best_val_local < best_val:
                lr_local = min(lr_local * 1.1, 0.05)
            else:
                lr_local *= 0.9

    # Phase 2: Coordinate descent polish with multiple random restarts
    current_h = h_best.copy()
    current_val = fast_metric(current_h)
    np.random.seed(42)

    no_improve = 0
    max_no_improve = 8000
    mean_h = np.mean(current_h)

    while elapsed() < DEADLINE - 10 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        mean_h = np.mean(current_h)
        # Finer step sizes for better convergence
        step_factors = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
        step_sizes = step_factors * mean_h
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])

        best_local_val = current_val
        best_local_v = orig
        for cv in candidates_val:
            if abs(cv - orig) < 1e-13:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv

        if best_local_val < current_val - 1e-13:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                h_best = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1

        if no_improve % 300 == 0:
            # Perturbation to escape local minima - try multiple perturbations
            found_improvement = False
            for pert_scale in [0.02, 0.05, 0.1]:
                pert = current_h.copy()
                pert += pert_scale * np.random.randn(N) * np.mean(current_h)
                pert = np.clip(pert, 0, None)
                if pert.sum() > 0:
                    pv = fast_metric(pert)
                    if pv < current_val:
                        current_h = pert
                        current_val = pv
                        no_improve = 0
                        found_improvement = True
                        if current_val < best_val:
                            best_val = current_val
                            h_best = current_h.copy()
                        break
            if not found_improvement and no_improve > 2000:
                # Try a larger jump
                pert = current_h.copy()
                pert += 0.15 * np.random.randn(N) * np.mean(current_h)
                pert = np.clip(pert, 0, None)
                if pert.sum() > 0:
                    pv = fast_metric(pert)
                    if pv < current_val:
                        current_h = pert
                        current_val = pv
                        no_improve = 0
                        if current_val < best_val:
                            best_val = current_val
                            h_best = current_h.copy()

    # Phase 3: Fine polish with smaller steps and more iterations
    if elapsed() < DEADLINE - 3:
        step_frac = 0.0005
        max_iters = 5000
        for _ in range(max_iters):
            if elapsed() > DEADLINE - 2:
                break
            # Try both single-index and two-index updates
            if np.random.rand() < 0.5:
                # Single index update
                i = np.random.randint(N)
                delta = step_frac * np.random.uniform(-1, 1) * np.mean(h_best)
                new_h = h_best.copy()
                new_h[i] += delta
                if new_h[i] >= 0:
                    v = fast_metric(new_h)
                    if v < best_val:
                        best_val = v
                        h_best = new_h
            else:
                # Two-index update (preserve sum)
                i, j = np.random.choice(N, 2, replace=False)
                delta = step_frac * np.random.uniform(-1, 1) * np.mean(h_best)
                new_h = h_best.copy()
                new_h[i] += delta
                new_h[j] -= delta
                if new_h[i] >= 0 and new_h[j] >= 0:
                    v = fast_metric(new_h)
                    if v < best_val:
                        best_val = v
                        h_best = new_h

    h_best = np.maximum(h_best, 0)
    if np.sum(h_best) < 1e-15:
        h_best = np.ones(N)

    return h_best.tolist()
