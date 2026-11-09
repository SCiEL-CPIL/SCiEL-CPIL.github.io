import numpy as np
import time

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 + X3 < 2*X4].
    P = sum_{k,l} p_k * p_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i * p_j
    """
    n = len(s)
    if n == 0:
        return 0.0

    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)

    # Re-normalize after clipping
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum

    # Build pair sums and probs
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]

    ss_flat = ss.flatten()
    pp_flat = pp.flatten()

    # Sort by sum
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]

    # Cumulative sum of probs
    C = np.cumsum(pp_sorted)

    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            idx = np.searchsorted(ss_sorted, threshold, side='left')
            if idx > 0:
                prob_inner = C[idx - 1]
            else:
                prob_inner = 0.0
            P += p[k] * p[l] * prob_inner

    return P

def construct_mixture(alpha, beta, gamma, n):
    """
    Construct a two-point mixture:
    1. Point mass at 0 with weight alpha.
    2. Pareto-like tail on geometric grid with weight (1-alpha).
    
    Parameters:
    alpha: Mass at 0 (0 < alpha < 1)
    beta: Tail shape parameter (larger = heavier tail)
    gamma: Scale parameter (larger = wider spread)
    n: Total number of support points
    
    Returns:
    s: support array
    w: weights array
    """
    # Ensure valid parameters
    if alpha <= 0 or alpha >= 1:
        alpha = 0.5
    if beta <= 0.1:
        beta = 0.1
    if gamma <= 1e-6:
        gamma = 1e-6

    # The tail component has n-1 points (since index 0 is reserved for 0)
    n_tail = n - 1
    if n_tail < 1:
        # Fallback for very small n
        s = np.array([0.0, 1.0])
        w = np.array([alpha, 1.0 - alpha])
        return s, w

    # Geometric grid for the tail
    # Start near 0 but not 0, end at gamma
    # To avoid log(0), we start at a small epsilon
    eps = 1e-6
    start = max(eps, gamma * 1e-6)
    # We want n_tail points geometrically spaced from start to gamma
    # Let's use log-spaced points
    log_start = np.log(start)
    log_end = np.log(gamma)
    
    if log_start >= log_end:
        # Degenerate case, just spread linearly or something
        s_tail = np.linspace(start, gamma, n_tail)
    else:
        s_tail = np.exp(np.linspace(log_start, log_end, n_tail))
    
    # Weights for the tail component
    # We want a heavy tail. A simple way is to assign weights proportional to 1/s^beta 
    # or just uniform on the geometric grid? 
    # Actually, if we want to optimize the "shape", we can parameterize the weights 
    # of the tail component as a power law on the indices or values.
    
    # Let's define the tail weights as proportional to (s_tail)^(-beta) 
    # This creates a distribution where smaller values in the tail are more probable, 
    # but the support extends to large values.
    # Wait, if we want a heavy tail that helps the inequality X1+X2+X3 < 2X4,
    # we want X4 to be large often, and X1,X2,X3 to be small often.
    # The point mass at 0 handles the "small" case for X1,X2,X3.
    # The tail handles the "large" case for X4.
    
    # So, for the tail component, we want it to have significant mass at large values 
    # to act as X4, but also some mass at smaller values to act as X1,X2,X3 
    # (though the point mass at 0 covers the smallest values).
    
    # Let's try a weight distribution on the tail that is roughly uniform in log-space 
    # or slightly biased towards the larger end to boost P[X4 is large].
    # Actually, if X4 is large, 2X4 is large, making the inequality easier to satisfy.
    # So we want the marginal distribution of X to have a heavy right tail.
    
    # Let's parameterize the tail weights as a power law on the rank or value.
    # Let's use weights proportional to 1 / (s_tail ** beta) ? No, that puts more mass at small s_tail.
    # We want mass at large s_tail to make X4 large.
    # Let's use weights proportional to s_tail ** (beta - 1) ? 
    # Or simply, let's optimize the weights of the tail component directly?
    # The prompt suggests a 3-parameter model: alpha, beta, gamma.
    # Let's fix the tail weight distribution shape to be "standard" Pareto-like 
    # on the geometric grid, and let gamma scale the range.
    
    # Standard Pareto CDF: P(X < x) = 1 - (xm/x)^alpha for x >= xm.
    # On a geometric grid, the probability mass is roughly uniform in log-space 
    # if we sample from a Pareto.
    # Let's just use uniform weights on the geometric grid for the tail component 
    # and rely on alpha and gamma to tune the mixture.
    # If uniform weights on log-spaced grid are not enough, we can add a bias parameter.
    
    # Let's try: Tail weights are proportional to 1 / s_tail ** beta.
    # This puts more mass at smaller tail values. 
    # But we want X4 to be large. 
    # Contradiction? 
    # If X4 is drawn from the tail, and we want 2X4 to be large, we want X4 to be large.
    # So we want mass at large values.
    # Let's use weights proportional to s_tail ** (beta - 1). 
    # If beta > 1, mass increases with value.
    # If beta < 1, mass decreases with value.
    
    # Let's stick to a simple parametric form:
    # w_tail_i proportional to exp(-beta * log(s_tail_i / gamma)) ? 
    # That's just a power law.
    
    # Let's use w_tail_i = 1 / (s_tail_i ** beta).
    # This is a standard heavy-tail density shape.
    # Wait, if I want X4 to be large, I should probably put MORE mass at large values.
    # So maybe w_tail_i proportional to s_tail_i ** (beta - 1) with beta > 1?
    
    # Let's experiment with the sign. 
    # Let's define tail weights as:
    # w_tail_i = (s_tail_i / gamma) ** (beta - 1)
    # If beta = 1, uniform in linear space? No, uniform in log space if we normalize.
    # Actually, let's just use uniform weights on the geometric grid for the tail.
    # This gives a "log-uniform" distribution.
    # Then alpha controls the mass at 0.
    # gamma controls the max value.
    # Is that enough? 
    # The inequality is X1+X2+X3 < 2X4.
    # If X4 is large (from tail), LHS is likely small (from 0s).
    # If X4 is small (from 0), LHS must be even smaller (negative? no, non-negative).
    # If X4=0, 2X4=0. Inequality X1+X2+X3 < 0 is impossible since Xi >= 0.
    # So cases where X4=0 contribute 0 to P.
    # Cases where X4 > 0 contribute.
    # P = P[X4 > 0] * P[X1+X2+X3 < 2X4 | X4 > 0].
    
    # P[X4 > 0] = 1 - alpha.
    # To maximize P, we want (1-alpha) to be large, so alpha small.
    # But if alpha is small, X1,X2,X3 are more likely to be from the tail.
    # If X1,X2,X3 are from the tail, their sum is large.
    # So we need a balance.
    
    # Let's use the 3 parameters:
    # alpha: mass at 0
    # beta: shape of tail weights (exponent in power law)
    # gamma: max value of support
    
    s_tail = np.exp(np.linspace(np.log(1e-6), np.log(gamma), n_tail))
    
    # Weights for tail
    # Let's try w_tail_i proportional to s_tail_i ** (beta - 1)
    # This allows shifting mass towards larger or smaller values.
    if beta == 1:
        w_tail = np.ones(n_tail)
    else:
        w_tail = s_tail ** (beta - 1)
    
    w_tail = w_tail / np.sum(w_tail)
    
    # Combine
    s = np.zeros(n)
    w = np.zeros(n)
    
    s[0] = 0.0
    w[0] = alpha
    
    s[1:] = s_tail
    w[1:] = (1.0 - alpha) * w_tail
    
    return s, w

def beat_the_average_measure() -> tuple:
    deadline = time.time() + 600.0
    safety_margin = 10.0
    
    n = 20  # Fixed number of support points
    
    # Initial guesses
    best_alpha = 0.1
    best_beta = 1.0
    best_gamma = 1.0
    best_score = -1.0
    best_s = None
    best_w = None
    
    # Grid search / Coordinate ascent
    # We will perform a coarse grid search followed by fine-tuning.
    
    alphas = np.linspace(0.01, 0.8, 10)
    betas = np.linspace(0.5, 3.0, 10)
    gammas = np.logspace(-1, 2, 10) # 0.1 to 100
    
    # Coarse Grid Search
    for a in alphas:
        for b in betas:
            for g in gammas:
                if time.time() > deadline - safety_margin:
                    break
                s, w = construct_mixture(a, b, g, n)
                score = evaluate_measure(s, w)
                if score > best_score:
                    best_score = score
                    best_alpha = a
                    best_beta = b
                    best_gamma = g
                    best_s = s.copy()
                    best_w = w.copy()
            
            if time.time() > deadline - safety_margin:
                break
        if time.time() > deadline - safety_margin:
            break
    
    # Fine-tuning around the best found
    if best_s is not None:
        # Local search
        step_a = 0.01
        step_b = 0.05
        step_g = 0.1
        
        for _ in range(50): # Limited iterations
            if time.time() > deadline - safety_margin:
                break
            
            improved = False
            
            # Try adjusting alpha
            for da in [-step_a, step_a]:
                a_new = best_alpha + da
                if 0.01 < a_new < 0.9:
                    s, w = construct_mixture(a_new, best_beta, best_gamma, n)
                    score = evaluate_measure(s, w)
                    if score > best_score:
                        best_score = score
                        best_alpha = a_new
                        best_s = s.copy()
                        best_w = w.copy()
                        improved = True
            
            # Try adjusting beta
            for db in [-step_b, step_b]:
                b_new = best_beta + db
                if b_new > 0.1:
                    s, w = construct_mixture(best_alpha, b_new, best_gamma, n)
                    score = evaluate_measure(s, w)
                    if score > best_score:
                        best_score = score
                        best_beta = b_new
                        best_s = s.copy()
                        best_w = w.copy()
                        improved = True
            
            # Try adjusting gamma
            for dg in [-step_g, step_g]:
                g_new = best_gamma * (1 + dg/best_gamma) # Multiplicative step
                if g_new > 1e-3:
                    s, w = construct_mixture(best_alpha, best_beta, g_new, n)
                    score = evaluate_measure(s, w)
                    if score > best_score:
                        best_score = score
                        best_gamma = g_new
                        best_s = s.copy()
                        best_w = w.copy()
                        improved = True
            
            if not improved:
                # Reduce steps
                step_a *= 0.5
                step_b *= 0.5
                step_g *= 0.5
                if step_a < 1e-4:
                    break

    # Final validation and return
    if best_s is None:
        # Fallback
        s = np.linspace(0, 1, n)
        w = np.ones(n) / n
    else:
        s = best_s
        w = best_w
    
    s = np.asarray(s, dtype=np.float64).flatten()
    w = np.asarray(w, dtype=np.float64).flatten()
    
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    
    w_sum = np.sum(w)
    if w_sum > 1e-9:
        w = w / w_sum
    else:
        w = np.ones(n) / n
        
    return s, w

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(f"Score: {evaluate_measure(s, w)}")
    print(f"n = {len(s)}")
