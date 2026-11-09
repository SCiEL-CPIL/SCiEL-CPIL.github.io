import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(s, w):
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

def _optimize_weights(s, time_budget):
    """
    Optimize weights for a fixed support using gradient ascent.
    Returns optimal weights (normalized) and the score.
    """
    n = len(s)
    
    # Initial weights: uniform
    w = np.ones(n) / n
    
    best_w = w.copy()
    best_score = _compute_P(s, w)
    
    start_time = time.time()
    step_size = 0.01
    no_improve_count = 0
    
    for iteration in range(1000):
        if time.time() > time_budget:
            break
        if step_size < 1e-6:
            break
            
        improved = False
        
        # Try adjusting each weight slightly (coordinate ascent)
        for i in range(n):
            if time.time() > time_budget:
                break
                
            # Try increasing w_i
            w_trial = w.copy()
            w_trial[i] += step_size
            if w_trial[i] < 0:
                w_trial[i] = 0
            w_sum = np.sum(w_trial)
            if w_sum > 1e-9:
                p_trial = w_trial / w_sum
                score = _compute_P(s, p_trial * w_sum) # Pass unnormalized or normalized? _compute_P normalizes.
                if score > best_score:
                    best_score = score
                    w = w_trial.copy()
                    best_w = w.copy()
                    improved = True
                    
            # Try decreasing w_i
            w_trial = w.copy()
            w_trial[i] -= step_size
            if w_trial[i] < 0:
                w_trial[i] = 0
            w_sum = np.sum(w_trial)
            if w_sum > 1e-9:
                p_trial = w_trial / w_sum
                score = _compute_P(s, p_trial * w_sum)
                if score > best_score:
                    best_score = score
                    w = w_trial.copy()
                    best_w = w.copy()
                    improved = True
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 5:
                step_size *= 0.5
                no_improve_count = 0
            else:
                break
                
    # Normalize final weights
    w_sum = np.sum(best_w)
    if w_sum > 1e-9:
        best_w = best_w / w_sum
    else:
        best_w = np.ones(n) / n
        
    return best_w, best_score

def beat_the_average_measure() -> tuple:
    deadline = time.time() + 600.0
    safety_margin = 10.0
    
    # Strategy:
    # 1. Define a parametric family of supports.
    #    A key insight is that the inequality X1+X2+X3 < 2*X4 is easier to satisfy
    #    if X4 is large and X1,X2,X3 are small.
    #    A distribution with a point mass at 0 and a heavy tail is good.
    #    Let's parameterize the support as:
    #    s_0 = 0
    #    s_i = r^(i-1) * S_max for i=1..n-1
    #    where r is a ratio (0 < r < 1) and S_max is the max value.
    #    Actually, let's use log-spaced points from s_min to s_max.
    #    s_i = s_min * (s_max/s_min)^((i-1)/(n-2)) for i=1..n-1
    #    Let's parameterize by s_min, s_max, and maybe n.
    
    # Let's try a simpler parameterization:
    # s_0 = 0
    # s_1 = a
    # s_i = s_{i-1} * r for i=2..n-1
    # So s_i = a * r^(i-1)
    # Parameters: a (scale), r (ratio), n (number of points)
    
    # Let's search over n, a, r.
    # Then optimize weights for each (n, a, r).
    
    best_score = -1.0
    best_s = None
    best_w = None
    
    # Grid search over n, a, r
    # n: 5 to 30
    # a: 0.01 to 10
    # r: 0.1 to 0.9
    
    # To save time, we can do a coarse search first.
    # Also, note that scaling all s by a constant does not change P?
    # Let's check:
    # P = Prob[X1+X2+X3 < 2*X4]
    # If we scale s by c, X_i -> c*X_i.
    # c*X1 + c*X2 + c*X3 < 2*c*X4  <=> X1+X2+X3 < 2*X4.
    # So P is scale invariant!
    # This means 'a' does not matter for the score, only the shape of the support matters.
    # So we can fix a=1 and just search over r and n.
    # Wait, is it strictly scale invariant?
    # The support values are scaled. The weights are unchanged.
    # The inequality is homogeneous of degree 1.
    # Yes, P is invariant under scaling of the support.
    # So we can fix s_1 = 1 (or any value) and just vary the ratios.
    
    # Let's fix s_0 = 0, s_1 = 1.
    # s_i = r^(i-1) for i=1..n-1?
    # If r < 1, s_i decreases. We want increasing support.
    # So let s_i = r^(i-1) with r > 1? Or s_i = r^(-(i-1)) with r < 1?
    # Let's use s_i = c * r^(i-1) with r > 1.
    # Since scale is invariant, fix c=1.
    # s_0 = 0
    # s_1 = 1
    # s_2 = r
    # s_3 = r^2
    # ...
    # s_{n-1} = r^(n-2)
    
    # So parameters are n and r.
    # r > 1.
    
    # Let's search over n and r.
    # n: 4 to 20 (smaller n is faster to evaluate, but might limit expressiveness)
    # r: 1.1 to 10
    
    # Coarse grid
    n_values = range(4, 16)
    r_values = np.logspace(np.log10(1.01), np.log10(100), 50)
    
    # We will iterate through these, optimize weights, and keep the best.
    # To be efficient, we can skip if time is running out.
    
    # Precompute supports? No, they depend on n and r.
    
    # Let's also consider that maybe a non-geometric support is better?
    # But geometric is a strong candidate for heavy tails.
    
    # Let's also try a uniform support?
    # s_i = i/(n-1) * S_max? Scale invariant, so s_i = i.
    # s_0=0, s_1=1, s_2=2, ..., s_{n-1}=n-1.
    # This is a linear grid.
    
    # Let's add linear grid to the search.
    
    for n in n_values:
        if time.time() > deadline - safety_margin:
            break
            
        # Try geometric supports
        for r in r_values:
            if time.time() > deadline - safety_margin:
                break
                
            # Construct support
            s = np.zeros(n)
            s[0] = 0.0
            if n > 1:
                s[1:] = r ** np.arange(n-1)
                
            # Optimize weights
            w, score = _optimize_weights(s, deadline - safety_margin)
            
            if score > best_score:
                best_score = score
                best_s = s.copy()
                best_w = w.copy()
                
        # Try linear support
        s = np.linspace(0, 1, n) # Scale doesn't matter
        w, score = _optimize_weights(s, deadline - safety_margin)
        if score > best_score:
            best_score = score
            best_s = s.copy()
            best_w = w.copy()
            
    # Fallback
    if best_s is None:
        n = 10
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
        w = np.ones(len(s)) / len(s)
        
    return s, w

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(f"Score: {_compute_P(s, w)}")
    print(f"n = {len(s)}")
