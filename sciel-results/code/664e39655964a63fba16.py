"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlap_max(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    For a step function h on [0,2] with N steps of width 2/N,
    h[i] is the height on [2*i/N, 2*(i+1)/N).
    
    The overlap of h with its shift by t is:
    overlap(t) = integral_0^2 h(x) * (1 - h(x-t)) dx
    (with appropriate boundary handling, assuming h is 0 outside [0,2])
    
    For step functions, we can compute this exactly for shifts that are
    multiples of the step width. But the maximum might occur at non-multiple shifts.
    
    Actually, for step functions, the overlap as a function of shift is piecewise
    linear (or piecewise constant depending on interpretation). Let me think more carefully.
    
    h is a step function: h(x) = h[i] for x in [2i/N, 2(i+1)/N)
    1-h is also a step function.
    
    overlap(t) = integral_0^2 h(x) (1-h(x-t)) dx
    
    Let's compute this for t in [0, 2]. For a given t, the product h(x)(1-h(x-t))
    is a step function with breakpoints at the breakpoints of h and at the breakpoints
    of h(x-t) shifted by t.
    
    This is complex. Let's use a different approach: for each pair of steps i, j,
    determine the measure of x such that x is in step i and x-t is in step j.
    
    Step i covers [2i/N, 2(i+1)/N).
    Step j covers [2j/N, 2(j+1)/N).
    
    x in step i AND x-t in step j means:
    2i/N <= x < 2(i+1)/N
    2j/N <= x-t < 2(j+1)/N  =>  2j/N + t <= x < 2(j+1)/N + t
    
    So x is in the intersection of [2i/N, 2(i+1)/N) and [2j/N + t, 2(j+1)/N + t).
    
    The measure of this intersection is:
    max(0, min(2(i+1)/N, 2(j+1)/N + t) - max(2i/N, 2j/N + t))
    
    Then overlap(t) = sum over i,j of h[i] * (1-h[j]) * measure(i,j,t)
    
    To find max over t in [0,2], note that overlap(t) is piecewise linear in t,
    with breakpoints at t = 2(i-j)/N for various i,j. So the maximum occurs at
    one of these breakpoints or at the endpoints.
    
    Actually, let's just evaluate at all candidate t values: t = 2k/N for k = 0, 1, ..., N.
    Wait, that's only N+1 points. But the breakpoints are at t = 2(i-j)/N, which are
    also multiples of 2/N. So yes, evaluating at t = 2k/N for k=0..N should suffice.
    
    Hmm, but actually the intersection measure changes slope when t crosses 2(i-j)/N,
    so the maximum of a piecewise linear function on [0,2] occurs at an endpoint of
    a linear piece, i.e., at t = 2k/N for some integer k.
    
    Let's implement this.
    """
    h = np.asarray(h, dtype=float)
    n = len(h)
    w = 2.0 / n  # step width
    
    # For each shift k (0 to n), t = k * w
    # Compute overlap(t) for each k
    max_ov = 0.0
    for k in range(n + 1):
        t = k * w
        ov = 0.0
        for i in range(n):
            # x in [2i/N, 2(i+1)/N)
            x_lo = 2.0 * i / n
            x_hi = 2.0 * (i + 1) / n
            # x - t in [2j/N, 2(j+1)/N) for some j
            # x in [2j/N + t, 2(j+1)/N + t)
            # So for each j, the intersection of [x_lo, x_hi) and [2j/N + t, 2(j+1)/N + t)
            for j in range(n):
                y_lo = 2.0 * j / n + t
                y_hi = 2.0 * (j + 1) / n + t
                # Intersection of [x_lo, x_hi) and [y_lo, y_hi)
                lo = max(x_lo, y_lo)
                hi = min(x_hi, y_hi)
                if lo < hi:
                    ov += h[i] * (1.0 - h[j]) * (hi - lo)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _evaluate(h):
    """Return the score: upper bound on C5, which is the max overlap."""
    return _overlap_max(h)

def construct():
    """
    Construct step heights to minimize the maximum overlap.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with uniform
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _evaluate(h)
    
    # Try to improve using local search / gradient descent
    # The problem is to minimize max_t overlap(t)
    
    # Let's try a simple approach: try all "balanced" step functions
    # where the sum of h[i] = N/2, and see if we can do better than 0.5
    
    # Actually, let's think about what the optimal solution looks like.
    # For h = 0.5 everywhere, overlap(t) = 0.25 * 2 = 0.5 for all t.
    # So the max overlap is 0.5.
    
    # Can we do better? If h is not constant, then 1-h is not constant either.
    # The overlap involves h(x) * (1-h(x-t)). If h varies, then at some t,
    # h(x) and 1-h(x-t) might be correlated in a way that reduces the product.
    
    # Let's try a simple periodic pattern: h alternates between a and b.
    # For N even, h[i] = a if i even, b if i odd.
    # Sum = N/2 * (a+b) = N/2 => a+b = 1.
    # So b = 1-a.
    
    # For this pattern, let's compute the overlap.
    # By symmetry, the max overlap might be at t = 0 or t = w (one step shift).
    
    # Let's just try a range of values and see.
    best_score = 0.5  # baseline
    best_h = np.full(N, 0.5)
    
    # Try alternating patterns
    for a in np.linspace(0, 1, 101):
        b = 1.0 - a
        h_alt = np.zeros(N)
        for i in range(N):
            if i % 2 == 0:
                h_alt[i] = a
            else:
                h_alt[i] = b
        # Check sum
        if abs(h_alt.sum() - N/2) > 1e-9:
            continue
        s = _evaluate(h_alt)
        if s < best_score - 1e-12:
            best_score = s
            best_h = h_alt.copy()
    
    # Try other patterns: blocks
    for block_len in range(1, N//2 + 1):
        for a in np.linspace(0, 1, 51):
            b = 1.0 - a
            h_block = np.zeros(N)
            for i in range(N):
                if (i // block_len) % 2 == 0:
                    h_block[i] = a
                else:
                    h_block[i] = b
            if abs(h_block.sum() - N/2) > 1e-9:
                continue
            s = _evaluate(h_block)
            if s < best_score - 1e-12:
                best_score = s
                best_h = h_block.copy()
    
    # Try a gradient descent approach
    # Parametrize h with the constraint sum(h) = N/2
    # Use a parameterization: h = 0.5 + u, where sum(u) = 0
    # Optimize over u with sum(u) = 0
    
    u = np.zeros(N)
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline:
            break
        h_curr = 0.5 + u
        h_curr = np.clip(h_curr, 0, 1)
        # Project back to satisfy sum constraint
        diff = h_curr.sum() - N/2
        h_curr -= diff / N
        h_curr = np.clip(h_curr, 0, 1)
        
        s = _evaluate(h_curr)
        if s < best_score - 1e-12:
            best_score = s
            best_h = h_curr.copy()
        
        # Compute gradient numerically
        grad = np.zeros(N)
        eps = 0.001
        for i in range(N):
            u_plus = u.copy()
            u_plus[i] += eps
            h_plus = 0.5 + u_plus
            h_plus = np.clip(h_plus, 0, 1)
            h_plus -= (h_plus.sum() - N/2) / N
            h_plus = np.clip(h_plus, 0, 1)
            s_plus = _evaluate(h_plus)
            
            u_minus = u.copy()
            u_minus[i] -= eps
            h_minus = 0.5 + u_minus
            h_minus = np.clip(h_minus, 0, 1)
            h_minus -= (h_minus.sum() - N/2) / N
            h_minus = np.clip(h_minus, 0, 1)
            s_minus = _evaluate(h_minus)
            
            grad[i] = (s_plus - s_minus) / (2 * eps)
        
        u -= lr * grad
        u -= u.mean()  # enforce zero mean
    
    if time.time() > deadline:
        return best_h
    
    return best_h
# EVOLVE-BLOCK-END
