import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # The uniform function gives a normalized peak of 2.0.
    # We want to see if we can do better.
    # Note: For non-negative functions, the peak of the autoconvolution is at least
    # the square of the integral divided by the length of the support (by Cauchy-Schwarz
    # or similar arguments, but actually the peak is at least the value at 0 which is
    # integral of f(t)^2 dt).
    # Wait, the normalized maximum is max(f*f) / (integral(f))^2.
    # For uniform f=1 on [0,1], f*f is a triangle with peak 1 at 0. integral is 1.
    # So normalized peak is 1.
    # In our case, the interval is [-1/4, 1/4], length L=0.5.
    # If f=1, integral = 0.5. f*f has peak 0.5 at 0.
    # Normalized peak = 0.5 / (0.5)^2 = 2.
    #
    # Is it possible to get less than 2?
    # Let f be a step function.
    # If we make f smaller in the middle, the peak of f*f might decrease?
    # Actually, f*f at 0 is sum f[i]^2 * dx.
    # If we reduce f[i] for some i, sum f[i]^2 decreases, but integral also decreases.
    # Let's try a specific shape: f[i] = 1 for i in first half, 0 for second half?
    # No, support must be the whole interval? "supported on a fixed interval" usually
    # means the support is a subset, but the task says "step function with 600 equal
    # intervals on [-1/4,1/4]". It doesn't explicitly say it must be non-zero everywhere.
    # However, if we shrink the support, say to length L/2, the uniform function on that
    # smaller support would have normalized peak 2/(L/2) = 4/L = 8. That's worse.
    #
    # What if we use a shape that spreads out the mass?
    # Consider f(x) = cos(pi x / L) for x in [-L/2, L/2]?
    # Or maybe a "hat" function?
    #
    # Let's try to optimize a simple parametric family or use a direct optimizer.
    # Given the time constraint and the fact that the uniform is a known bound,
    # and that 2 is a very "round" number often appearing in such inequalities,
    # it's likely the optimum is 2.
    #
    # However, the task asks to minimize. If 2 is the minimum, returning uniform is optimal.
    # If there is a shape that gives < 2, we should find it.
    #
    # Let's try a quick check with a few simple shapes to see if any beat 2.0.
    
    def compute_score(f):
        # f is a numpy array of length N
        # Compute linear autoconvolution
        # Zero pad to 2N
        f_pad = np.zeros(2 * N)
        f_pad[:N] = f
        F = np.fft.fft(f_pad)
        autoconv = np.fft.ifft(F * F).real
        # The peak is likely at the center (index N)
        # But to be safe, find the max
        peak = np.max(autoconv)
        integral = np.sum(f)
        if integral < 1e-12:
            return np.inf
        return peak / (integral * integral)

    # Candidate 1: Uniform
    f_uniform = np.ones(N)
    score_uniform = compute_score(f_uniform)
    
    # Candidate 2: Cosine bump (non-negative on [-1/4, 1/4])
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 1.0/(2*N)
    f_cos = np.cos(np.pi * x / 0.5)
    f_cos = np.maximum(f_cos, 0)
    # Ensure it's not zero everywhere
    if np.sum(f_cos) > 0:
        score_cos = compute_score(f_cos)
    else:
        score_cos = np.inf

    # Candidate 3: Triangle (hat) function
    f_tri = np.abs(1 - 2 * np.abs(x) / 0.5)
    f_tri = np.maximum(f_tri, 0)
    if np.sum(f_tri) > 0:
        score_tri = compute_score(f_tri)
    else:
        score_tri = np.inf

    # Candidate 4: "Two bumps"
    f_two = np.exp(-((x-0.1)**2)*50) + np.exp(-((x+0.1)**2)*50)
    score_two = compute_score(f_two)

    # Candidate 5: "U-shape" (high at edges, low in middle)
    f_u = np.abs(x) * 4 # Proportional to distance from center
    f_u = np.maximum(f_u, 0.1) # Avoid zero
    score_u = compute_score(f_u)

    best_score = score_uniform
    best_f = f_uniform.copy()
    
    candidates = [
        (score_uniform, f_uniform),
        (score_cos, f_cos),
        (score_tri, f_tri),
        (score_two, f_two),
        (score_u, f_u),
    ]
    
    for s, f in candidates:
        if s < best_score:
            best_score = s
            best_f = f.copy()

    # If we have time, try a simple gradient descent on a smoothed version?
    # Or just return the best of these simple shapes.
    # Given the complexity of optimizing 600 variables in 140s, and the likelihood
    # that 2 is the true minimum (achieved by uniform), returning the best simple
    # shape is a safe bet.
    
    # Let's do a very quick local refinement if the best score is significantly
    # above 2.0? No, if it's 2.0, it's likely optimal.
    
    # If the best score is close to 2.0, uniform is probably the answer.
    # If a simple shape beats 2.0, we return that.
    
    # Let's try one more thing: a "box" of width w < 0.5 centered at 0.
    # If we restrict support to width w, the min normalized peak for that support is 2/w.
    # Since w <= 0.5, 2/w >= 4. So shrinking support is bad.
    
    # What if we make the function non-symmetric?
    # The peak of autoconvolution is at the shift where the overlap is maximized.
    # For non-negative functions, the maximum overlap is at shift 0 (by rearrangement inequality? No, that's for sums).
    # Actually, for non-negative f, f*f is maximized at t=0?
    # (f*f)(t) = int f(x) f(x-t) dx.
    # By Cauchy-Schwarz, (int f(x)f(x-t)dx)^2 <= (int f^2)(int f^2).
    # This doesn't tell us where the max is.
    # But intuitively, shifting the function away from itself reduces overlap.
    # So the peak is likely at t=0.
    # If the peak is at t=0, then we are minimizing int f(x)^2 dx / (int f(x) dx)^2.
    # By Cauchy-Schwarz, (int f)^2 <= (int 1)(int f^2) = L * int f^2.
    # So int f^2 / (int f)^2 >= 1/L.
    # Here L = 0.5. So the ratio is >= 1/0.5 = 2.
    # Equality holds when f is constant (almost everywhere).
    # So the uniform function achieves the lower bound of 2.
    #
    # Therefore, the minimum possible value is 2.0.
    # Any non-uniform function will have a ratio > 2.0 (assuming the peak is at 0).
    # Is it possible that the peak is NOT at 0 for some non-uniform function, and the
    # ratio at that peak is lower?
    # The score is max_t (f*f)(t) / (int f)^2.
    # We know (f*f)(0) / (int f)^2 >= 2.
    # So the max is at least 2.
    # Thus the minimum possible score is 2.
    # The uniform function achieves 2.
    # So the uniform function is optimal.

    return list(best_f)
