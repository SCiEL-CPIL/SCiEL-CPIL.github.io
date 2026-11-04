"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE_MARGIN = 10.0  # seconds before the 150s kill to stop

def construct():
    t_start = time.time()
    
    # Helper: compute the autoconvolution and the metric
    # The task asks for signed equal-width step functions minimizing
    # max_abs(autoconv) / (integral^2)
    # For a discrete sequence x of length N:
    # autoconv = np.convolve(x, x, mode='full')
    # integral = sum(x)
    # metric = max(abs(autoconv)) / (sum(x)**2)
    
    def metric(x):
        s = np.sum(x)
        if np.abs(s) < 1e-12:
            return float('inf')
        ac = np.convolve(x, x, mode='full')
        return np.max(np.abs(ac)) / (s * s)
    
    def gradient(x):
        # Gradient of the soft-max surrogate for max(|ac|)
        # Let ac = conv(x, x, 'full')
        # We want to minimize max(|ac|) / (sum(x)^2)
        # This is equivalent to minimizing log(max(|ac|)) - 2*log(|sum(x)|)
        # But max is non-differentiable. We use a soft-max approximation.
        # Alternatively, we can use the subgradient of max(|ac|).
        # Let m = max(|ac|). The subgradient of max(|ac_i|) is at the index i* where |ac_i*| = m.
        # grad_ac = conv(x, x, 'full')
        # Let i* be argmax(|grad_ac|).
        # The gradient of ac with respect to x is:
        # d(ac_k)/dx_j = x_{k-j} + x_{k-j+N-1} (for circular) but here it's linear.
        # Actually, ac_k = sum_i x_i x_{k-i}.
        # d(ac_k)/dx_j = x_{k-j} + (if k-j+N-1 in range) ... wait.
        # ac_k = sum_{i=0}^{N-1} x_i x_{k-i}
        # d(ac_k)/dx_j = x_{k-j} + x_{k-j} (if k-j != j? No)
        # Let's use FFT for efficiency and gradient.
        # F = fft(x, 2*N)
        # AC_freq = F * F
        # AC = ifft(AC_freq).real[:N+N-1]
        # This is the autoconvolution.
        
        # Let's stick to the soft-max approach for smoothness.
        # L(x) = log( sum_i exp(beta * |ac_i|) ) / beta - 2 * log(|sum(x)|)
        # This approximates max(|ac|)/beta - 2*log(|sum(x)|)
        # Wait, the objective is max(|ac|) / sum(x)^2.
        # Let's minimize J(x) = log( max(|ac|) / sum(x)^2 ) = log(max(|ac|)) - 2*log(|sum(x)|)
        
        # Soft-max for max(|ac|):
        # S(beta) = (1/beta) * log( sum_i exp(beta * |ac_i|) )
        # As beta -> inf, S(beta) -> max(|ac_i|)
        
        # Gradient of S(beta) with respect to x:
        # dS/dx_j = (1/beta) * ( sum_i exp(beta|ac_i|) * d(|ac_i|)/dx_j ) / ( sum_i exp(beta|ac_i|) )
        # d(|ac_i|)/dx_j = sign(ac_i) * d(ac_i)/dx_j
        
        # This is complex. Let's use the subgradient of the exact max for simplicity,
        # or a simpler heuristic.
        
        # Actually, the "inspiration" suggests:
        # "optimize the continuous heights using projected gradient descent with a soft-max surrogate"
        
        # Let's implement the soft-max gradient.
        beta = 50.0  # temperature for soft-max
        
        ac = np.convolve(x, x, mode='full')
        abs_ac = np.abs(ac)
        
        # Numerical stability for log-sum-exp
        max_ac = np.max(abs_ac)
        exp_term = np.exp(beta * (abs_ac - max_ac))
        sum_exp = np.sum(exp_term)
        weights = exp_term / sum_exp
        
        # Gradient of ac with respect to x
        # ac_k = sum_i x_i x_{k-i}
        # d(ac_k)/dx_j = x_{k-j} + x_{k-j} (if the term appears twice?)
        # No, ac_k = x_0 x_k + x_1 x_{k-1} + ... + x_k x_0
        # If j is in [0, N-1] and k-j is in [0, N-1]:
        # d(ac_k)/dx_j = x_{k-j}
        # If k-j == j, it's 2*x_j? No, in sum_i, if i=j, term is x_j x_{k-j}.
        # If k-j != j, derivative is x_{k-j}.
        # If k-j == j, derivative is 2*x_j.
        
        # Let's construct the gradient matrix implicitly.
        # grad_x_j = sum_k weights_k * sign(ac_k) * d(ac_k)/dx_j
        # = sum_k weights_k * sign(ac_k) * (x_{k-j} + delta_{k-j, j} x_j)
        
        # This is a convolution-like operation.
        # Let w_k = weights_k * sign(ac_k)
        # grad_x_j = sum_k w_k x_{k-j} + sum_k w_k delta_{k-j, j} x_j
        # The second term is x_j * sum_k w_k delta_{k-j, j} = x_j * w_{2j} (if 2j in range)
        
        # The first term is a cross-correlation of x and w.
        # corr[j] = sum_k w_k x_{k-j}
        
        # Let's use FFT for the correlation.
        # corr = ifft(fft(w, 2*N) * fft(x, 2*N).conj()).real
        # Note: correlation is ifft(fft(a) * fft(b).conj())
        
        size = 2 * N
        W = np.fft.fft(w, size)
        X = np.fft.fft(x, size)
        corr = np.fft.ifft(W * X.conj()).real[:N]
        
        # The second term: x_j * w_{2j}
        # We need w_{2j} for j in 0..N-1.
        # w is defined on 0..2N-2.
        # If 2j > 2N-2, w_{2j} is 0.
        diag_term = np.zeros(N)
        for j in range(N):
            if 2*j < len(w):
                diag_term[j] = x[j] * w[2*j]
            # Wait, is it x_j * w_{2j}?
            # d(ac_{2j})/dx_j = 2*x_j.
            # So term is weights_{2j} * sign(ac_{2j}) * 2*x_j.
            # My formula above had delta * x_j. So it should be 2*x_j * w_{2j}.
            # Let's re-verify.
            # ac_k = sum_i x_i x_{k-i}
            # d(ac_k)/dx_j = x_{k-j} + (if k-j==j) x_j.
            # If k=2j, k-j=j. So d(ac_{2j})/dx_j = x_j + x_j = 2x_j.
            # So the term is w_{2j} * 2*x_j.
            # My previous line `diag_term[j] = x[j] * w[2*j]` is missing the factor 2.
            # Let's correct.
            if 2*j < len(w):
                diag_term[j] = 2 * x[j] * w[2*j]
        
        grad_ac_part = corr + diag_term
        
        # Now gradient of -2*log(|sum(x)|)
        s = np.sum(x)
        if s == 0:
            grad_sum_part = np.zeros(N)
        else:
            grad_sum_part = -2 / s * np.ones(N)
        
        return grad_ac_part + grad_sum_part

    # Initial guess: random signs
    np.random.seed(42)
    best_x = np.random.choice([-1, 1], N)
    best_score = metric(best_x)
    
    # Optimization loop
    # Use projected gradient descent.
    # The variables are continuous, but the "step functions" implies discrete?
    # "signed equal-width step functions" usually implies the heights are from a set, e.g. {-1, 1}.
    # However, the task says "construct() -> sequence of signed step heights".
    # The seed returns np.ones(N).
    # The metric is defined for the sequence.
    # If we allow continuous heights, we can scale them.
    # Note: metric(x) is invariant to scaling x -> c*x.
    # max(|ac(c*x)|) / (sum(c*x))^2 = c^2 max(|ac(x)|) / (c^2 sum(x)^2) = metric(x).
    # So we can fix the scale, e.g., sum(x^2) = 1 or max(|x|)=1.
    
    # Let's optimize continuous x, then project to signs?
    # Or just optimize continuous x.
    # The "signed step heights" might imply the values are restricted.
    # But the seed is ones.
    # Let's try optimizing continuous values first.
    
    x = np.random.randn(N)
    # Normalize
    x = x / np.linalg.norm(x)
    
    lr = 0.1
    for i in range(10000):
        if time.time() - t_start > DEADLINE_MARGIN:
            break
            
        g = gradient(x)
        x_new = x - lr * g
        # Project to unit sphere? Or just keep as is?
        # Since scale invariant, let's keep norm fixed to avoid drift.
        norm = np.linalg.norm(x_new)
        if norm > 1e-12:
            x_new = x_new / norm
        else:
            x_new = x
            
        # Check improvement
        score_new = metric(x_new)
        if score_new < best_score:
            best_score = score_new
            best_x = x_new.copy()
            
        x = x_new
        # Decay learning rate
        lr *= 0.999
        
    # Try to project to {-1, 1} and see if it's better
    x_sign = np.sign(best_x)
    # Handle zeros
    x_sign[x_sign == 0] = 1
    score_sign = metric(x_sign)
    if score_sign < best_score:
        best_x = x_sign
        best_score = score_sign
        
    return best_x
# EVOLVE-BLOCK-END
