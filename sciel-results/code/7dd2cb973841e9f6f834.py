import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    # np.correlate(h, h, 'full') gives lags from -(N-1) to N-1.
    # Index N-1 corresponds to lag 0.
    corr = np.correlate(h, h, mode='full')
    mask = np.ones(len(corr), dtype=bool)
    mask[N-1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _soft_max(x, k):
    """
    Soft maximum approximation using log-sum-exp.
    max(x) ~ (1/k) * log(sum(exp(k*x)))
    """
    if len(x) == 0:
        return 0.0
    # Scale for numerical stability
    max_val = np.max(x)
    if max_val > 1e10:
        return max_val
    # Avoid overflow
    x_scaled = k * (x - max_val)
    # Clip to prevent overflow in exp
    x_scaled = np.clip(x_scaled, -50, 50)
    sum_exp = np.sum(np.exp(x_scaled))
    if sum_exp == 0:
        return max_val
    return max_val + np.log(sum_exp) / k

def _objective_and_grad(x, k=10.0):
    """
    Objective function and its gradient for continuous optimization.
    x is a vector of length N.
    We want to minimize max_{k!=0} |corr_k(x)|.
    """
    # Compute autocorrelation
    # x*x is convolution. We need circular or linear?
    # The problem implies linear convolution on the interval, but since it's a step function
    # on a finite domain, usually we consider the linear autocorrelation.
    # However, for FFT-based gradient computation, circular convolution is easier.
    # If we pad x with zeros to length 2N-1, circular conv of length 2N-1 is linear conv.
    # Let's stick to linear conv for correctness, but compute gradient via FFT if possible.
    # For N=400, direct conv is O(N^2) = 160k ops, which is fast.
    # But we need gradient.
    
    # Let's use the property that the gradient of ||x*x||_inf is related to the subgradient.
    # Using soft max: f(x) = (1/k) log(sum exp(k |c_i|)) where c_i are non-zero lags of conv.
    # This is differentiable.
    
    # To compute gradient efficiently, we can use FFT.
    # Let X = FFT(x, 2*N)
    # C = X * X (circular convolution of length 2N)
    # The linear convolution c_lin is the first N + (N-1) elements of the circular convolution of length 2N-1?
    # Actually, if we pad x to length 2N, the circular convolution length 2N contains the linear convolution
    # in the first 2N-1 elements? No.
    # Let M = 2*N.
    # x_padded = [x, 0, ..., 0] of length M.
    # y = ifft(fft(x_padded) * fft(x_padded))
    # y contains the circular convolution.
    # The linear convolution is y[0:N+N-1] if we consider x as length N.
    # Wait, standard linear conv of two length N sequences has length 2N-1.
    # If we embed in length M >= 2N-1, circular conv matches linear conv for indices 0..M-1.
    # So let M = 2*N.
    
    M = 2 * N
    x_padded = np.zeros(M)
    x_padded[:N] = x
    
    X = np.fft.fft(x_padded)
    Y = np.fft.ifft(X * X).real
    
    # Y[0] is lag 0 (sum x^2). We want lags 1..N-1 and -(N-1)..-1.
    # In circular conv of length M, lag k corresponds to index k if k>=0, and M-k if k<0.
    # We want |Y_k| for k in {1, ..., N-1, M-N+1, ..., M-1}.
    # Note: Y is real because x is real and we take conjugate? No, X*X is not conjugate.
    # Autocorrelation is x * reverse(x).
    # FFT of reverse(x) is conj(X) if x is real?
    # Let's use correlation definition: R[k] = sum x[n] x[n+k].
    # This is conv(x, reverse(x)).
    # Let r = x[::-1].
    # R = conv(x, r).
    # FFT(R) = FFT(x) * FFT(r) = X * conj(X) = |X|^2.
    # So Y = ifft(|X|^2) gives the circular autocorrelation.
    
    X = np.fft.fft(x_padded)
    Y = np.fft.ifft(X * np.conj(X)).real
    
    # Extract non-zero lags
    # Lags: 1, ..., N-1, and -1, ..., -(N-1).
    # Indices in Y:
    # Positive lags: 1 to N-1 -> indices 1 to N-1
    # Negative lags: -1 to -(N-1) -> indices M-1 down to M-(N-1)
    
    indices_pos = np.arange(1, N)
    indices_neg = np.arange(M - (N - 1), M)
    
    vals = np.concatenate([Y[indices_pos], Y[indices_neg]])
    
    # Objective: soft max of |vals|
    abs_vals = np.abs(vals)
    f_val = _soft_max(abs_vals, k)
    
    # Gradient
    # d/dx f = (1/k) * sum_i (k * sign(vals_i) * exp(k*|vals_i|) / sum_j exp(k*|vals_j|)) * d|vals_i|/dx
    # Let w_i = exp(k*|vals_i|) / sum_j exp(k*|vals_j|)
    # Then grad f = sum_i w_i * sign(vals_i) * d(vals_i)/dx
    
    max_val = np.max(abs_vals)
    if max_val > 1e10:
        # If overflow, just return the max index gradient?
        # For simplicity, if we are here, the value is huge.
        idx_max = np.argmax(abs_vals)
        # We need to map idx_max back to the original lag index
        # And compute gradient of that specific lag
        # This is a fallback. Usually k=10 is safe for N=400 values ~ N.
        # Max val ~ 400. k*400 = 4000. exp(4000) overflows.
        # So we MUST use the stable log-sum-exp formulation for weights too.
        
        # w_i = exp(k*(|vals_i| - max_val)) / sum_j exp(k*(|vals_j| - max_val))
        exp_terms = np.exp(k * (abs_vals - max_val))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return f_val, np.zeros_like(x)
        weights = exp_terms / sum_exp
        
    else:
        exp_terms = np.exp(k * (abs_vals - max_val))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return f_val, np.zeros_like(x)
        weights = exp_terms / sum_exp
        
    # We need d(vals_i)/dx.
    # vals is a subset of Y.
    # Y = ifft(|X|^2).
    # This gradient calculation is complex to derive analytically in one step.
    # Alternative: Use numerical gradient? No, too slow.
    # Alternative: Use the fact that Y is linear in X^2? No.
    
    # Let's use a simpler approach for the gradient:
    # The gradient of the soft max of a linear function Ax is A^T * weights.
    # Here, vals = L(Y) where L extracts lags.
    # Y = F^{-1}(|F(x_padded)|^2).
    # This is not linear in x.
    
    # Let's use the chain rule:
    # grad_x f = sum_i w_i * sign(vals_i) * grad_x vals_i
    # vals_i is a component of the autocorrelation.
    # R[k] = sum_n x[n] x[n+k].
    # dR[k]/dx[m] = x[m+k] + x[m-k] (with boundary conditions).
    # Since we are using circular convolution in the FFT domain, the boundaries are circular.
    # But we want linear autocorrelation.
    # The FFT method computes circular autocorrelation of the padded signal.
    # Since the signal is zero-padded to length 2N, the circular autocorrelation
    # matches the linear autocorrelation for lags 0 to N-1.
    # Specifically, R_circ[k] = R_lin[k] for k in [0, N-1].
    # And R_circ[k] = R_lin[k-N] for k in [N, 2N-1]?
    # Actually, for zero-padded x of length 2N:
    # R_circ[k] = sum_{n=0}^{2N-1} x[n] x[(n+k) mod 2N].
    # If x[n]=0 for n>=N, then for k in 1..N-1:
    # R_circ[k] = sum_{n=0}^{N-1-k} x[n] x[n+k]. This is R_lin[k].
    # For k in N+1..2N-1 (which correspond to negative lags -N+1..-1):
    # Let k' = k - 2N (negative).
    # R_circ[k] = sum_{n=2N-k}^{2N-1} x[n] x[n+k-2N] + ...
    # This is getting messy.
    
    # Simpler: Compute gradient numerically? No.
    # Let's use the property that the gradient of the autocorrelation peak can be approximated.
    # Or, we can just use the discrete local search on the rounded result, which is robust.
    # The gradient descent is just to find a good starting point for the local search.
    # We can use a simpler objective: sum of squares of non-zero lags?
    # min sum_{k!=0} R[k]^2.
    # This is equivalent to minimizing the energy in the sidelobes.
    # The gradient of sum R[k]^2 is easier?
    # R[k] = x * x_shifted.
    # Let's stick to the soft max but compute gradient via FFT trick?
    
    # There is a known trick for gradient of ||F^{-1}(|F(x)|^2)||.
    # But let's just return a zero gradient if we can't compute it easily, 
    # and rely on the fact that L-BFGS-B will use numerical gradients if we don't provide one?
    # No, L-BFGS-B uses provided gradient. If we provide None, it uses numerical.
    # Numerical gradient for N=400 is 400 evaluations.
    # Each evaluation is FFT (O(N log N)).
    # 400 * 400 log 400 ~ 1.6M ops. Very fast.
    # So we can just return the function value and let scipy compute the gradient.
    
    return f_val, None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_h = None
    best_score = float('inf')

    # Strategy:
    # 1. Generate several random continuous initializations.
    # 2. Optimize soft_max(|autocorr|) using L-BFGS-B (numerical grad).
    # 3. Round to +/- 1.
    # 4. Local search (bit flipping) on the rounded vector.
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Random initialization
        # Mix of random +/- 1 and random continuous
        if i % 2 == 0:
            x0 = np.random.choice([-1, 1], size=N).astype(float)
        else:
            x0 = np.random.randn(N)
            
        # Optimize
        # We use numerical gradient because analytic is complex
        res = minimize(_objective_and_grad, x0, method='L-BFGS-B', 
                       options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
        
        x_opt = res.x
        
        # Round to +/- 1
        h = np.sign(x_opt)
        # Handle zeros
        h[h == 0] = 1
        
        # Local search refinement
        score = _calc_score(h)
        
        # Simple local search: flip bits that improve score
        improved = True
        max_iter_local = 500
        while improved and time.time() < deadline - 10:
            improved = False
            # Try flipping a random subset or all?
            # All is N * O(N) = O(N^2). N=400, 160k ops per pass.
            # 500 passes = 80M ops. Might be too slow.
            # Let's do a few passes.
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                for j in range(N):
                    h[j] *= -1
                    new_score = _calc_score(h)
                    if new_score < score:
                        score = new_score
                        improved = True
                    else:
                        h[j] *= -1
                        
                if not improved:
                    break
            
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # If we found a very good score, break early?
        if best_score < 0.001:
            break

    if best_h is None:
        best_h = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
        best_score = _calc_score(best_h)

    return best_h
