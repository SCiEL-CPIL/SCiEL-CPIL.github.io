import numpy as np
import time
from scipy.optimize import minimize, lsq_linear

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    def compute_value(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))
    
    # Strategy:
    # 1. Normalize h so that sum(h) = 1 (scale invariance)
    # 2. The objective becomes 2*N*max(h*h)
    # 3. We want to minimize max(h*h). Since max of a vector is convex in the vector,
    #    and h*h is a quadratic form, this is not directly convex in h.
    #    However, we can use a smooth surrogate: soft-max via log-sum-exp.
    #    Let f(h) = log-sum-exp(beta * (h*h)) for large beta.
    #    Then grad f = (h*h * exp(beta*(h*h))) / sum(exp(beta*(h*h))) * 2h
    #    This is differentiable and we can use L-BFGS-B.
    # 4. We also need to handle the constraint sum(h)=1.
    #    We can parameterize h as [1 + x_1, ..., 1 + x_{N-1}, 1 - sum(x_i)] to enforce sum=1.
    #    But this couples all variables. Alternatively, use a penalty or use constrained optimization.
    #    Let's use the reparameterization: h_i = (1 + x_i) / (1 + sum(x)/N) ... no that's messy.
    #    Better: use Lagrange multipliers or just add a large penalty term (sum(h)-1)^2.
    
    # Actually, let's use the epigraph formulation with a smooth max surrogate.
    # Objective: min_h 2*N*max_k (h*h)_k  s.t. sum(h)=1
    # Smooth surrogate: min_h 2*N * (1/beta) * log(sum_k exp(beta*(h*h)_k)) + lambda*(sum(h)-1)^2
    # But lambda is tricky. Let's use SLSQP with constraints.
    
    # For SLSQP, we need gradients.
    # Let's define the smooth max: M(h) = (1/beta) * log(sum_k exp(beta * c_k)) where c = h*h
    # dM/dh = (1/beta) * (1/sum(exp(beta*c))) * sum_k (exp(beta*c_k) * beta * dc_k/dh)
    # dc_k/dh = 2 * (h * shifted_h) where c_k = sum_i h_i h_{i-k+N-1} (for linear conv)
    # Actually, c = h*h is the convolution. The gradient of c_k w.r.t. h is:
    # dc_k/dh_j = h_{k-j} + h_{k-j+N-1} ... this is getting complicated.
    
    # Let's use FFT to compute the gradient efficiently.
    # The autoconvolution c_k = sum_i h_i h_{i-k+N-1} for k=0..2N-2
    # In frequency domain: C = F(h)^2 (element-wise, with appropriate scaling)
    # The gradient of max(c) or soft-max(c) can be computed via FFT.
    
    # Actually, let's use a simpler approach:
    # Since N=400, we can just compute the convolution directly and use finite differences
    # or analytic gradients for the soft-max.
    
    # Let's use the following smooth objective:
    # J(h) = (1/beta) * log(sum_k exp(beta * c_k))
    # where c = np.convolve(h, h)
    # grad J = (1/beta) * (1/Z) * sum_k (exp(beta*c_k) * beta * grad c_k)
    # grad c_k = 2 * (h * h_k) where h_k is h shifted by k
    # Actually, for linear convolution c = h*h, we have:
    # c_k = sum_{i} h_i h_{k-i} (with appropriate bounds)
    # d(c_k)/d(h_j) = h_{k-j} + (if k-j+N-1 in range) ... 
    # It's easier to use the fact that c = h*h is a quadratic form: c = H^T * D_k * h
    # where D_k is a shift matrix. Then grad(c_k) = 2 * D_k * h.
    
    # For efficiency, let's use FFT:
    # F(c) = F(h) * F(h) (with zero-padding to length 2N-1)
    # The gradient of the soft-max can be computed as:
    # Let w_k = exp(beta*c_k) / sum(exp(beta*c))
    # grad J = (1/beta) * sum_k w_k * beta * grad c_k = sum_k w_k * grad c_k
    # grad c_k = 2 * circulant_shift(h, k)  (approximately, for linear conv it's a bit different)
    # Actually, for linear convolution, grad c_k is not exactly a circulant shift.
    # Let's just use finite differences for the gradient if needed, or use the direct formula.
    
    # For N=400, direct computation of grad c_k is O(N^2) which is 160000, fine.
    
    beta = 50.0  # Sharpness of soft-max
    
    def smooth_objective_and_grad(h):
        h = np.asarray(h)
        s = np.sum(h)
        # Penalty for sum != 1
        penalty = 1e4 * (s - 1.0) ** 2
        penalty_grad = 2e4 * (s - 1.0) * np.ones_like(h)
        
        c = np.convolve(h, h)
        # Soft max
        beta_c = beta * c
        # Clip to avoid overflow
        beta_c = np.clip(beta_c, -50, 50)
        exp_beta_c = np.exp(beta_c)
        Z = np.sum(exp_beta_c)
        if Z == 0 or not np.isfinite(Z):
            return 1e10, np.zeros_like(h)
        
        w = exp_beta_c / Z
        soft_max_val = (1.0 / beta) * np.log(Z)
        
        # Gradient of soft-max
        # d(soft_max)/d(c_k) = w_k
        # d(c_k)/d(h_j) = 2 * (h shifted)
        # For linear convolution c = h*h:
        # c_k = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h_i * h_{k-i}
        # d(c_k)/d(h_j) = h_{k-j} + h_{k-j} (if j appears twice) ... actually:
        # If j and k-j are both in [0,N-1], then d(c_k)/d(h_j) = h_{k-j}
        # But if j == k-j (i.e., k=2j), then it's 2*h_j.
        # More carefully: c_k = sum_i h_i h_{k-i}
        # d(c_k)/d(h_j) = h_{k-j} + (if k-j != j) h_{k-j} ... no.
        # Let's write c_k = sum_i h_i h_{k-i}
        # Term i=j: h_j * h_{k-j} -> derivative w.r.t. h_j is h_{k-j}
        # Term i=k-j: h_{k-j} * h_j -> derivative w.r.t. h_j is h_{k-j}
        # If j == k-j, i.e., k=2j, then these are the same term, so derivative is 2*h_j.
        # So d(c_k)/d(h_j) = h_{k-j} if k-j in [0,N-1] and k-j != j
        #                  + h_{k-j} if k-j in [0,N-1] and k-j == j (i.e., double count)
        # Actually, it's simpler: d(c_k)/d(h_j) = h_{k-j} if 0 <= k-j <= N-1, plus
        # if k-j == j, we have an extra h_j because the term appears twice in the sum.
        # Wait, no. In the sum c_k = sum_i h_i h_{k-i}, when i=j, we get h_j h_{k-j}.
        # When i=k-j, we get h_{k-j} h_j. These are the same product, but they are
        # different terms in the sum unless j = k-j.
        # So d(c_k)/d(h_j) = (if k-j in [0,N-1]) h_{k-j} + (if k-j in [0,N-1] and j != k-j) h_{k-j}
        # = 2*h_{k-j} if k-j in [0,N-1] and j != k-j
        # = h_j if k-j == j (i.e., k=2j)
        
        # This is O(N^2) to compute the full gradient matrix, but we only need the vector.
        # Let's compute it as:
        grad_soft = np.zeros_like(h)
        for k in range(len(c)):
            if w[k] < 1e-10:
                continue
            # d(c_k)/d(h_j) for j in [0, N-1]
            # k-j must be in [0, N-1]
            j_min = max(0, k - (N - 1))
            j_max = min(N - 1, k)
            for j in range(j_min, j_max + 1):
                kj = k - j
                if 0 <= kj <= N - 1:
                    if j == kj:
                        grad_soft[j] += w[k] * 2.0 * h[j]  # Wait, is it 2*h[j] or h[j]?
                    else:
                        grad_soft[j] += w[k] * 2.0 * h[kj]
        # Actually, let me re-derive:
        # c_k = sum_{i: 0<=i<N, 0<=k-i<N} h_i h_{k-i}
        # Let S_k = {i: 0<=i<N, 0<=k-i<N} = {i: max(0,k-N+1) <= i <= min(N-1,k)}
        # c_k = sum_{i in S_k} h_i h_{k-i}
        # d(c_k)/d(h_j) = sum_{i in S_k} [delta_{i,j} h_{k-i} + h_i delta_{k-i,j}]
        # = (if j in S_k) h_{k-j} + (if k-j in S_k) h_{k-j}
        # = (if j in S_k and k-j in S_k) 2*h_{k-j}  [but if j==k-j, it's counted once in S_k but the derivative has two terms that are equal]
        # Wait, if j == k-j, then the two terms are both h_j, so d(c_k)/d(h_j) = h_j + h_j = 2*h_j.
        # If j != k-j, then d(c_k)/d(h_j) = h_{k-j} + h_{k-j} = 2*h_{k-j} (provided both indices are valid).
        # So in all cases where j in S_k (which implies k-j in S_k by definition of S_k),
        # d(c_k)/d(h_j) = 2*h_{k-j}.
        
        # So the loop above should just use 2*h[kj] always. Let me fix that.
        
        # Recompute gradient correctly:
        grad_soft = np.zeros_like(h)
        for k in range(len(c)):
            if w[k] < 1e-10:
                continue
            j_min = max(0, k - (N - 1))
            j_max = min(N - 1, k)
            for j in range(j_min, j_max + 1):
                kj = k - j
                grad_soft[j] += w[k] * 2.0 * h[kj]
        
        total_val = 2.0 * N * soft_max_val + penalty
        total_grad = 2.0 * N * grad_soft + penalty_grad
        
        return total_val, total_grad
    
    # Initial guesses
    # Start with flat, then some structured guesses
    h0 = np.ones(N) / N  # sum = 1
    
    starts = [h0.copy()]
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(3):
        h_rand = np.random.randn(N)
        h_rand -= np.mean(h_rand)
        if np.sum(h_rand) != 0:
            h_rand /= np.sum(h_rand)
        starts.append(h_rand)
    
    # Add a "center bump" guess
    x = np.linspace(0, 1, N)
    h_center = np.exp(-50 * (x - 0.5) ** 2)
    h_center /= np.sum(h_center)
    starts.append(h_center)
    
    best_h = starts[0]
    best_score = compute_value(best_h)
    
    for h_init in starts:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                smooth_objective_and_grad,
                h_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 1000}
            )
            
            if result.success or result.fun < best_score:
                h_candidate = result.x
                if np.all(np.isfinite(h_candidate)):
                    score_candidate = compute_value(h_candidate)
                    if score_candidate < best_score:
                        best_score = score_candidate
                        best_h = h_candidate.copy()
                        
        except Exception as e:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            result = minimize(
                smooth_objective_and_grad,
                best_h,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 2000}
            )
            if result.success:
                h_candidate = result.x
                if np.all(np.isfinite(h_candidate)):
                    score_candidate = compute_value(h_candidate)
                    if score_candidate < best_score:
                        best_score = score_candidate
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Best score: {compute_value(h)}")
