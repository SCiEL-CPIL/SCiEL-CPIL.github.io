"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a signed step sequence minimizing the maximum absolute autoconvolution
    divided by squared integral.
    
    Strategy:
    1. Use time-reversal symmetry h_i = h_{N-1-i} to reduce dimensionality.
    2. Optimize a continuous proxy using projected gradient descent with soft-max surrogate.
    3. Refine with local sign-flip search.
    
    Returns:
        np.ndarray of shape (N,) with signed step heights
    """
    start_time = time.time()
    deadline = 130  # Safety margin before 150s limit
    
    # Half length for symmetric representation
    half = N // 2
    
    # Initialize with random symmetric pattern
    np.random.seed(42)
    x = np.zeros(half)
    x[:] = np.random.choice([-1, 1], size=half)
    
    def build_full(x_half):
        """Build full symmetric vector from half vector."""
        if N % 2 == 0:
            full = np.empty(N)
            full[:half] = x_half
            full[half:] = x_half[::-1]
        else:
            full = np.empty(N)
            full[:half] = x_half
            full[half:] = x_half[:half-1][::-1]
        return full
    
    def autoconvolution(h):
        """Compute autoconvolution of h."""
        return np.convolve(h, h, mode='full')
    
    def objective(x_half, beta=10.0):
        """
        Objective: soft-max approximation of max |autoconv| / (sum(h))^2
        """
        full = build_full(x_half)
        ac = autoconvolution(full)
        integral = np.sum(full)
        if abs(integral) < 1e-10:
            return 1e10
        # Soft-max of |ac| / integral^2
        ac_abs = np.abs(ac)
        max_val = np.max(ac_abs)
        # Soft-max: (1/beta) * log(sum(exp(beta * ac_abs / integral^2)))
        scaled = beta * ac_abs / (integral * integral)
        # Numerically stable log-sum-exp
        m = np.max(scaled)
        if m > 700:
            m = 700
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        return lse / beta
    
    def gradient(x_half, beta=10.0):
        """
        Gradient of the soft-max objective w.r.t. x_half.
        """
        full = build_full(x_half)
        ac = autoconvolution(full)
        integral = np.sum(full)
        if abs(integral) < 1e-10:
            return np.zeros(half)
        
        ac_abs = np.abs(ac)
        scaled = beta * ac_abs / (integral * integral)
        m = np.max(scaled)
        if m > 700:
            m = 700
        exp_scaled = np.exp(scaled - m)
        weights = exp_scaled / np.sum(exp_scaled)
        
        # Gradient of soft-max w.r.t. ac_abs
        grad_ac = weights * beta / (integral * integral)
        
        # Gradient of |ac| w.r.t. ac
        sign_ac = np.sign(ac)
        grad_ac = grad_ac * sign_ac
        
        # Gradient of ac w.r.t. h: ac[k] = sum_i h[i] * h[k-i]
        # d(ac[k])/d(h[j]) = h[k-j] + h[j-k] (for valid indices)
        # We need to map back to x_half
        
        grad_full = np.zeros(N)
        for k in range(len(ac)):
            if grad_ac[k] == 0:
                continue
            # ac[k] = sum_i h[i]*h[k-i] where 0<=i<N and 0<=k-i<N
            i_min = max(0, k - N + 1)
            i_max = min(N - 1, k)
            for i in range(i_min, i_max + 1):
                # d(ac[k])/d(h[i]) = h[k-i]
                # d(ac[k])/d(h[k-i]) = h[i]
                grad_full[i] += grad_ac[k] * full[k - i]
                if k - i != i:
                    grad_full[k - i] += grad_ac[k] * full[i]
        
        # Map grad_full to grad_x_half using symmetry
        grad_x = np.zeros(half)
        for i in range(half):
            # x_half[i] affects full[i]
            grad_x[i] += grad_full[i]
            # x_half[i] also affects full[N-1-i]
            j = N - 1 - i
            if j < N:
                grad_x[i] += grad_full[j]
        
        # Gradient of 1/(integral^2) factor
        # d/dx [softmax / integral^2] = (1/integral^2) * d(softmax)/dx + softmax * d(1/integral^2)/dx
        # d(1/integral^2)/dx_i = -2 * (sum_j dh_j/dx_i) / integral^3
        # For symmetric: d(integral)/d(x_half[i]) = 2 (or 1 if center)
        d_integral_dx = np.ones(half) * 2
        if N % 2 == 1:
            d_integral_dx[half - 1] = 1  # Center element only appears once
        
        obj_val = objective(x_half, beta)
        grad_x = grad_x / (integral * integral) - 2 * obj_val * d_integral_dx / (integral ** 3)
        
        return grad_x
    
    # Projected gradient descent
    best_x = x.copy()
    best_obj = objective(x)
    lr = 0.1
    steps = 500
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        g = gradient(x)
        x_new = x - lr * g
        # Project to [-1, 1]
        x_new = np.clip(x_new, -1, 1)
        
        obj_new = objective(x_new)
        if obj_new < best_obj:
            best_obj = obj_new
            best_x = x_new.copy()
        x = x_new
        
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            lr *= 0.9
    
    # Round to signs
    x_rounded = np.sign(best_x)
    # Handle zeros
    x_rounded[x_rounded == 0] = 1
    
    # Local sign-flip refinement
    def score(h):
        ac = autoconvolution(h)
        integral = np.sum(h)
        if abs(integral) < 1e-10:
            return 1e10
        return np.max(np.abs(ac)) / (integral * integral)
    
    current_score = score(build_full(x_rounded))
    improved = True
    max_iter = 200
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(half):
            if time.time() > deadline:
                break
            # Try flipping sign at position i
            old_val = x_rounded[i]
            x_rounded[i] = -old_val
            new_score = score(build_full(x_rounded))
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                x_rounded[i] = old_val
    
    # Try more aggressive random restarts if time permits
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            x_r = np.random.choice([-1, 1], size=half)
            # Quick gradient descent
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                g = gradient(x_r)
                x_r = np.clip(x_r - 0.1 * g, -1, 1)
            x_r = np.sign(x_r)
            x_r[x_r == 0] = 1
            s_r = score(build_full(x_r))
            if s_r < current_score:
                current_score = s_r
                x_rounded = x_r.copy()
                # Local refinement
                improved2 = True
                while improved2 and time.time() < deadline - 5:
                    improved2 = False
                    for i in range(half):
                        if time.time() > deadline - 5:
                            break
                        old_val = x_rounded[i]
                        x_rounded[i] = -old_val
                        new_score = score(build_full(x_rounded))
                        if new_score < current_score - 1e-12:
                            current_score = new_score
                            improved2 = True
                        else:
                            x_rounded[i] = old_val
    
    return build_full(x_rounded)
# EVOLVE-BLOCK-END
