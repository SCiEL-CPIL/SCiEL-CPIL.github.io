import numpy as np
import time

N = 600
DELTA = 1.0 / 1200.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # The objective is min max(h*h) / sum(h^2).
    # This is equivalent to finding the smallest epsilon such that there exists h >= 0
    # with sum(h^2) = 1 and max(h*h) <= epsilon.
    
    # We use a projection method (similar to POCS - Projection Onto Convex Sets)
    # Sets:
    # 1. C1: h >= 0
    # 2. C2: ||h||_2 = 1 (Sphere)
    # 3. C3: max(h*h) <= epsilon (Ball in l-infinity norm of convolution)
    
    # Since C3 depends on epsilon, we can try to find a feasible h for a given epsilon,
    # or better, use a method that adapts epsilon.
    # However, a simpler and often effective approach for "minimize max convolution"
    # is to iterate:
    # 1. Start with h
    # 2. Compute conv = h*h
    # 3. If max(conv) > 1, scale h by 1/sqrt(max(conv)) to make max(conv)=1?
    #    No, we want to minimize max(conv) subject to ||h||=1.
    #    So we want to push the mass of h away from where the convolution peaks.
    
    # Alternative Strategy: Gradient Descent on the "Max" part with smoothing.
    # But let's try a different structural approach: 
    # The optimal function for minimizing the L-infinity norm of the autoconvolution
    # given L2 norm is related to the "flattest" possible convolution.
    # A known good candidate is the "triangular" or "tent" function, but we are on a finite interval.
    
    # Let's use a specific iterative algorithm:
    # 1. Initialize h
    # 2. Compute c = h*h
    # 3. Identify peaks in c.
    # 4. Reduce h at indices that contribute most to these peaks.
    # 5. Renormalize h to L2=1.
    # This is a form of coordinate descent or subgradient method.
    
    # Better: Use the fact that min max(c) s.t. ||h||=1, h>=0 is convex?
    # The feasible set {h: h>=0, ||h||=1, max(h*h)<=M} is not convex due to ||h||=1.
    # But {h: h>=0, ||h||<=1, max(h*h)<=M} is convex? No, h*h is quadratic.
    
    # Let's stick to a robust heuristic that is structurally different from L-BFGS on smooth windows.
    # We will use a "Peak Reduction" iterative scheme.
    
    def compute_conv(h):
        # Use FFT for speed
        # Zero pad to length 2N to avoid circular convolution artifacts
        n_fft = 2 * N
        h_fft = np.fft.rfft(h, n=n_fft)
        conv_fft = h_fft * h_fft
        conv = np.fft.irfft(conv_fft, n=n_fft)[:2*N-1]
        return conv

    def score(h):
        ss = np.sum(h**2)
        if ss < 1e-10:
            return 1e10
        conv = compute_conv(h)
        return np.max(conv) / ss

    # Initial guess: Gaussian
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = np.exp(-100 * x**2)
    h = np.maximum(h, 0)
    h /= np.linalg.norm(h)
    
    best_h = h.copy()
    best_score = score(best_h)
    
    # Iterative Peak Reduction
    # Idea: The peaks in the convolution are caused by large values of h at symmetric positions.
    # We can iteratively dampen the "worst" elements of h.
    
    for _ in range(500): # Max iterations
        if time.time() > deadline - 5:
            break
            
        h_curr = best_h.copy()
        conv = compute_conv(h_curr)
        max_val = np.max(conv)
        
        if max_val < 1e-6:
            break
            
        # Find indices in conv that are close to max
        # The contribution of h[i] to conv[k] is sum_j h[j]h[k-j].
        # We want to reduce h where it causes high conv values.
        
        # Simple heuristic: 
        # 1. Compute gradient of max(conv) w.r.t h? 
        #    If we approximate max(conv) by sum(exp(beta * conv)), we get a smooth objective.
        #    Let's try Log-Sum-Exp approximation for the max.
        
        # Objective: f(h) = log(sum(exp(beta * (h*h)))) / ||h||^2
        # We can do gradient descent on this.
        
        beta = 50.0 # Sharpness of max approximation
        
        # Compute smooth max
        conv_scaled = beta * conv
        # Shift to avoid overflow
        max_c = np.max(conv_scaled)
        sum_exp = np.sum(np.exp(conv_scaled - max_c))
        smooth_max = (max_c + np.log(sum_exp)) / beta
        
        # Gradient of smooth_max w.r.t h
        # d/dh_i log(sum_j exp(beta * c_j)) = (1/sum_exp) * sum_j exp(beta*c_j) * beta * dc_j/dh_i
        # c_j = sum_k h_k h_{j-k}
        # dc_j/dh_i = h_{j-i} + h_{j+i} (with appropriate bounds)
        
        # This is equivalent to:
        # grad = (2 * beta / sum_exp) * (w * h)
        # where w is the vector such that (w*h)_i = sum_j exp(beta*c_j) * (h_{j-i} + h_{j+i}) / 2 ?
        # Actually, let's use the correlation property.
        # c = h*h.
        # d c / d h is a Toeplitz-like matrix.
        # Let w = (1/sum_exp) * exp(beta * conv)
        # Then grad_smooth_max = 2 * beta * (w * h) where * is circular convolution?
        # No, linear convolution.
        
        # Let's compute g = w * h (linear convolution)
        # Then grad = 2 * beta * g
        # Wait, c_j = sum_k h_k h_{j-k}.
        # d c_j / d h_i = h_{j-i} + h_{j+i} (if indices valid).
        # This is exactly (h * h)_i? No.
        # (h*h)_i = sum_k h_k h_{i-k}.
        # d/dh_i (sum_j w_j c_j) = sum_j w_j (h_{j-i} + h_{j+i})
        # This is 2 * (w * h)_i if w is symmetric? No.
        # Let's just compute it via FFT for the gradient step.
        
        w = np.exp(conv_scaled - max_c) / sum_exp
        
        # Compute gradient
        # grad_i = sum_j w_j (h_{j-i} + h_{j+i})
        # This is 2 * (w * h) if we consider circular, but we have linear.
        # Let's use np.convolve
        # term1 = w * h (linear)
        # term2 = w * h (linear) but shifted?
        # Actually, d/dh_i sum_j w_j sum_k h_k h_{j-k}
        # = sum_j w_j (h_{j-i} + h_{j+i})
        # Let u = w.
        # sum_j u_j h_{j-i} = (u*h)_i (linear conv)
        # sum_j u_j h_{j+i} = (u*h)_{-i}? No.
        # Let v = h.
        # sum_j u_j v_{j-i} = (u*v)_i
        # sum_j u_j v_{j+i} = (u*v)_{?}
        # Let k = j+i => j = k-i. Sum_k u_{k-i} v_k = (u_rev * v)_i?
        
        # Simpler:
        # grad = 2 * (w * h) is only true for circular or if we are careful.
        # Let's use the property that for real signals, the gradient of ||h*h||_2^2 is 4*(h*h*h).
        # Here we have weighted max.
        
        # Let's just use finite differences or a simpler heuristic if gradient is complex.
        # Heuristic: Reduce h at points where h is large AND the local convolution is high.
        
        # Local convolution at i is roughly h[i] * sum(h) ? No.
        # Let's stick to the L-Sum-Exp gradient which is standard.
        # grad = 2 * beta * (w * h) where * is linear convolution?
        # Let's verify:
        # f = sum_j w_j (h*h)_j
        # df/dh_i = sum_j w_j (h_{j-i} + h_{j+i})
        # Let g = w * h (linear). g_k = sum_m w_m h_{k-m}.
        # g_i = sum_m w_m h_{i-m}
        # h_{j-i} = h_{(j)-i}. Let m = j-i => j = m+i.
        # sum_j w_j h_{j-i} = sum_m w_{m+i} h_m. This is not simply g.
        
        # Given the complexity and time limit, let's use a simpler "water-filling" or "peak shaving" approach.
        # 1. Compute conv
        # 2. Find indices i where conv is high.
        # 3. The peak at conv[k] is driven by h[k/2] if k is even, etc.
        # 4. Reduce h at indices that are "central" to the high conv peaks.
        
        # Identify high conv peaks
        threshold = max_val * 0.9
        peak_indices = np.where(conv > threshold)[0]
        
        # For each peak index p, the main contributors are h[p//2] (if p even) or nearby.
        # Actually, the derivative of conv[p] w.r.t h[i] is h[p-i].
        # So we should reduce h[i] proportional to h[p-i] for peaks p.
        
        # Accumulate reduction signal
        reduction = np.zeros(N)
        for p in peak_indices:
            # d conv[p] / d h = h[p::-1] (reversed h up to p)
            # We want to reduce h in the direction of the gradient.
            # Gradient of conv[p] w.r.t h is h[p::-1] (padded)
            # Let's just add the reversed h segment
            if p < 2*N - 1:
                # The contribution of h[0..N-1] to conv[p]
                # conv[p] = sum_{i} h[i] h[p-i]
                # d conv[p] / d h[i] = h[p-i] (if 0 <= p-i < N)
                # So we want to reduce h[i] by amount proportional to h[p-i]
                start_idx = max(0, p - N + 1)
                end_idx = min(N, p + 1)
                for i in range(start_idx, end_idx):
                    j = p - i
                    if 0 <= j < N:
                        reduction[i] += h[j]
        
        # Normalize reduction
        max_red = np.max(reduction)
        if max_red > 0:
            reduction /= max_red
            
        # Update h
        # h_new = h - step * reduction
        # But we must keep h >= 0
        step_size = 0.1
        h_new = h_curr - step_size * reduction
        h_new = np.maximum(h_new, 0)
        
        # Renormalize to L2=1
        norm = np.linalg.norm(h_new)
        if norm < 1e-10:
            break
        h_new /= norm
        
        s_new = score(h_new)
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
        else:
            # If it didn't improve, try a smaller step or break
            # To save time, we might just break if no progress in many steps
            # But let's try a few more with smaller step
            pass
            
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    print(len(res))
