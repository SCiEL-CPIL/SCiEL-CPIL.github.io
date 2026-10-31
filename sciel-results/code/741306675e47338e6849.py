import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 145

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    k = np.arange(N)
    
    def compute_autoconv(h):
        # Linear autoconvolution
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv

    def smooth_max_obj_and_grad(coeffs, tau=0.1):
        # coeffs: a_0 ... a_M
        # h_k = sum a_m cos(2 pi m k / N)
        h = np.zeros(N)
        for m in range(len(coeffs)):
            if m == 0:
                h += coeffs[0]
            else:
                h += coeffs[m] * np.cos(2 * np.pi * m * k / N)
        
        # Enforce non-negativity via softplus or clip?
        # Clipping breaks gradient. 
        # We assume the optimizer stays in a region where h > 0.
        # If h goes negative, we penalize heavily.
        neg_penalty = np.sum(np.maximum(-h, 0)**2)
        
        s = np.sum(h)
        if s < 1e-10:
            return 1e10 + neg_penalty, np.zeros(len(coeffs))
        
        h_norm = h / s
        c = compute_autoconv(h_norm)
        c_max = np.max(c)
        
        # Smooth max
        exp_c = np.exp((c - c_max) / tau)
        sum_exp = np.sum(exp_c)
        obj = tau * (np.log(sum_exp) + c_max / tau) + 1e6 * neg_penalty
        
        # Gradient
        w = exp_c / sum_exp
        
        # Gradient of obj w.r.t h_norm
        # d(obj)/dh_norm = 2 * (w * h_norm)_{corr}
        # corr[w, h]_l = sum_k w_k h_{k-l}
        # We need G_h[l] for l=0..N-1
        L = 4 * N
        w_pad = np.zeros(L)
        w_pad[:len(w)] = w
        h_pad = np.zeros(L)
        h_pad[:N] = h_norm
        
        W = np.fft.fft(w_pad, n=L)
        H = np.fft.fft(h_pad, n=L)
        
        # Cross correlation
        corr_full = np.fft.ifft(W * np.conj(H)).real
        G_h = 2.0 * corr_full[0:N]
        
        # Gradient w.r.t coeffs
        grad_c = np.zeros(len(coeffs))
        for m in range(len(coeffs)):
            if m == 0:
                grad_c[m] = np.sum(G_h)
            else:
                grad_c[m] = np.dot(G_h, np.cos(2 * np.pi * m * k / N))
                
        # Add gradient of penalty
        if neg_penalty > 0:
            grad_penalty = -2 * np.maximum(-h, 0)
            # d(neg_penalty)/dc_m
            for m in range(len(coeffs)):
                if m == 0:
                    grad_c[m] += 1e6 * np.sum(grad_penalty)
                else:
                    grad_c[m] += 1e6 * np.dot(grad_penalty, np.cos(2 * np.pi * m * k / N))
                    
        return obj, grad_c

    # M = number of cosine terms
    M = 20
    
    def h_to_coeffs(h):
        H_freq = np.fft.rfft(h)
        c = np.zeros(M + 1)
        c[0] = H_freq[0].real / N
        for m in range(1, M + 1):
            if m < len(H_freq):
                c[m] = 2 * H_freq[m].real / N
        return c

    def coeffs_to_h(c):
        h = np.zeros(N)
        for m in range(len(c)):
            if m == 0:
                h += c[0]
            else:
                h += c[m] * np.cos(2 * np.pi * m * k / N)
        return h

    # Generate initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Bimodal
    for sigma in [N/10, N/8, N/6]:
        for alpha in [0.1, 0.2, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((k - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((k - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
            
    # 3. W-shape
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * k / N)
    inits.append(np.maximum(w_shape, 0.01))
    
    # 4. Random Smooth
    for _ in range(5):
        freqs = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amps = np.random.uniform(0.5, 1.5, size=3)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * k / N + p)
        inits.append(np.maximum(random_shape, 0.01))
        
    best_h = None
    best_val = np.inf
    
    # Evaluate inits
    for h_init in inits:
        if time_left() < 10: break
        h_norm = h_init / np.sum(h_init)
        c = compute_autoconv(h_norm)
        val = np.max(c)
        if val < best_val:
            best_val = val
            best_h = h_norm.copy()
            
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = np.max(compute_autoconv(best_h))

    # Optimize each init
    # We have ~15 inits. 
    # Time per opt: ~5s. Total 75s.
    # Budget left: 145 - 10 (init) - 5 (safety) = 130s.
    
    for h_init in inits:
        if time_left() < 15: break
        
        c_init = h_to_coeffs(h_init / np.sum(h_init))
        
        def obj_func(c):
            val, grad = smooth_max_obj_and_grad(c, tau=0.1)
            return val
        
        def grad_func(c):
            val, grad = smooth_max_obj_and_grad(c, tau=0.1)
            return grad
            
        try:
            # Use L-BFGS-B
            res = minimize(obj_func, c_init, jac=grad_func, method='L-BFGS-B', 
                           options={'maxiter': 50, 'maxfun': 100})
            
            h_res = coeffs_to_h(res.x)
            h_res = np.maximum(h_res, 0)
            s = np.sum(h_res)
            if s < 1e-10: continue
            h_res = h_res / s
            
            c_res = compute_autoconv(h_res)
            val_res = np.max(c_res)
            
            if val_res < best_val:
                best_val = val_res
                best_h = h_res.copy()
                
        except Exception as e:
            pass

    return best_h.tolist()

construct()
