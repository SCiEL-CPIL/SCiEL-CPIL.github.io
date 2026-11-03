import numpy as np
import time

N = 600
DOMAIN = (-0.25, 0.25)

def compute_score(h):
    h_arr = np.array(h)
    l1 = np.sum(h_arr)
    if l1 == 0:
        return float('inf')
    conv = np.convolve(h_arr, h_arr, mode='full')
    peak = np.max(conv)
    return peak / (l1 * l1)

def gen_box():
    return [1.0] * N

def gen_gaussian(center, width, amp):
    dx = (DOMAIN[1] - DOMAIN[0]) / N
    x = np.linspace(DOMAIN[0] + dx/2, DOMAIN[1] - dx/2, N)
    h = amp * np.exp(-0.5 * ((x - center) / width) ** 2)
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    return h.tolist()

def gen_mixture(params):
    dx = (DOMAIN[1] - DOMAIN[0]) / N
    x = np.linspace(DOMAIN[0] + dx/2, DOMAIN[1] - dx/2, N)
    h = np.zeros(N)
    for c, w, a in params:
        h += a * np.exp(-0.5 * ((x - c) / w) ** 2)
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    return h.tolist()

def optimize(params, max_iters=1000):
    start_time = time.time()
    deadline = start_time + 100
    
    current_params = [p[:] for p in params]
    current_h = gen_mixture(current_params)
    current_score = compute_score(current_h)
    
    best_params = [p[:] for p in current_params]
    best_score = current_score
    best_h = current_h
    
    step_size = 0.05
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        p_idx = np.random.randint(0, len(current_params) * 3)
        param_idx = p_idx // 3
        comp_idx = p_idx % 3
        
        new_params = [p[:] for p in current_params]
        old_val = new_params[param_idx][comp_idx]
        delta = np.random.uniform(-step_size, step_size)
        new_val = old_val + delta
        
        if comp_idx == 0:
            new_val = np.clip(new_val, -0.24, 0.24)
        elif comp_idx == 1:
            new_val = np.clip(new_val, 0.01, 0.5)
        else:
            new_val = np.clip(new_val, 0.0, 5.0)
            
        new_params[param_idx][comp_idx] = new_val
        new_h = gen_mixture(new_params)
        new_score = compute_score(new_h)
        
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            current_h = new_h
            step_size *= 0.95
            if current_score < best_score:
                best_score = current_score
                best_params = [p[:] for p in current_params]
                best_h = current_h
        else:
            step_size = min(step_size * 1.05, 0.2)
            
    return best_h, best_score

def construct():
    # Try Box
    h_box = gen_box()
    s_box = compute_score(h_box)
    
    # Try Single Gaussian
    h_gauss = gen_gaussian(0.0, 0.1, 1.0)
    s_gauss = compute_score(h_gauss)
    
    # Try Mixture
    params_init = [(-0.1, 0.1, 1.0), (0.0, 0.1, 1.0), (0.1, 0.1, 1.0)]
    h_mix, s_mix = optimize(params_init, max_iters=500)
    
    # Pick best
    candidates = [(s_box, h_box), (s_gauss, h_gauss), (s_mix, h_mix)]
    candidates.sort(key=lambda x: x[0])
    
    return candidates[0][1]

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    return construct()
