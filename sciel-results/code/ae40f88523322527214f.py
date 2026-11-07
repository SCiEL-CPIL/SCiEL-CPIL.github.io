import numpy as np
import time

N = 50

def compute_value(h):
    h = np.array(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)  # spacing of linspace(-0.5, 0.5, m+2)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = 0.0
    for i in range(m + 1):
        l2 += (w / 3.0) * (y[i]**2 + y[i] * y[i+1] + y[i+1]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s
    
    best_h = [1.0] * N
    best_val = compute_value(best_h)
    
    # Try several analytical initial shapes
    def make_gaussian(sigma):
        x = np.linspace(-1, 1, N)
        h = np.exp(-x**2 / (2 * sigma**2))
        return h
    
    def make_triangular():
        x = np.linspace(-1, 1, N)
        h = np.maximum(0, 1 - np.abs(x))
        return h
    
    def make_parabolic():
        x = np.linspace(-1, 1, N)
        h = np.maximum(0, 1 - x**2)
        return h
    
    def make_flat():
        return np.ones(N)
    
    def make_bell():
        x = np.linspace(-1, 1, N)
        h = 1.0 / (1.0 + x**4)
        return h
    
    initial_shapes = []
    for sigma in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        initial_shapes.append(make_gaussian(sigma))
    initial_shapes.append(make_triangular())
    initial_shapes.append(make_parabolic())
    initial_shapes.append(make_flat())
    initial_shapes.append(make_bell())
    
    def coordinate_descent(h, max_iters=200, step_size=0.1):
        h = h.copy()
        h = np.maximum(0, h)
        current_val = compute_value(h)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                # Try increasing
                old = h[i]
                for delta in [step_size, step_size * 2, step_size * 0.5, -step_size, -step_size * 0.5]:
                    new_val = old + delta
                    if new_val < 0:
                        continue
                    h[i] = new_val
                    val = compute_value(h)
                    if val > current_val:
                        current_val = val
                        improved = True
                        break
                if not improved:
                    # Try finer steps around current
                    for delta in [step_size * 0.1, -step_size * 0.1, step_size * 0.2, -step_size * 0.2]:
                        new_val = old + delta
                        if new_val < 0:
                            continue
                        h[i] = new_val
                        val = compute_value(h)
                        if val > current_val:
                            current_val = val
                            improved = True
                            break
                if not improved:
                    h[i] = old
                else:
                    # Reset improved flag for next i
                    improved = False
            if time.time() > deadline - 5:
                break
        return h, current_val
    
    # Optimize each initial shape
    for shape in initial_shapes:
        if time.time() > deadline - 10:
            break
        h_opt, val_opt = coordinate_descent(shape, max_iters=100, step_size=0.1)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Do a final round of optimization with smaller steps
    if time.time() < deadline - 10:
        h_final, val_final = coordinate_descent(best_h, max_iters=100, step_size=0.01)
        if val_final > best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    return [max(0.0, float(x)) for x in best_h]
