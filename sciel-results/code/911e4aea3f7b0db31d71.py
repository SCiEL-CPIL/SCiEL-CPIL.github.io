import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin for 180s limit

    # Hermite Polynomials (Probabilists' Hermite Polynomials)
    # H_0(x) = 1
    # H_4(x) = 16x^4 - 48x^2 + 12
    # H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H_12(x) = 4096x^12 - 122880x^10 + 1843200x^8 - 12902400x^6 + 44352000x^4 - 71232000x^2 + 10321600
    
    # We use a fixed grid for integration to be fast and robust.
    # The function is Gaussian-weighted, so it decays fast.
    # Range [-10, 10] is sufficient.
    x_grid = np.linspace(-10, 10, 2000)
    dx = x_grid[1] - x_grid[0]
    weight = np.exp(-x_grid**2 / 2.0)
    
    # Precompute polynomial values on the grid
    x2 = x_grid**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    
    H0_grid = np.ones_like(x_grid)
    H4_grid = 16*x4 - 48*x2 + 12
    H8_grid = 256*x8 - 3072*x6 + 11520*x4 - 13440*x2 + 1680
    H12_grid = 4096*x12 - 122880*x10 + 1843200*x8 - 12902400*x6 + 44352000*x4 - 71232000*x2 + 10321600
    
    def compute_bound(a0, a4, a8):
        # a12 is fixed by P(0)=0
        a12 = -(a0 + 12*a4 + 1680*a8) / 10321600.0
        
        # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
        P_grid = a0*H0_grid + a4*H4_grid + a8*H8_grid + a12*H12_grid
        
        # Integrand is |P(x) * exp(-x^2/2)|
        integrand = np.abs(P_grid * weight)
        # Riemann sum approximation
        integral = np.sum(integrand) * dx
        return integral

    # SPSA Optimization Function
    def run_spsa(x_init, max_iter):
        x = x_init.copy()
        best_x_local = x.copy()
        best_val_local = compute_bound(x[0], x[1], x[2])
        
        A = 0.5
        c = 0.1
        gamma = 0.6
        alpha = 0.6
        A0 = 5
        c0 = 5
        
        for k in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
                
            delta = np.random.choice([-1, 1], size=3)
            c_k = c / (k + c0)**gamma
            
            x_plus = x + c_k * delta
            x_minus = x - c_k * delta
            
            f_plus = compute_bound(x_plus[0], x_plus[1], x_plus[2])
            f_minus = compute_bound(x_minus[0], x_minus[1], x_minus[2])
            
            A_k = A / (k + A0)**alpha
            step = A_k * (f_plus - f_minus) / (2.0 * c_k)
            
            x = x - step
            
            current_val = compute_bound(x[0], x[1], x[2])
            if current_val < best_val_local:
                best_val_local = current_val
                best_x_local = x.copy()
                
        return best_x_local, best_val_local

    # Multi-start strategy
    start_points = [
        np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    
    best_overall = start_points[0].copy()
    best_val_overall = compute_bound(best_overall[0], best_overall[1], best_overall[2])
    
    # Allocate time to each start point
    # We have ~170s. Let's do 500 iterations per start point.
    # 8 start points * 500 iterations * 3 evals/iter * ~0.5ms/eval = ~6s.
    # We can do much more. Let's do 2000 iterations per start.
    iters_per_start = 2000
    
    for sp in start_points:
        if time.time() - start_time > deadline - 10:
            break
        x_res, val_res = run_spsa(sp, iters_per_start)
        if val_res < best_val_overall:
            best_val_overall = val_res
            best_overall = x_res.copy()
            
    return best_overall
