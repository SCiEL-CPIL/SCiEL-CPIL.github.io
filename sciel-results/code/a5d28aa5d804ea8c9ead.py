import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 24
    k_vals = np.arange(N)
    
    # Grid
    num_points = 10000
    thetas = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    Z = np.exp(1j * np.outer(thetas, k_vals))
    
    def fast_eval(coeffs):
        V = Z @ coeffs
        return np.max(np.abs(V))
        
    def refine_peak(coeffs, peak_theta):
        delta = 2*np.pi / num_points
        a = peak_theta - delta
        b = peak_theta + delta
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        def f(theta):
            z = np.exp(1j * k_vals * theta)
            return np.abs(np.dot(coeffs, z))
            
        fc = f(c)
        fd = f(d)
        for _ in range(40):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
        return max(fc, fd)
        
    def slow_eval(coeffs):
        V = Z @ coeffs
        abs_V = np.abs(V)
        top_indices = np.argsort(abs_V)[-3:]
        max_val = 0
        for idx in top_indices:
            val = refine_peak(coeffs, thetas[idx])
            if val > max_val:
                max_val = val
        return max_val

    # Golay Sequence
    def generate_grs(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
        
    initial = np.array(generate_grs(N), dtype=np.int8)
    
    # SA
    def sa(initial_coeffs, max_iters=200000):
        current = initial_coeffs.copy()
        current_fast = fast_eval(current)
        best = current.copy()
        best_slow = slow_eval(best)
        best_fast = current_fast
        
        T = 1.0
        cooling_rate = 0.99995
        min_T = 1e-6
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
                
            idx = np.random.randint(N)
            new = current.copy()
            new[idx] *= -1
            
            new_fast = fast_eval(new)
            delta = new_fast - current_fast
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current = new
                current_fast = new_fast
                
                # Check if we might have a new best
                # Use a heuristic: if fast eval is better than best_fast, check slow
                if current_fast < best_fast:
                    new_slow = slow_eval(current)
                    if new_slow < best_slow:
                        best = current.copy()
                        best_slow = new_slow
                        best_fast = current_fast
                # Even if not better than best_fast, it could be a better peak?
                # Unlikely. Stick to this heuristic for speed.
                
            T *= cooling_rate
            
        return best, best_slow

    best_coeffs, best_val = sa(initial)
    
    # Random restarts
    if time.time() < deadline - 20:
        for _ in range(10):
            random_coeffs = np.random.choice([-1, 1], N).astype(np.int8)
            res_coeffs, res_val = sa(random_coeffs, max_iters=50000)
            if res_val < best_val:
                best_coeffs = res_coeffs
                best_val = res_val
            if time.time() > deadline - 5:
                break

    return list(best_coeffs)
