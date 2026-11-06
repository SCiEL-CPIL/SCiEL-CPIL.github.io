import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 24
    M = 4096 # Higher resolution for better surrogate
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    
    k_vals = np.arange(N)
    W = np.exp(1j * np.outer(k_vals, thetas))
    
    def calc_cost(c_vec):
        P = c_vec @ W
        return np.max(np.abs(P))
        
    best_c = None
    best_cost = np.inf
    
    # 1. Simulated Annealing
    T = 1.0
    cooling_rate = 0.99
    min_T = 1e-5
    
    # Run SA for a fixed number of steps or until time limit
    c = np.random.choice([-1, 1], N).astype(np.float64)
    current_cost = calc_cost(c)
    best_c = c.copy()
    best_cost = current_cost
    
    steps = 0
    while T > min_T and time.time() < deadline - 20: # Leave 20s for refinement/restarts
        idx = np.random.randint(N)
        c[idx] = -c[idx]
        new_cost = calc_cost(c)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_c = c.copy()
        else:
            c[idx] = -c[idx]
            
        T *= cooling_rate
        steps += 1
        
    # 2. Local Search Restarts
    # Use the remaining time for multiple restarts with steepest descent
    while time.time() < deadline - 10:
        c = np.random.choice([-1, 1], N).astype(np.float64)
        current_cost = calc_cost(c)
        
        improved = True
        while improved:
            improved = False
            for k in range(N):
                c[k] = -c[k]
                new_cost = calc_cost(c)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    c[k] = -c[k]
                    
        if current_cost < best_cost:
            best_cost = current_cost
            best_c = c.copy()
            
    # 3. Refinement
    if best_c is None:
        # Fallback to GRS if something went wrong
        p, q = [1], [1]
        while len(p) < 24:
            p, q = p + q, p + [-c for c in q]
        return p[:24]
        
    P_best = best_c @ W
    abs_P_best = np.abs(P_best)
    
    # Define high-precision eval function
    def f_high(theta):
        z = np.exp(1j * theta)
        powers = z ** np.arange(N)
        return np.abs(np.dot(best_c, powers))
        
    gr = (np.sqrt(5) - 1) / 2
    max_norm_final = 0
    
    # Check top 10 peaks
    top_indices = np.argsort(abs_P_best)[-10:][::-1]
    window = 2 * np.pi / M
    
    for idx in top_indices:
        a = thetas[idx] - window
        b = thetas[idx] + window
        
        # Ensure a < b and handle wrap-around if necessary?
        # thetas are in [0, 2pi). window is small.
        # If idx=0, a is negative. f_high handles any theta.
        
        c_gs = b - gr * (b - a)
        d_gs = a + gr * (b - a)
        fc = f_high(c_gs)
        fd = f_high(d_gs)
        
        for _ in range(60):
            if fc < fd:
                b = d_gs
                d_gs = c_gs
                fd = fc
                c_gs = b - gr * (b - a)
                fc = f_high(c_gs)
            else:
                a = c_gs
                c_gs = d_gs
                fc = fd
                d_gs = a + gr * (b - a)
                fd = f_high(d_gs)
                
        theta_opt = (a + b) / 2
        val = f_high(theta_opt)
        if val > max_norm_final:
            max_norm_final = val
            
    # Convert to list of ints
    return best_c.astype(int).tolist()
