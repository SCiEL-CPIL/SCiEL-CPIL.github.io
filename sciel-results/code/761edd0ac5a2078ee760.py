import numpy as np

def construct():
    n, m, p = 2, 4, 5
    deadline = _start_time + TIME_LIMIT - 5
    
    # Fallback: rank 36
    def make_rank36():
        # ... (as in the current program)
        pass
    
    fallback = make_rank36()
    
    # Try to find lower rank
    np.random.seed(42)
    
    for r in range(20, 36):
        if time.time() > deadline - 10:
            break
        
        max_tries = 20
        for attempt in range(max_tries):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with a specific structure
            # Assume U has a block structure: U[i*m+j, r] = a[i, j, r]
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            
            # Check
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r_idx in range(r):
                            val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        if abs(val - 1) > 1e-6:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
            
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return fallback
