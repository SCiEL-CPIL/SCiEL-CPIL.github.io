import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for safety
    
    # Precompute the evaluation matrix
    M = 20000
    thetas = np.linspace(0, 2*np.pi, M, endpoint=False)
    Z = np.exp(1j * thetas[:, None] * np.arange(N)[None, :])
    
    best_c = None
    best_score = np.inf
    
    def get_score(c):
        vals = Z @ c
        return np.max(np.abs(vals))
    
    def local_search(c_init, max_iter=100):
        """
        Performs local search starting from c_init.
        Returns (best_c, best_score).
        """
        c = c_init.astype(float).copy()
        current_score = get_score(c)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            for i in range(N):
                old_val = c[i]
                c[i] = -old_val
                new_score = get_score(c)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    c[i] = old_val
                    
        return c, current_score
    
    def simulated_annealing(c_init, t_start=10.0, t_end=0.01, cooling=0.99, max_iter=5000):
        """
        Simulated annealing to escape local minima.
        """
        c = c_init.astype(float).copy()
        current_score = get_score(c)
        best_c = c.copy()
        best_score = current_score
        
        t = t_start
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Pick a random bit to flip
            i = np.random.randint(N)
            old_val = c[i]
            c[i] = -old_val
            new_score = get_score(c)
            
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / t):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_c = c.copy()
            else:
                c[i] = old_val
            
            t *= cooling
        
        return best_c, best_score
    
    # List of initializations to try
    inits = []
    
    # 1. GRS polynomial (truncated)
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    inits.append(np.array(p[:N], dtype=float))
    
    # 2. Alternating signs
    inits.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=float))
    
    # 3. All ones
    inits.append(np.ones(N, dtype=float))
    
    # 4. Some structured patterns
    # Pattern: 1,1,1,-1,1,1,1,-1,...
    pattern1 = []
    for i in range(N):
        if (i // 3) % 2 == 0:
            pattern1.append(1)
        else:
            pattern1.append(-1)
    inits.append(np.array(pattern1, dtype=float))
    
    # Pattern: 1,-1,1,-1,1,1,1,-1,1,-1,1,-1,...
    pattern2 = []
    for i in range(N):
        if i % 6 < 3:
            pattern2.append(1 if i % 2 == 0 else -1)
        else:
            pattern2.append(1 if i % 2 == 0 else -1)
    inits.append(np.array(pattern2, dtype=float))
    
    # 5. Random restarts
    np.random.seed(42)
    num_restarts = 200
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        inits.append(np.random.choice([-1, 1], N).astype(float))
    
    # First pass: local search on all inits
    for c_init in inits:
        if time.time() > deadline:
            break
        c, score = local_search(c_init, max_iter=50)
        if score < best_score:
            best_score = score
            best_c = c.copy()
    
    # Second pass: simulated annealing on best candidates
    if best_c is not None and time.time() < deadline:
        c, score = simulated_annealing(best_c, t_start=5.0, t_end=0.01, cooling=0.995, max_iter=10000)
        if score < best_score:
            best_score = score
            best_c = c.copy()
    
    # Third pass: more local search from best
    if best_c is not None and time.time() < deadline:
        c, score = local_search(best_c, max_iter=200)
        if score < best_score:
            best_score = score
            best_c = c.copy()
    
    # Fourth pass: try more random restarts with SA if time permits
    if time.time() < deadline - 30:
        for _ in range(50):
            if time.time() > deadline:
                break
            c_init = np.random.choice([-1, 1], N).astype(float)
            c, score = simulated_annealing(c_init, t_start=3.0, t_end=0.01, cooling=0.995, max_iter=5000)
            if score < best_score:
                best_score = score
                best_c = c.copy()
    
    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
    
    return [int(x) for x in best_c]
