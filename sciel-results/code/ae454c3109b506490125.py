import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Fast grid for search
    M_fast = 8192
    thetas_fast = np.linspace(0, 2 * np.pi, M_fast, endpoint=False)
    Z_fast = np.exp(1j * thetas_fast[:, None] * np.arange(N)[None, :])
    
    # High precision grid for final polish
    M_high = 100000
    thetas_high = np.linspace(0, 2 * np.pi, M_high, endpoint=False)
    Z_high = np.exp(1j * thetas_high[:, None] * np.arange(N)[None, :])

    def get_score_fast(c):
        vals = Z_fast @ c
        return np.max(np.abs(vals))

    def get_score_high(c):
        vals = Z_high @ c
        return np.max(np.abs(vals))

    def local_search(c_init, score_func, max_iter=100):
        c = c_init.astype(float).copy()
        current_score = score_func(c)
        
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
                new_score = score_func(c)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    c[i] = old_val
                    
        return c, current_score

    def simulated_annealing(c_init, t_start=5.0, cooling=0.995, max_iter=20000):
        c = c_init.astype(float).copy()
        current_score = get_score_fast(c)
        best_c = c.copy()
        best_score = current_score
        
        t = t_start
        for it in range(max_iter):
            if time.time() > deadline:
                break
            i = np.random.randint(N)
            old_val = c[i]
            c[i] = -old_val
            new_score = get_score_fast(c)
            
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

    # Generate GRS (Golay-Rudin-Shapiro) polynomial
    # g[0]=1, g[1]=1, g[2n]=g[n], g[2n+1]=-g[n]
    def generate_grs(n):
        g = [0] * n
        g[0] = 1
        if n > 1: g[1] = 1
        for i in range(2, n):
            if i % 2 == 0:
                g[i] = g[i // 2]
            else:
                g[i] = -g[i // 2]
        return np.array(g, dtype=float)

    grs = generate_grs(N)
    
    # List of initializations
    inits = [
        grs,
        -grs,
        grs[::-1],
        -grs[::-1],
        np.ones(N, dtype=float),
        np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=float),
    ]
    
    # Add some random inits
    np.random.seed(42)
    for _ in range(20):
        inits.append(np.random.choice([-1, 1], N).astype(float))

    best_c = None
    best_score = np.inf

    # Phase 1: Local Search on all inits with fast scoring
    for c_init in inits:
        if time.time() > deadline:
            break
        c, score = local_search(c_init, get_score_fast, max_iter=100)
        if score < best_score:
            best_score = score
            best_c = c.copy()

    # Phase 2: Simulated Annealing from best candidate
    if best_c is not None and time.time() < deadline:
        c, score = simulated_annealing(best_c, t_start=5.0, cooling=0.995, max_iter=20000)
        if score < best_score:
            best_score = score
            best_c = c.copy()
            
    # Phase 3: Local Search again after SA
    if best_c is not None and time.time() < deadline:
        c, score = local_search(best_c, get_score_fast, max_iter=100)
        if score < best_score:
            best_score = score
            best_c = c.copy()

    # Phase 4: High Precision Polish
    # Use high-resolution grid to verify and potentially improve via single-bit flips
    if best_c is not None and time.time() < deadline:
        c = best_c.astype(float).copy()
        current_score = get_score_high(c)
        
        improved = True
        iter_count = 0
        while improved and iter_count < 10: # Limit iterations as high-res is slow
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            for i in range(N):
                old_val = c[i]
                c[i] = -old_val
                new_score = get_score_high(c)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    c[i] = old_val
        
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()

    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
    
    return [int(x) for x in best_c]
