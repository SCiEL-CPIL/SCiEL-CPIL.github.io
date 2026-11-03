import numpy as np
import time

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    # Continuous autoconvolution of step function f(t) = sum h_i 1_{[x_i, x_{i+1})}(t)
    # The peak is proportional to the max of the discrete convolution of h with itself.
    # Specifically, max(f*f) = DELTA * max(conv(h,h))
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Objective to maximize: S(h) = sum(h^2) / (sum(h) * max(conv(h,h)))
    # We normalize h to sum to 1, so S(h) = sum(h^2) / max(conv(h,h)) * (1/DELTA) ? 
    # Actually _score_raw includes DELTA in denominator.
    # Let's just use _score_raw directly.
    
    def get_score(h):
        return _score_raw(h)
    
    def normalize(h):
        s = np.sum(h)
        if s > 0:
            return h / s
        return h
    
    def generate_inits():
        inits = []
        x_idx = np.arange(N)
        
        # 1. Uniform
        inits.append(np.ones(N) / N)
        
        # 2. Gaussians with varying centers and widths
        centers = [N//2, N//4, 3*N//4, 10, 40]
        widths = [2, 5, 10, 15, 25]
        for c in centers:
            for w in widths:
                h = np.exp(-0.5 * ((x_idx - c) / w)**2)
                if np.sum(h) > 0:
                    inits.append(normalize(h))
        
        # 3. Triangles
        for c in centers:
            for w in [5, 10, 20]:
                h = np.maximum(0, 1 - np.abs(x_idx - c) / w)
                if np.sum(h) > 0:
                    inits.append(normalize(h))
        
        # 4. Sparse peaks
        for k in [1, 2, 5, 10]:
            for pos in [N//2, N//4, 3*N//4]:
                h = np.zeros(N)
                start = max(0, pos - k//2)
                end = min(N, pos + k//2 + 1)
                h[start:end] = 1.0
                if np.sum(h) > 0:
                    inits.append(normalize(h))
                    
        # 5. Alternating / Oscillating
        for freq in [1, 2, 4]:
            h = 0.5 + 0.5 * np.cos(2 * np.pi * freq * x_idx / N)
            h = np.maximum(h, 0)
            if np.sum(h) > 0:
                inits.append(normalize(h))
                
        # 6. Random smooth (deterministic seed for reproducibility if needed, but np.random is fine)
        np.random.seed(42)
        for _ in range(5):
            h = np.random.rand(N)
            # Smooth it
            kernel = np.ones(5)/5
            h = np.convolve(h, kernel, mode='same')
            h = np.maximum(h, 0)
            if np.sum(h) > 0:
                inits.append(normalize(h))
                
        return inits

    def simulated_annealing(h_init, max_iters=5000, T_start=1e-2, T_end=1e-8, alpha=0.99):
        h = h_init.copy()
        h = normalize(h)
        current_score = get_score(h)
        best_h = h.copy()
        best_score = current_score
        
        T = T_start
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Generate a neighbor
            # Strategy: Randomly pick a type of move
            move_type = np.random.randint(0, 3)
            
            if move_type == 0:
                # Shift mass from j to i
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j:
                    continue
                # Move a fraction of mass
                # Try to move a significant chunk to explore better
                amount = h[j] * np.random.uniform(0.1, 0.5)
                h_new = h.copy()
                h_new[i] += amount
                h_new[j] -= amount
                if h_new[j] < 0:
                    continue
                h_new = np.maximum(h_new, 0)
                h_new = normalize(h_new)
                
            elif move_type == 1:
                # Local Spread: Take mass from a random index and spread to neighbors
                j = np.random.randint(N)
                if h[j] == 0:
                    continue
                # Spread to j-1 and j+1
                neighbors = []
                if j > 0: neighbors.append(j-1)
                if j < N-1: neighbors.append(j+1)
                if not neighbors:
                    continue
                    
                amount = h[j] * np.random.uniform(0.2, 0.8)
                h_new = h.copy()
                h_new[j] -= amount
                # Distribute amount to neighbors
                if len(neighbors) == 1:
                    h_new[neighbors[0]] += amount
                else:
                    h_new[neighbors[0]] += amount / 2
                    h_new[neighbors[1]] += amount / 2
                h_new = np.maximum(h_new, 0)
                h_new = normalize(h_new)
                
            else:
                # Concentration: Pull mass from neighbors into a random index
                i = np.random.randint(N)
                neighbors = []
                if i > 0: neighbors.append(i-1)
                if i < N-1: neighbors.append(i+1)
                if not neighbors:
                    continue
                    
                # Pull from all neighbors
                h_new = h.copy()
                total_pulled = 0
                for n_idx in neighbors:
                    if h_new[n_idx] > 0:
                        pull_amt = h_new[n_idx] * np.random.uniform(0.1, 0.5)
                        h_new[n_idx] -= pull_amt
                        total_pulled += pull_amt
                if total_pulled > 0:
                    h_new[i] += total_pulled
                    h_new = np.maximum(h_new, 0)
                    h_new = normalize(h_new)
                else:
                    continue
            
            new_score = get_score(h_new)
            
            # Acceptance criterion
            delta_score = new_score - current_score
            if delta_score > 0:
                # Always accept improvement
                h = h_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Accept with probability exp(delta / T)
                # Note: scores are positive. delta is negative.
                prob = np.exp(delta_score / T)
                if np.random.rand() < prob:
                    h = h_new
                    current_score = new_score
            
            # Update Temperature
            T *= alpha
            if T < T_end:
                T = T_end
                
        return best_h, best_score

    # Main Loop
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    inits = generate_inits()
    
    # Run SA from each init
    # We have ~30 inits. We can afford to run SA for each if we limit iterations.
    # Or just run a few times.
    # Let's run SA from the top 10 initial scores to save time, or all if fast enough.
    # SA is O(N^2) per step (convolution). 5000 steps * 2500 ops = 12.5M ops per run.
    # 30 runs = 375M ops. Should be well within 120s in NumPy.
    
    # Sort inits by score to prioritize good starts?
    # Actually, SA is good at escaping local minima, so bad starts can sometimes lead to better global mins.
    # But starting from a decent place helps.
    
    scored_inits = [(h, get_score(h)) for h in inits]
    scored_inits.sort(key=lambda x: x[1], reverse=True)
    
    # Use all inits if time permits, otherwise top 10
    # Let's try all, but break if time is up.
    
    for h_init, _ in scored_inits:
        if time.time() > deadline - 5:
            break
            
        h_opt, s_opt = simulated_annealing(h_init, max_iters=2000)
        
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Final local refinement with a faster method on the best solution found
    # Use a simple coordinate descent polish
    if time.time() < deadline - 2:
        h = best_h.copy()
        current_score = get_score(h)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try moving mass from j to i for all pairs? Too slow.
            # Just try random pairs a few times.
            for _ in range(500):
                if time.time() > deadline - 2:
                    break
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j: continue
                
                # Binary search for best eps
                lo, hi = 0.0, h[j]
                best_eps = 0
                best_local_score = current_score
                for _ in range(10):
                    mid = (lo + hi) / 2
                    h_test = h.copy()
                    h_test[i] += mid
                    h_test[j] -= mid
                    h_test = np.maximum(h_test, 0)
                    h_test = normalize(h_test)
                    s_test = get_score(h_test)
                    if s_test > best_local_score:
                        best_local_score = s_test
                        best_eps = mid
                        lo = mid
                    else:
                        hi = mid
                
                if best_eps > 0:
                    h[i] += best_eps
                    h[j] -= best_eps
                    h = np.maximum(h, 0)
                    h = normalize(h)
                    current_score = best_local_score
                    if current_score > best_score:
                        best_score = current_score
                        best_h = h.copy()
                    improved = True
    
    return list(best_h)
