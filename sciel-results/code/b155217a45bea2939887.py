import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n = 24
    half = 12
    
    # Precompute DFT matrices for the two halves
    # We use a dense grid for fast scoring
    M = 1 << 16  # 65536 points
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    
    # W[k, j] = exp(i * k * theta[j]) for k in 0..11
    k_idx = np.arange(half)
    W = np.exp(1j * np.outer(k_idx, theta))
    
    # Precompute the phase factor for the second half: z^12 = exp(i * 12 * theta)
    phase12 = np.exp(1j * 12 * theta)

    def fast_score(c):
        """
        Compute max |P(z)| / sqrt(24) on the dense grid.
        c: list or array of 24 coefficients in {+1, -1}
        """
        c_arr = np.array(c)
        c_A = c_arr[:half]
        c_B = c_arr[half:]
        
        # V_A[j] = sum_{k=0}^{11} c_A[k] exp(i k theta_j)
        V_A = c_A @ W
        V_B = c_B @ W
        
        P = V_A + phase12 * V_B
        max_val = np.max(np.abs(P))
        return max_val / np.sqrt(n)

    def precise_score(c):
        """
        Compute sup norm with dense sampling and golden-section refinement of top peaks.
        Returns the max |P(z)| / sqrt(24).
        """
        c_arr = np.array(c)
        
        # First, find peaks on the dense grid
        c_A = c_arr[:half]
        c_B = c_arr[half:]
        V_A = c_A @ W
        V_B = c_B @ W
        P = V_A + phase12 * V_B
        abs_P = np.abs(P)
        
        # Find top 10 peaks
        top_indices = np.argsort(abs_P)[-10:][::-1]
        
        best_val = float(np.max(abs_P))
        dtheta = 2 * np.pi / M

        def f_single(t):
            # Evaluate P at a single point t
            # P(t) = sum_{k=0}^{23} c_k exp(i k t)
            # Using the split: sum_{k=0}^{11} c_k exp(i k t) + exp(i 12 t) sum_{k=0}^{11} c_{12+k} exp(i k t)
            exp_ik_t = np.exp(1j * k_idx * t)
            val_A = np.dot(c_A, exp_ik_t)
            val_B = np.dot(c_B, exp_ik_t)
            P_val = val_A + np.exp(1j * 12 * t) * val_B
            return np.abs(P_val)

        for idx in top_indices:
            center = theta[idx]
            a = center - dtheta
            b = center + dtheta
            
            # Handle wrap-around
            if a < 0:
                a += 2 * np.pi
            if b > 2 * np.pi:
                b -= 2 * np.pi
            
            a_c = max(0, a)
            b_c = min(2 * np.pi, b)
            
            if a_c < b_c:
                def obj(t):
                    return -f_single(t)
                
                res = minimize_scalar(obj, bounds=(a_c, b_c), method='bounded', options={'xatol': 1e-11})
                if res.success:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        # Update the peak location? Not strictly necessary for score, 
                        # but good for tracking. We just need the max value.
                        
        return best_val / np.sqrt(n)

    # Generate initial candidates
    candidates = []
    
    # Rudin-Shapiro
    def rudin_shapiro(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    rs = rudin_shapiro(n)
    candidates.append(rs)
    
    # Golay pairs (if known, or just use RS variants)
    # Symmetries
    for base in [rs]:
        candidates.append(base)
        candidates.append(list(reversed(base)))
        candidates.append([-c for c in base])
        candidates.append([-c for c in reversed(base)])
        
        for shift in range(n):
            rotated = base[shift:] + base[:shift]
            candidates.append(rotated)
            candidates.append(list(reversed(rotated)))
            candidates.append([-c for c in rotated])
            candidates.append([-c for c in reversed(rotated)])

    # Random candidates
    random.seed(42)
    for _ in range(200):
        candidates.append([random.choice([-1, 1]) for _ in range(n)])

    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Score all candidates with fast score
    scored = []
    for c in unique_candidates:
        s = fast_score(c)
        scored.append((s, c))
    
    scored.sort(key=lambda x: x[0])
    
    best_c = scored[0][1]
    best_fast_score = scored[0][0]
    
    # Local Search
    def local_search(c, time_limit=2.0):
        current = list(c)
        current_score = fast_score(current)
        improved = True
        start = time.time()
        
        while improved:
            improved = False
            if time.time() - start > time_limit:
                break
            
            # Single flips
            for i in range(n):
                if time.time() - start > time_limit:
                    break
                old = current[i]
                current[i] = -old
                new_score = fast_score(current)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old
            
            # Double flips
            if time.time() - start > time_limit:
                break
            for i in range(n):
                if time.time() - start > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() - start > time_limit:
                        break
                    vi, vj = current[i], current[j]
                    current[i] = -vi
                    current[j] = -vj
                    new_score = fast_score(current)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = vi
                        current[j] = vj
                        
        return current, current_score

    # Run local search on top candidates
    top_k = 20
    for i in range(min(top_k, len(scored))):
        if time.time() > deadline - 20:
            break
        c0 = scored[i][1]
        res_c, res_score = local_search(c0, time_limit=1.0)
        if res_score < best_fast_score:
            best_fast_score = res_score
            best_c = res_c

    # Random restarts
    if time.time() < deadline - 15:
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            c = [random.choice([-1, 1]) for _ in range(n)]
            res_c, res_score = local_search(c, time_limit=0.5)
            if res_score < best_fast_score:
                best_fast_score = res_score
                best_c = res_c

    # Final Polish
    # Use precise score to verify and potentially adjust
    # The fast score might be slightly off from the true sup norm due to grid resolution.
    # We should check if a neighbor has a better *precise* score, though this is expensive.
    # Instead, we just return the best_c found by fast score, but we need to ensure
    # the evaluator's precise score is minimized.
    # Since fast_score correlates well with precise_score, best_c is likely optimal.
    
    # One last check: try flipping each bit and seeing if precise score improves.
    # This is 24 precise evaluations, which is feasible.
    if time.time() < deadline - 5:
        current_best_precise = precise_score(best_c)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                old = best_c[i]
                best_c[i] = -old
                new_precise = precise_score(best_c)
                if new_precise < current_best_precise - 1e-9:
                    current_best_precise = new_precise
                    improved = True
                else:
                    best_c[i] = old

    return best_c
