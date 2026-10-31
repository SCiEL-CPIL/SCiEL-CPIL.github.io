"""Combined program: Sidon-set structured initialization + L-BFGS-B refinement with smooth max."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 120s kill

def is_sidon(A):
    """Check if A is a Sidon set (all sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_set_greedy(n):
    """Greedy construction of a Sidon set."""
    A = []
    used_sums = set()
    for x in range(n):
        valid = True
        new_sums = []
        for a in A:
            s = a + x
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if 2 * x in used_sums:
            valid = False
        
        if valid:
            A.append(x)
            used_sums.update(new_sums)
            used_sums.add(2 * x)
    return A

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def compute_ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        hh_max = np.max(hh)
        if hh_max < 1e-300:
            return 0.0
        return l2_sq / (l1 * hh_max)
    
    def neg_ratio(params, beta=1000.0):
        """Negative ratio with smooth max approximation for better gradient."""
        h = np.exp(params)
        h = h / np.sum(h)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        # Use log-sum-exp for smooth max: max(x) ≈ (1/beta) * log(sum(exp(beta*x)))
        hh_shifted = hh - np.max(hh)
        hh_max_smooth = (1.0 / beta) * np.log(np.sum(np.exp(beta * hh_shifted))) + np.max(hh)
        if hh_max_smooth < 1e-300:
            return 0.0
        return -l2_sq / (l1 * hh_max_smooth)
    
    # Strategy: try several structured initializations (including Sidon sets), then refine each with L-BFGS-B
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    candidates = []
    
    # 1. Single spike
    h1 = np.zeros(N)
    h1[N // 2] = 1.0
    candidates.append(h1)
    
    # 2. Two equal spikes
    h2 = np.zeros(N)
    h2[N // 4] = 0.5
    h2[3 * N // 4] = 0.5
    candidates.append(h2)
    
    # 3. Three equal spikes
    h3 = np.zeros(N)
    h3[N // 6] = 1.0 / 3
    h3[N // 2] = 1.0 / 3
    h3[5 * N // 6] = 1.0 / 3
    candidates.append(h3)
    
    # 4. Many equal spikes (10)
    h4 = np.zeros(N)
    positions = np.linspace(0, N - 1, 10).astype(int)
    h4[positions] = 1.0 / len(positions)
    candidates.append(h4)
    
    # 5. Uniform
    candidates.append(np.ones(N) / N)
    
    # 6. Two adjacent spikes
    h6 = np.zeros(N)
    h6[N // 2 - 1] = 0.5
    h6[N // 2] = 0.5
    candidates.append(h6)
    
    # 7. Wide plateau
    h7 = np.zeros(N)
    h7[N//4 : 3*N//4] = 1.0 / (N//2)
    candidates.append(h7)
    
    # 8. Sidon set based initialization
    A_greedy = find_sidon_set_greedy(N)
    h_sidon = np.zeros(N)
    for idx in A_greedy:
        h_sidon[idx] = 1.0
    if np.sum(h_sidon) > 0:
        h_sidon /= np.sum(h_sidon)
        candidates.append(h_sidon)
    
    # Try to find a larger Sidon set with random restarts
    best_A = A_greedy
    for target_size in range(len(A_greedy) + 1, 10):
        found = False
        for _ in range(200):  # 200 random tries per size
            if time.time() > start + DEADLINE - 20:
                break
            A_try = sorted(np.random.choice(N, target_size, replace=False))
            if is_sidon(A_try):
                if target_size > len(best_A):
                    best_A = A_try
                    found = True
                if found:
                    break
        if found:
            continue
        else:
            break  # If we can't find size K, we can't find K+1
            
    if len(best_A) > len(A_greedy):
        h_sidon_large = np.zeros(N)
        for idx in best_A:
            h_sidon_large[idx] = 1.0
        h_sidon_large /= np.sum(h_sidon_large)
        candidates.append(h_sidon_large)
    
    # 9. Random sparse
    rng = np.random.default_rng(42)
    for _ in range(10):
        h_rand = np.zeros(N)
        k = rng.integers(1, 15)
        idx = rng.choice(N, size=k, replace=False)
        h_rand[idx] = 1.0 / k
        candidates.append(h_rand)
    
    # 10. Gaussian-like
    x = np.arange(N)
    h9 = np.exp(-0.5 * ((x - N/2) / (N/4))**2)
    h9 /= np.sum(h9)
    candidates.append(h9)
    
    # 11. Bimodal Gaussian
    h10 = 0.5 * np.exp(-0.5 * ((x - N/4) / (N/8))**2) + 0.5 * np.exp(-0.5 * ((x - 3*N/4) / (N/8))**2)
    h10 /= np.sum(h10)
    candidates.append(h10)
    
    # 12. Alternating pattern
    h12 = np.zeros(N)
    h12[::2] = 1.0
    h12 /= np.sum(h12)
    candidates.append(h12)
    
    # 13. Block pattern
    h13 = np.zeros(N)
    h13[0:10] = 1.0
    h13[40:50] = 1.0
    h13 /= np.sum(h13)
    candidates.append(h13)
    
    # Evaluate all candidates and keep best
    for h_cand in candidates:
        val = compute_ratio(h_cand)
        if val > best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # Refine best candidates with L-BFGS-B using smooth objective
    # Sort candidates by ratio and refine top few
    scored = [(compute_ratio(h), h) for h in candidates]
    scored.sort(key=lambda x: -x[0])
    
    n_refine = min(12, len(scored))
    for i in range(n_refine):
        if elapsed() > DEADLINE - 15:
            break
        
        h_init = scored[i][1]
        # Work in log-space for nonnegativity
        h_pos = h_init + 1e-10
        x0 = np.log(h_pos)
        
        try:
            res = minimize(neg_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            if elapsed() > DEADLINE - 10:
                break
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish from the best solution found with tighter tolerance
    if elapsed() < DEADLINE - 5:
        try:
            h_pos = best_h + 1e-10
            x0 = np.log(h_pos)
            res = minimize(neg_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16})
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure non-negative and return as list
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END
