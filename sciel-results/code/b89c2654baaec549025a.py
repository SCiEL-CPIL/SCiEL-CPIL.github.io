"""Rewritten program for C3-prime task using FRFT-inspired phase optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Construct a signed step function (sequence of +1/-1) that minimizes
    the maximum absolute autoconvolution divided by squared integral.
    
    Approach: Use FRFT-inspired chirp initializations combined with
    efficient incremental local search and simulated annealing.
    """
    def compute_autoconv(x):
        """Compute full linear autoconvolution via FFT."""
        X = np.fft.fft(x, n=2*N)
        auto = np.fft.ifft(X * np.conj(X)).real[:N]
        return auto
    
    def autoconv_max_from_ac(ac):
        """Get max abs from autoconv array."""
        return np.max(np.abs(ac))
    
    def score_from_ac(ac, s):
        """Compute score from autoconv and sum."""
        if abs(s) < 1e-6:
            return 1e10
        return autoconv_max_from_ac(ac) / (s * s)
    
    def incremental_flip(x, ac, s, i):
        """
        Flip element i in x, update ac and s incrementally.
        Returns new x, new ac, new s.
        """
        old_val = x[i]
        new_val = -old_val
        x[i] = new_val
        s_new = s - 2 * old_val
        
        # Update autoconvolution: flipping x[i] changes ac by -2*x[i_old]*x[k] for each lag
        # ac[k] = sum_j x[j]*x[j+k]
        # After flip at i: ac_new[k] = ac[k] - 2*x_old*(x[i+k] if i+k<N else 0) - 2*x_old*(x[i-k] if i>=k else 0) + 2*x_new*x_new... 
        # Actually simpler: ac[k] = sum_j x[j]*x[j+k]
        # Flipping x[i] changes terms where j=i or j+k=i
        # Term where j=i: x[i]*x[i+k] changes from old_val*x[i+k] to new_val*x[i+k] = -old_val*x[i+k], delta = -2*old_val*x[i+k]
        # Term where j+k=i, i.e., j=i-k: x[i-k]*x[i] changes from x[i-k]*old_val to x[i-k]*new_val = -x[i-k]*old_val, delta = -2*x[i-k]*old_val
        # But if i+k >= N, that term doesn't exist in linear conv
        # If i-k < 0, that term doesn't exist
        
        ac_new = ac.copy()
        delta = -2 * old_val
        for k in range(N):
            # Contribution from j=i: valid if i+k < N
            if i + k < N:
                ac_new[k] += delta * x[i+k]
            # Contribution from j=i-k: valid if i-k >= 0, and this affects ac[k] via x[i-k]*x[i]
            if i - k >= 0:
                ac_new[k] += delta * x[i-k]
        
        return x, ac_new, s_new
    
    def local_search_fast(x, max_sweeps=50, time_limit=30):
        """Local search with incremental updates."""
        x = x.copy()
        s = np.sum(x)
        ac = compute_autoconv(x)
        current_score = score_from_ac(ac, s)
        start_time = time.time()
        
        for sweep in range(max_sweeps):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for i in range(N):
                if time.time() - start_time > time_limit:
                    break
                x_new, ac_new, s_new = incremental_flip(x, ac, s, i)
                new_score = score_from_ac(ac_new, s_new)
                if new_score < current_score * (1 - 1e-8):
                    x = x_new
                    ac = ac_new
                    s = s_new
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    x[i] = -x[i]
            if not improved:
                break
        
        return x, current_score
    
    def simulated_annealing(x, T0=1.0, T1=0.01, n_iter=5000, time_limit=20):
        """Simulated annealing to escape local minima."""
        x = x.copy()
        s = np.sum(x)
        ac = compute_autoconv(x)
        current_score = score_from_ac(ac, s)
        best_x = x.copy()
        best_score = current_score
        start_time = time.time()
        
        for it in range(n_iter):
            if time.time() - start_time > time_limit:
                break
            T = T0 * (T1 / T0) ** (it / n_iter)
            
            # Pick random position to flip
            i = np.random.randint(N)
            x_new, ac_new, s_new = incremental_flip(x, ac, s, i)
            new_score = score_from_ac(ac_new, s_new)
            
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                x = x_new
                ac = ac_new
                s = s_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                x[i] = -x[i]
        
        return best_x, best_score
    
    # Generate diverse initializations
    candidates = []
    
    # Random initializations with sufficient DC
    for trial in range(10):
        rng_trial = np.random.default_rng(1000 + trial)
        x_init = rng_trial.choice([-1.0, 1.0], size=N)
        s = np.sum(x_init)
        if abs(s) >= N * 0.05:
            candidates.append(x_init)
    
    # FRFT-inspired chirp initializations
    for alpha in [np.pi/8, np.pi/6, np.pi/5, np.pi/4, np.pi/3, np.pi/2.5, np.pi/2.2, np.pi/2]:
        t = np.arange(N)
        # Various chirp-like patterns
        phase = alpha * t * t / (2 * N)
        for offset in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
            x_chirp = np.sign(np.sin(phase + offset))
            x_chirp[x_chirp == 0] = 1.0
            s = np.sum(x_chirp)
            if abs(s) >= N * 0.05:
                candidates.append(x_chirp)
    
    # M-sequence-like patterns using LFSR
    for seed in [7, 13, 31, 63]:
        rng_lfsr = np.random.default_rng(seed)
        # Generate a pseudo-random sequence with good correlation properties
        x_lfsr = np.zeros(N)
        state = seed
        for i in range(N):
            x_lfsr[i] = 1.0 if (state & 1) else -1.0
            # Simple LFSR update
            state = (state * 3 + 1) % (1 << 8)
        s = np.sum(x_lfsr)
        if abs(s) >= N * 0.05:
            candidates.append(x_lfsr)
    
    # Evaluate all candidates and pick the best few
    scored_candidates = []
    for x_cand in candidates:
        s_cand = np.sum(x_cand)
        if abs(s_cand) < 1e-6:
            continue
        ac_cand = compute_autoconv(x_cand)
        sc = score_from_ac(ac_cand, s_cand)
        scored_candidates.append((sc, x_cand))
    
    scored_candidates.sort(key=lambda t: t[0])
    
    # Take top 3 candidates for optimization
    top_candidates = [x for _, x in scored_candidates[:3]]
    if not top_candidates:
        # Fallback: use a simple pattern
        x_fallback = np.ones(N)
        x_fallback[N//2:] = -1.0
        top_candidates = [x_fallback]
    
    # Run local search on each top candidate
    best_x = None
    best_score = np.inf
    
    for x_init in top_candidates:
        x_ls, score_ls = local_search_fast(x_init, max_sweeps=30, time_limit=15)
        if score_ls < best_score:
            best_score = score_ls
            best_x = x_ls.copy()
    
    # Run simulated annealing on the best result
    x_sa, score_sa = simulated_annealing(best_x, T0=0.5, T1=0.001, n_iter=3000, time_limit=25)
    if score_sa < best_score:
        best_score = score_sa
        best_x = x_sa.copy()
    
    # Final local search polish
    x_final, score_final = local_search_fast(best_x, max_sweeps=20, time_limit=15)
    if score_final < best_score:
        best_score = score_final
        best_x = x_final.copy()
    
    return best_x
# EVOLVE-BLOCK-END
