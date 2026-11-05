import numpy as np

N = 400
w = 1.0 / 200.0

def construct():
    # We want to find a sign vector s of length N (entries in {+1, -1})
    # that minimizes the L-infinity norm of the autoconvolution.
    # The autoconvolution peak is proportional to max_k |r[k]| where r is the autocorrelation.
    # Specifically, the max of |f*f| is w * max_k |r[k]|.
    # So we minimize max_k |r[k]|.
    
    # We use a simulated annealing / random walk approach on the discrete signs,
    # with occasional random restarts and a local search phase.
    
    def compute_r(s):
        """Compute autocorrelation r[k] = sum_i s[i]*s[i+k] for k=0..N-1."""
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        return r
    
    def score(s):
        r = compute_r(s)
        return np.max(np.abs(r))
    
    # Start with a random sign vector
    np.random.seed(42)
    s = np.random.choice([-1.0, 1.0], size=N)
    best_s = s.copy()
    best_score = score(s)
    
    # Local search: flip individual signs and keep if it improves
    def local_search(s, max_iters=2000):
        s = s.copy()
        current_score = score(s)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try flipping each sign
            for i in range(N):
                s[i] = -s[i]
                new_score = score(s)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    s[i] = -s[i]
        return s, current_score
    
    # Run local search from initial random start
    best_s, best_score = local_search(s, max_iters=500)
    
    # Simulated annealing with random restarts
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    for restart in range(5):
        # Random restart
        s = np.random.choice([-1.0, 1.0], size=N)
        current_score = score(s)
        
        # Simulated annealing
        for it in range(3000):
            T *= alpha
            if T < T_min:
                break
            
            # Pick a random position to flip
            i = np.random.randint(N)
            s[i] = -s[i]
            new_score = score(s)
            
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_s = s.copy()
            else:
                s[i] = -s[i]
        
        # Local search from current state
        s_ls, score_ls = local_search(s, max_iters=1000)
        if score_ls < best_score:
            best_score = score_ls
            best_s = s_ls
    
    # Final local search from best
    best_s, best_score = local_search(best_s, max_iters=5000)
    
    return best_s
