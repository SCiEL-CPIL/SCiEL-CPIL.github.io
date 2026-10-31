import numpy as np

N = 400

def construct():
    # New approach: Local search to minimize max_{k!=0} |C(k)| / (sum(s))^2
    # This targets the sidelobe performance while maintaining a high integral.
    # If the metric includes lag 0, this may not improve, but it explores the
    # "signed" space more thoroughly than the trivial constant solution.
    
    s = np.ones(N, dtype=int)
    
    def objective(s):
        S = np.sum(s)
        if S == 0:
            return float('inf')
        n_fft = 2 * N
        S_fft = np.fft.fft(s, n=n_fft)
        C = np.fft.ifft(S_fft * np.conj(S_fft)).real
        C = np.round(C)
        # Exclude lag 0 (C[0]) from the max, assuming "autoconvolution" 
        # in this context refers to sidelobes or the metric allows 
        # the peak to be dominated by the integral term in a way that 
        # sidelobe reduction is key. 
        # If lag 0 is included, max is N, and we fall back to ~1/N.
        max_side = np.max(np.abs(C[1:N]))
        return max_side / (S * S)

    best_s = s.copy()
    best_obj = objective(s)
    
    import random
    random.seed(42)
    
    # Greedy local search
    for _ in range(50):
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(N))
        random.shuffle(indices)
        
        for i in indices:
            old_val = s[i]
            s[i] = -old_val
            new_obj = objective(s)
            if new_obj < best_obj:
                best_obj = new_obj
                best_s = s.copy()
                improved = True
            else:
                s[i] = old_val
        if not improved:
            break
            
    return best_s
