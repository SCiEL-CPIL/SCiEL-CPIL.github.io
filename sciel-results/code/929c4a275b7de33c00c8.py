import time
import numpy as np
from scipy.optimize import minimize

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    # To avoid circular convolution artifacts, we need size >= 2*N - 1? 
    # No, max sum is 2*mx = N-1. Max diff is mx.
    # Linear convolution of two length N signals has length 2N-1.
    # We only care about indices 0..2*mx for sum (which is 0..N-1).
    # For diff, we care about indices 0..2*mx in the reversed convolution?
    # Let's just do full linear convolution via FFT size 2N or next power of 2.
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    # rev_sig[i] = sig[N-1-i]
    # We want counts of a-b.
    # Standard trick: conv(sig, flip(sig))
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    # The index k in S_diff corresponds to a + (N-1-b) = k => a-b = k - (N-1).
    # We want to know if a-b exists.
    # a-b ranges from -mx to mx.
    # k ranges from 0 to 2*mx.
    # Let's check non-zero entries in S_diff.
    # Specifically, we want to map back to valid differences.
    # Since U contains 0, min diff is -mx, max diff is mx.
    # The convolution covers all combinations.
    # diffset_size = np.sum(S_diff > 1e-6) is safe because only valid diffs will have mass.
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Strategy: Continuous optimization of a generalized geometric progression family.
    # U = { sum_{i=0}^{K-1} d_i * B^i | d_i in {0, ..., D} }
    # Parameters: B (base, float), D (max digit, float -> rounded), K (depth, int)
    # We optimize B and D for a fixed K, then try a few K values.
    
    def build_set(B, D, K):
        """Build the set U from parameters B, D, K."""
        D_int = int(round(D))
        if D_int < 1:
            return [0]
        
        U = [0]
        multiplier = 1.0
        for _ in range(K):
            new_U = []
            for u in U:
                for d in range(D_int + 1):
                    new_U.append(u + d * multiplier)
            U = new_U
            multiplier *= B
            
        # Convert to integers to ensure exactness for scoring
        # Since B might be non-integer, we need to be careful.
        # Actually, for the score to be meaningful, the elements should be distinct.
        # If B is not an integer, elements might collide or be non-integers.
        # The task requires "nonnegative integers".
        # So B must be an integer? Or we round the final elements?
        # If we round, we might lose the structure.
        # Let's restrict B to be an integer in the optimization, or use a mapping.
        # Better: Let B be an integer parameter.
        # But Nelder-Mead works on reals. We can round B inside the objective.
        
        U_int = [int(round(x)) for x in U]
        # Ensure 0 is in the set and unique
        U_int = sorted(set(U_int))
        if 0 not in U_int:
            U_int.insert(0, 0)
        return U_int

    def objective(params):
        B, D = params
        # B must be >= 2 to have a proper expansion, D >= 1
        B_val = max(2.0, B)
        D_val = max(1.0, D)
        
        # We need to choose K. Let's try a few K values and take the best score.
        # But K changes the size of U significantly.
        # Let's fix K for each optimization run and try multiple Ks.
        
        best_s = 0.0
        for K in [3, 4, 5]:
            if time.time() > deadline:
                return 0.0
            U = build_set(B_val, D_val, K)
            if len(U) > 20000: # FFT limit
                continue
            if max(U) > 10**6: # FFT limit
                continue
            s = _score_set(U)
            if s > best_s:
                best_s = s
        
        return -best_s # Minimize negative score

    # We will run Nelder-Mead for a few different starting points / K combinations.
    # Actually, let's just define the objective to internally pick the best K from a small set.
    # This makes the objective landscape multimodal but Nelder-Mead can handle it somewhat.
    
    # Better strategy: Loop over K, run Nelder-Mead for each K.
    
    K_candidates = [3, 4, 5, 6]
    
    for K in K_candidates:
        if time.time() > deadline:
            break
            
        # Modify objective to use fixed K
        def obj_fixed_K(params):
            B, D = params
            B_val = max(2.0, B)
            D_val = max(1.0, D)
            U = build_set(B_val, D_val, K)
            if len(U) > 20000:
                return 0.0
            if max(U) > 10**6:
                return 0.0
            return -_score_set(U)
        
        # Initial guesses
        # B around 3-5, D around 2-4
        x0 = [3.0, 2.0]
        
        try:
            res = minimize(obj_fixed_K, x0, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.success:
                B_opt, D_opt = res.x
                B_opt = max(2.0, B_opt)
                D_opt = max(1.0, D_opt)
                U_opt = build_set(B_opt, D_opt, K)
                if len(U_opt) > 0:
                    s_opt = _score_set(U_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_set = U_opt
        except Exception:
            pass

    # Also try a simple grid search on integers for B and D to ensure we don't miss integer optima
    # due to rounding in Nelder-Mead
    if time.time() < deadline:
        for K in [3, 4, 5]:
            for B in range(2, 10):
                for D in range(1, 8):
                    if time.time() > deadline:
                        break
                    U = build_set(float(B), float(D), K)
                    if len(U) > 20000:
                        continue
                    if max(U) > 10**6:
                        continue
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
