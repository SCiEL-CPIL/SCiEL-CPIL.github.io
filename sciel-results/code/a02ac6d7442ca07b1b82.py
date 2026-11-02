import time
import numpy as np
from scipy.optimize import differential_evolution

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(sorted(set(U)), dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Efficient counting via FFT
    N = 2 * mx + 1
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
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

def _build_set_from_params(params):
    """
    params: [K_0, K_1, ..., K_{L-1}, M_0, M_1, ..., M_{L-1}]
    K_i: radix (int)
    M_i: base (int)
    """
    L = len(params) // 2
    K = params[:L].astype(int)
    M = params[L:].astype(int)
    
    # Filter out invalid params
    K = K[K > 0]
    M = M[M > 0]
    L = min(len(K), len(M))
    K = K[:L]
    M = M[:L]
    
    if L == 0:
        return [0]
        
    U = np.array([0])
    for i in range(L):
        k = K[i]
        m = M[i]
        if k <= 0 or m <= 0:
            continue
        # Generate digits 0..k-1
        digits = np.arange(k) * m
        # Combine
        new_U = np.unique(U[:, None] + digits[None, :])
        U = new_U.flatten()
        
        # Early exit if set becomes too large or max is too big
        if len(U) > 100000 or U.max() > 1000000:
            break
            
    return U.tolist()

def _objective(params):
    try:
        U = _build_set_from_params(params)
        if not U:
            return 0.0
        score = _score_set(U)
        # We want to maximize score, DE minimizes
        return -score
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Define the search space
    # We search for a sequence of (K, M) pairs.
    # Let's fix L=3 or L=4 levels to keep it tractable.
    L = 3
    # Bounds for K and M
    # K: 1 to 50
    # M: 1 to 10000
    # Log scale might be better for DE? Or just linear with wide bounds.
    # Let's use linear bounds for simplicity, DE handles it.
    
    bounds = []
    for _ in range(L):
        bounds.append((1, 50))   # K
    for _ in range(L):
        bounds.append((1, 10000)) # M
        
    # Initial guess
    # Start with a known good structure: Golomb-like
    # K=2, M=1; K=2, M=3; K=2, M=7 -> Binary powers of 3?
    # Or K=3, M=2; K=3, M=6 ...
    
    x0 = np.array([2, 2, 2, 1, 3, 9])
    
    # Run Differential Evolution
    # We need to limit the number of function evaluations.
    # Each evaluation involves FFT.
    # If max U is ~1000, FFT is ~100us.
    # 10000 evals -> 1s.
    # We have 115s. We can do many more.
    
    # However, DE with large bounds and discrete-like objective can be slow.
    # Let's run DE for a few seconds, then refine.
    
    # Strategy:
    # 1. Run DE for 60s to find a good region.
    # 2. Take the best result and run a local search (hill climbing) on the discrete params.
    
    # To make DE faster, we can use a smaller population and fewer generations,
    # or use `differential_evolution` with `maxiter` and `popsize` tuned.
    
    # Let's try to run DE with a time limit.
    # scipy DE doesn't have a direct time limit, but we can wrap the objective.
    
    global _deadline
    _deadline = deadline
    
    def timed_objective(params):
        if time.time() > _deadline:
            raise Exception("Time out")
        return _objective(params)
        
    try:
        # Use a moderate population size
        # L=3, so 6 variables.
        # popsize=20, maxiter=1000?
        # We'll just let it run until time is up or convergence.
        # DE can be stopped by exception.
        
        res = differential_evolution(
            timed_objective, 
            bounds, 
            maxiter=1000, 
            popsize=20, 
            tol=1e-5,
            seed=42,
            polish=False # Polish might be slow or redundant if we do local search later
        )
        
        best_params = res.x
        best_score = -res.fun
        
    except Exception:
        # If DE didn't finish or errored, use the best found so far?
        # DE doesn't easily expose intermediate bests in all versions without hooks.
        # Let's assume it finished or we catch the exception and use x0 if no better found.
        # Actually, if it times out, we might not have a result.
        # Let's implement a manual loop or use a callback?
        # Simpler: Just run DE with a small maxiter that fits in time, then local search.
        
        # Let's estimate time per eval.
        # If we assume 100 evals/sec, 100s -> 10000 evals.
        # popsize 20, maxiter 500 -> 10000 evals.
        # So maxiter=500 is a good guess.
        
        try:
            res = differential_evolution(
                _objective, 
                bounds, 
                maxiter=500, 
                popsize=20, 
                tol=1e-5,
                seed=42,
                polish=False
            )
            best_params = res.x
            best_score = -res.fun
        except Exception:
            best_params = x0
            best_score = -_objective(x0)

    # Local Search on the discrete parameters found by DE
    # We will perturb the integer parameters K and M.
    
    current_params = best_params.copy()
    current_score = best_score
    
    # Define a local search loop
    # We have ~50s left.
    
    # Convert to int for local search
    int_params = current_params.astype(int)
    
    # Local search moves:
    # 1. Increment/Decrement K_i
    # 2. Increment/Decrement M_i
    
    while time.time() < deadline - 5:
        improved = False
        # Try all single-step moves
        for i in range(len(int_params)):
            for delta in [-1, 1]:
                new_params = int_params.copy()
                new_params[i] += delta
                if new_params[i] < 1:
                    continue
                if i < L: # K
                    if new_params[i] > 100: continue
                else: # M
                    if new_params[i] > 20000: continue
                    
                try:
                    U = _build_set_from_params(new_params.astype(float))
                    if not U:
                        continue
                    s = _score_set(U)
                    if s > current_score:
                        current_score = s
                        int_params = new_params
                        improved = True
                except Exception:
                    pass
                    
        if not improved:
            # Try larger steps or random restarts
            # Random restart in neighborhood
            new_params = int_params.copy()
            idx = np.random.randint(0, len(new_params))
            step = np.random.randint(1, 5) * np.random.choice([-1, 1])
            new_params[idx] += step
            if new_params[idx] < 1:
                new_params[idx] = 1
                
            try:
                U = _build_set_from_params(new_params.astype(float))
                if U:
                    s = _score_set(U)
                    if s > current_score:
                        current_score = s
                        int_params = new_params
            except Exception:
                pass

    # Final construction
    U = _build_set_from_params(int_params.astype(float))
    if not U:
        return [0, 1]
    return U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
