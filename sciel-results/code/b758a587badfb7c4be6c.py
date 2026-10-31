import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _objective_soft(f, beta=10.0):
    """
    Soft objective: LSE(|C|) / S^2.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    # Numerical stability for LSE
    lse = max_val + np.log(np.sum(np.exp(beta * (absC - max_val))))
    return (lse / beta) / (S * S)

def _gradient_soft(f, beta=10.0):
    """
    Gradient of the soft objective.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv_fft(f)
    absC = np.abs(C)
    signC = np.sign(C)
    
    max_val = np.max(absC)
    exp_term = np.exp(beta * (absC - max_val))
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    # A_k = weights_k * sign(C_k)
    A = weights * signC
    
    # Gradient of LSE w.r.t f_i is sum_k A_k * dC_k/df_i
    # dC_k/df_i = 2 * f_{k-i} if 0 <= k-i < N
    # This is equivalent to 2 * convolve(A, f) at specific lags?
    # C_k = sum_j f_j f_{k-j}
    # dC_k/df_i = f_{k-i} + f_{k-i} = 2 f_{k-i}
    # grad_i = sum_k A_k * 2 f_{k-i}
    # This is 2 * (A * f)_i where * is correlation?
    # Let's use direct convolution for clarity, N=400 is small enough for O(N^2) if needed, 
    # but we can use FFT for the gradient too or just a loop since N is small.
    # Actually, grad = 2 * np.correlate(A, f, mode='same')? No.
    # grad_i = 2 * sum_k A_k f_{k-i}
    # This is the cross-correlation of A and f.
    
    # Using FFT for gradient calculation is also fast
    # grad_LSE = 2 * np.fft.irfft(np.fft.rfft(A, n=FFT_SIZE) * np.fft.rfft(f, n=FFT_SIZE))[:N]
    # Wait, correlation via FFT:
    # corr = irfft(fft(A) * conj(fft(f)))
    # Let's stick to the loop or np.convolve if N is small, but N=400, O(N^2) is 160k ops, very fast.
    # However, we are calling this many times. FFT is safer.
    
    # A is length 2N-1, f is length N.
    # We need sum_{k} A_k f_{k-i}.
    # This is the cross-correlation.
    # np.correlate(A, f, mode='full') gives sum_k A_k f_{k-i + offset}.
    # Let's just use the loop for robustness, it's fast enough for N=400.
    grad_LSE = np.zeros(N)
    # A has length 2N-1.
    for i in range(N):
        # k ranges from 0 to 2N-2
        # f_{k-i} is non-zero if 0 <= k-i < N => i <= k < i+N
        k_start = i
        k_end = i + N
        if k_start < len(A) and k_end <= len(A):
            grad_LSE[i] = 2.0 * np.dot(A[k_start:k_end], f)
        elif k_start < len(A):
            grad_LSE[i] = 2.0 * np.dot(A[k_start:], f[:len(A)-k_start])
        # else grad is 0 (which is handled by init)
        
    LSE_val = np.max(absC) + np.log(np.sum(np.exp(beta * (absC - np.max(absC)))))
    G_val = LSE_val / beta
    
    # Objective J = G_val / S^2
    # dJ/df = (1/S^2) grad_LSE - (2 G_val / S^3) S
    return (1.0 / (S * S)) * grad_LSE - (2.0 * G_val / (S * S * S)) * S

def _local_search_binary(f, max_iter=500, time_budget=2.0):
    """
    Local search on binary {-1, 1} sequences.
    """
    f = f.copy()
    # Normalize to binary if it looks close
    f = np.sign(f)
    # Ensure sum is not 0
    S = np.sum(f)
    if abs(S) < 5:
        for _ in range(N // 2):
            idx = np.random.randint(N)
            if f[idx] * S < 0:
                f[idx] = -f[idx]
                S = np.sum(f)
        if abs(S) < 5:
            return f, 1e10
            
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        best_improvement = 0
        best_flip_idx = -1
        
        # Try a random subset of positions
        trial_positions = np.random.choice(N, size=min(20, N), replace=False)
        
        for i in trial_positions:
            v = f[i]
            # Flip f[i]
            new_f = f.copy()
            new_f[i] = -v
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
            
            new_C = _autoconv_fft(new_f)
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_flip_idx = i
        
        if best_flip_idx >= 0:
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S -= 2.0 * v
            C = _autoconv_fft(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 20:
                break
    
    return f, score

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _objective_soft(best_f, beta=100.0) # High beta for initial estimate
    
    rng = np.random.default_rng(42)
    
    # 1. Continuous Optimization with Soft-Max
    # We try different initializations and beta schedules
    
    initializations = [
        np.ones(N),
        np.sin(2 * np.pi * np.arange(N) / N),
        np.cos(2 * np.pi * np.arange(N) / N),
        rng.uniform(-1, 1, N),
        rng.choice([-1.0, 1.0], size=N)
    ]
    
    # Ensure sums are non-zero for initializations
    for i in range(len(initializations)):
        S_init = np.sum(initializations[i])
        if abs(S_init) < 1e-6:
            initializations[i] = initializations[i] + 0.1 * rng.normal(size=N)
            S_init = np.sum(initializations[i])
        # Normalize to sum=1 to help optimization? 
        # The objective is scale invariant, but gradient magnitude depends on scale.
        # Let's normalize to sum=1 for the optimization, then the score is just max|C|.
        initializations[i] = initializations[i] / S_init
        
    for f0 in initializations:
        if time.time() > DEADLINE:
            break
            
        current_f = f0.copy()
        
        # Anneal beta from low to high
        betas = [1.0, 5.0, 20.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() > DEADLINE:
                break
            
            try:
                res = minimize(
                    _objective_soft,
                    current_f,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_gradient_soft,
                    options={'maxiter': 50, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                # The optimized f might have a different sum.
                # Calculate the true score with the original formula
                S_opt = np.sum(res.x)
                if abs(S_opt) < 1e-6:
                    continue
                    
                C_opt = _autoconv_fft(res.x)
                score_opt = np.max(np.abs(C_opt)) / (S_opt * S_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_f = res.x.copy()
                
                current_f = res.x
            except Exception:
                continue

    # 2. Binary Local Search from best continuous solution
    # Project best_f to binary
    f_bin = np.sign(best_f)
    if np.sum(f_bin) == 0:
        f_bin = np.ones(N)
    
    # Run local search on binary
    f_bin_local, score_bin_local = _local_search_binary(f_bin, max_iter=1000, time_budget=10.0)
    
    if score_bin_local < best_score:
        best_score = score_bin_local
        best_f = f_bin_local.copy()

    # 3. Try some random binary restarts with local search
    for _ in range(10):
        if time.time() > DEADLINE:
            break
        f_rand = rng.choice([-1.0, 1.0], size=N)
        S_rand = np.sum(f_rand)
        if abs(S_rand) < 5:
            continue
            
        f_rand_local, score_rand_local = _local_search_binary(f_rand, max_iter=500, time_budget=5.0)
        
        if score_rand_local < best_score:
            best_score = score_rand_local
            best_f = f_rand_local.copy()

    # Return the sequence. The evaluator calculates the score.
    # The task asks for "sequence of signed step heights".
    # Returning best_f is correct.
    
    return best_f
