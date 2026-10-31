import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    C = np.fft.irfft(F * F, n=fft_size)
    return C[:size]

def _score(f):
    """Compute max(|autoconv(f)|) / (sum(f))^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _simulated_annealing(f, time_budget=10.0, T0=1.0, cooling=0.999, rng=None):
    """
    Simulated annealing on binary {-1, 1} sequences.
    Uses direct flip evaluation with delta autoconvolution.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    f = f.copy()
    S = np.sum(f)
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    best_f = f.copy()
    best_score = score
    
    T = T0
    start_time = time.time()
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
        
        # Select candidate flip positions
        k_max = np.argmax(np.abs(C))
        
        # Build candidate set: positions affecting peak + random
        candidates = set()
        
        # Positions that affect the peak
        for j in range(N):
            if 0 <= k_max - j < N:
                candidates.add(j)
        
        # Add random positions
        for _ in range(30):
            candidates.add(rng.integers(0, N))
        
        candidates = list(candidates)
        rng.shuffle(candidates)
        
        improved = False
        for i in candidates:
            if time.time() > DEADLINE:
                break
            
            v = f[i]
            
            # Compute delta_C for flipping position i
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            k_end_clamped = min(k_end, len(C))
            f_slice = f[:k_end_clamped - k_start]
            
            delta_C = np.zeros_like(C)
            delta_C[k_start:k_end_clamped] = -2.0 * v * f_slice
            
            new_C = C + delta_C
            new_S = S - 2.0 * v
            
            if abs(new_S) < 1e-6:
                continue
            
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            delta_score = new_score - score
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta_score < 0 or (T > 1e-10 and rng.random() < np.exp(-delta_score / T)):
                f[i] = -v
                S = new_S
                C = new_C
                score = new_score
                improved = True
                
                if score < best_score:
                    best_score = score
                    best_f = f.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        # Cool down
        T *= cooling
        
        # If stuck too long, restart temperature
        if no_improve > 100:
            T = T0 * 0.5
            no_improve = 0
    
    return best_f, best_score

def _greedy_init(rng):
    """Greedy initialization: start with random, then greedily improve."""
    f = rng.choice([-1.0, 1.0], size=N)
    S = np.sum(f)
    if abs(S) < 10:
        # Flip some to ensure nonzero sum
        for _ in range(N // 4):
            idx = rng.integers(0, N)
            if f[idx] * S > 0:
                f[idx] = -f[idx]
                S = np.sum(f)
    
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    # Greedy improvement
    for _ in range(50):
        best_improvement = 0
        best_idx = -1
        k_max = np.argmax(np.abs(C))
        
        for j in range(N):
            if 0 <= k_max - j < N:
                v = f[j]
                k_start = j
                k_end = min(j + N, 2 * N - 1)
                k_end_clamped = min(k_end, len(C))
                f_slice = f[:k_end_clamped - k_start]
                
                delta_C = np.zeros_like(C)
                delta_C[k_start:k_end_clamped] = -2.0 * v * f_slice
                new_C = C + delta_C
                new_S = S - 2.0 * v
                
                if abs(new_S) < 1e-6:
                    continue
                
                new_score = np.max(np.abs(new_C)) / (new_S * new_S)
                improvement = score - new_score
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_idx = j
        
        if best_idx >= 0:
            v = f[best_idx]
            f[best_idx] = -v
            S -= 2.0 * v
            C = _autoconv_fft(f)
            score = np.max(np.abs(C)) / (S * S)
        else:
            break
    
    return f, score

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # Multiple diverse initializations
    initializations = []
    
    # 1. All ones
    initializations.append(np.ones(N))
    
    # 2. Random binary
    f_rand = rng.choice([-1.0, 1.0], size=N)
    S_rand = np.sum(f_rand)
    if abs(S_rand) < 10:
        for _ in range(N // 4):
            idx = rng.integers(0, N)
            if f_rand[idx] * S_rand > 0:
                f_rand[idx] = -f_rand[idx]
    initializations.append(f_rand)
    
    # 3. Block patterns
    for block_size in [2, 4, 8, 16]:
        f_block = np.zeros(N)
        for i in range(N):
            f_block[i] = 1.0 if (i // block_size) % 2 == 0 else -1.0
        S_block = np.sum(f_block)
        if abs(S_block) < 10:
            f_block[0] = 1.0
        initializations.append(f_block)
    
    # 4. Sinusoidal patterns
    for freq in [1, 2, 4, 8]:
        f_sin = np.sign(np.sin(2 * np.pi * freq * np.arange(N) / N))
        f_sin[f_sin == 0] = 1.0
        S_sin = np.sum(f_sin)
        if abs(S_sin) < 10:
            f_sin[0] = 1.0
        initializations.append(f_sin)
    
    # 5. Random with structure (random blocks)
    for _ in range(3):
        f_struct = np.ones(N)
        pos = 0
        while pos < N:
            block_len = rng.integers(2, 20)
            val = rng.choice([-1.0, 1.0])
            end = min(pos + block_len, N)
            f_struct[pos:end] = val
            pos = end
        S_struct = np.sum(f_struct)
        if abs(S_struct) < 10:
            f_struct[0] = 1.0
        initializations.append(f_struct)
    
    # Evaluate all initializations
    for f_init in initializations:
        if time.time() > DEADLINE:
            break
        s = _score(f_init)
        if s < best_score:
            best_score = s
            best_f = f_init.copy()
    
    # Run simulated annealing from multiple starts
    num_sa_runs = 10
    for r in range(num_sa_runs):
        if time.time() > DEADLINE:
            break
        
        # Choose starting point
        if r < len(initializations):
            f_start = initializations[r].copy()
        else:
            f_start = best_f.copy()
        
        # Adaptive time budget
        remaining = DEADLINE - time.time()
        time_budget = min(remaining * 0.8, 12.0)
        if time_budget < 1.0:
            break
        
        # Run simulated annealing
        f_sa, score_sa = _simulated_annealing(f_start, time_budget=time_budget, rng=rng)
        
        if score_sa < best_score:
            best_score = score_sa
            best_f = f_sa.copy()
        
        # Check deadline
        if time.time() > DEADLINE - 5.0:
            break
    
    # Final polish: run SA from the best solution with more time
    if time.time() < DEADLINE - 3.0:
        remaining = DEADLINE - time.time() - 3.0
        if remaining > 2.0:
            f_polish, score_polish = _simulated_annealing(best_f, time_budget=remaining, T0=0.5, cooling=0.9995, rng=rng)
            if score_polish < best_score:
                best_score = score_polish
                best_f = f_polish.copy()
    
    return best_f
