import time
import numpy as np

N = 400
_T0 = time.time()
_DEADLINE = _T0 + 120.0


def _c3_prime_val(h):
    """
    Compute max_k |autocorr_k| / (sum h)^2 for signed step function h of length N.
    """
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if abs(total) < 1e-12:
        return np.inf
    denom = total * total
    ac = np.correlate(h, h, mode='full')
    return np.max(np.abs(ac)) / denom


def _m_sequence(n, m):
    """Generate an M-Sequence (maximal-length LFSR) of length n as ±1."""
    if n <= 0:
        return np.ones(0)
    # Find a primitive polynomial of degree d such that 2^d - 1 >= n
    # We'll use known primitive polynomials for various degrees
    # For simplicity, we'll use a general LFSR approach
    # M-sequences have period 2^d - 1 where d is the LFSR length
    
    # Try different LFSR lengths
    for d in range(8, 20):
        period = (1 << d) - 1
        if period >= n:
            break
    else:
        d = 10
        period = (1 << d) - 1
    
    # Use a simple LFSR with a known primitive polynomial
    # For degree 8: x^8 + x^4 + x^3 + x^2 + 1 (0x1D7)
    # For degree 10: x^10 + x^7 + x^5 + x^3 + 1 (0x37D)
    # For degree 12: x^12 + x^6 + x^4 + x + 1 (0x10C1)
    # For degree 16: x^16 + x^5 + x^3 + x^2 + 1 (0x1006D)
    
    # Simple approach: use a linear recurrence
    # We'll generate the sequence using a simple LFSR
    # The specific polynomial matters for maximality, but for our purposes
    # any good pseudo-random sequence will work
    
    # Use numpy random with a specific seed based on m for reproducibility
    # But actually, let's use proper M-sequences for better spectral properties
    
    # For practical purposes, let's use a mix of approaches
    if m == 0:
        # Pure M-sequence style
        rng = np.random.RandomState(42 + m)
        # Generate using LFSR-like approach
        state = rng.randint(1, (1 << min(d, 16)))
        seq = []
        for i in range(period):
            seq.append(1.0 if (state & 1) == 0 else -1.0)
            # Simple LFSR step (not truly maximal length, but good enough)
            new_bit = (state >> 1) & 1
            # XOR with appropriate bits based on polynomial
            if d <= 8:
                poly_mask = 0x1D7  # x^8+x^4+x^3+x^2+1
            elif d <= 10:
                poly_mask = 0x37D  # x^10+x^7+x^5+x^3+1
            else:
                poly_mask = 0x1006D  # x^16+x^5+x^3+x^2+1
            
            # Simplified: just shift and add a feedback bit
            feedback = ((state >> (d-1)) ^ (state >> 4) ^ (state >> 3) ^ (state >> 2) ^ 1) & 1
            state = ((state >> 1) | (feedback << (d-1))) & ((1 << d) - 1)
        
        return np.array(seq[:n])
    else:
        # Other variations
        rng = np.random.RandomState(100 + m * 17)
        # Generate a ±1 sequence
        h = rng.choice([-1.0, 1.0], size=n)
        # Ensure near-zero mean
        total = h.sum()
        if abs(total) > n * 0.05:
            # Flip some to reduce mean
            n_flip = int(abs(total) / 2)
            if total > 0:
                idx = np.where(h == 1.0)[0]
                flip_idx = rng.choice(idx, size=min(n_flip, len(idx)), replace=False)
                h[flip_idx] = -1.0
            else:
                idx = np.where(h == -1.0)[0]
                flip_idx = rng.choice(idx, size=min(n_flip, len(idx)), replace=False)
                h[flip_idx] = 1.0
        return h


def _balanced_sequence(n, seed):
    """Generate a balanced ±1 sequence (equal or near-equal +1 and -1)."""
    rng = np.random.RandomState(seed)
    n_plus = n // 2
    n_minus = n - n_plus
    h = np.concatenate([np.ones(n_plus), -np.ones(n_minus)])
    rng.shuffle(h)
    return h


def _sinusoidal_balance(n, freq, phase=0.0):
    """Generate sign(sin) sequence at given frequency."""
    t = np.arange(n)
    return np.sign(np.sin(2 * np.pi * freq * t / n + phase))


def _chirp_sequence(n, f0, f1, phase0=0.0):
    """Generate a chirp (frequency sweep) sequence."""
    t = np.arange(n)
    freqs = f0 + (f1 - f0) * t / n
    phase = phase0 + 2 * np.pi * np.cumsum(freqs) / n
    return np.sign(np.sin(phase))


def _local_search_flip(h, max_sweeps=100, time_budget=30.0):
    """
    Local search: single flip operations with FFT-accelerated evaluation.
    """
    h = h.copy().astype(float)
    total = h.sum()
    if abs(total) < 1e-12:
        total = 1.0  # Avoid division by zero
    
    # Use FFT-based autocorrelation for speed
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
    # For full autocorrelation, we need both positive and negative lags
    # np.correlate(h, h, 'full') gives 2N-1 values
    # But for the max absolute value, we can use the circular autocorrelation
    # which is faster with FFT
    ac_full = np.fft.irfft(H * np.conj(H), n=n_fft)
    # The circular autocorrelation has the same max absolute value as linear
    # for our purposes (approximately)
    current_val = np.max(np.abs(ac_full)) / (total * total)
    
    start_time = time.time()
    improved = True
    sweeps = 0
    
    while improved and sweeps < max_sweeps and time.time() - start_time < time_budget:
        improved = False
        sweeps += 1
        
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            # Try flipping h[i]
            h[i] *= -1
            new_total = total + (-2.0 * h[i] if h[i] == 1 else 2.0 * h[i])
            # Wait, this is getting complicated. Let me redo.
            h[i] *= -1  # Undo
            trial = h.copy()
            trial[i] *= -1
            new_total = total + (-2.0 * trial[i] + 2.0 * h[i])
            # Actually: trial[i] = -h[i], so new_total = total - 2*h[i]
            new_total = total - 2.0 * h[i]
            
            if abs(new_total) < 1e-12:
                h[i] *= -1  # Undo
                continue
            
            # Compute autocorrelation for trial using FFT
            H_trial = np.fft.rfft(trial, n=n_fft)
            ac_trial = np.fft.irfft(H_trial * np.conj(H_trial), n=n_fft)
            val_trial = np.max(np.abs(ac_trial)) / (new_total * new_total)
            
            if val_trial < current_val - 1e-12:
                h = trial
                total = new_total
                current_val = val_trial
                improved = True
                H = np.fft.rfft(h, n=n_fft)
                ac_full = np.fft.irfft(H * np.conj(H), n=n_fft)
    
    return h, current_val


def _local_search_segment_flip(h, seg_lengths=[2, 3, 4, 5], max_sweeps=50, time_budget=20.0):
    """Local search: flip small contiguous segments."""
    h = h.copy().astype(float)
    total = h.sum()
    if abs(total) < 1e-12:
        total = 1.0
    
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac_full = np.fft.irfft(H * np.conj(H), n=n_fft)
    current_val = np.max(np.abs(ac_full)) / (total * total)
    
    start_time = time.time()
    improved = True
    sweeps = 0
    
    while improved and sweeps < max_sweeps and time.time() - start_time < time_budget:
        improved = False
        sweeps += 1
        
        for seg_len in seg_lengths:
            for start in range(0, N - seg_len + 1, max(1, seg_len // 2)):
                if time.time() - start_time > time_budget:
                    break
                
                trial = h.copy()
                trial[start:start+seg_len] *= -1
                new_total = total - 2.0 * trial[start:start+seg_len].sum() + 2.0 * h[start:start+seg_len].sum()
                # Actually: trial[start:start+seg_len] = -h[start:start+seg_len]
                # So new_total = total - 2 * sum(h[start:start+seg_len])
                new_total = total - 2.0 * h[start:start+seg_len].sum()
                
                if abs(new_total) < 1e-12:
                    continue
                
                H_trial = np.fft.rfft(trial, n=n_fft)
                ac_trial = np.fft.irfft(H_trial * np.conj(H_trial), n=n_fft)
                val_trial = np.max(np.abs(ac_trial)) / (new_total * new_total)
                
                if val_trial < current_val - 1e-12:
                    h = trial
                    total = new_total
                    current_val = val_trial
                    improved = True
                    H = np.fft.rfft(h, n=n_fft)
                    ac_full = np.fft.irfft(H * np.conj(H), n=n_fft)
    
    return h, current_val


def _genetic_algorithm(pop_size=50, generations=50, time_budget=40.0):
    """
    Simple genetic algorithm to explore the solution space.
    """
    start_time = time.time()
    n_fft = 2 * N
    
    # Initialize population
    population = []
    # Seed with diverse initial solutions
    population.append(np.ones(N))
    population.append(np.ones(N) * -1)
    for i in range(10):
        population.append(_m_sequence(N, i))
    for i in range(10):
        population.append(_balanced_sequence(N, 1000 + i * 13))
    for freq in [1, 2, 3, 5, 7]:
        for phase in [0, np.pi/4, np.pi/2]:
            population.append(_sinusoidal_balance(N, freq, phase))
    for f0, f1 in [(1, 5), (2, 10), (5, 20)]:
        population.append(_chirp_sequence(N, f0, f1))
    
    # Pad to pop_size
    rng = np.random.RandomState(42)
    while len(population) < pop_size:
        population.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate initial population
    def evaluate(h):
        total = h.sum()
        if abs(total) < 1e-12:
            return np.inf
        H = np.fft.rfft(h, n=n_fft)
        ac = np.fft.irfft(H * np.conj(H), n=n_fft)
        return np.max(np.abs(ac)) / (total * total)
    
    fitness = [evaluate(h) for h in population]
    
    best_idx = np.argmin(fitness)
    best_h = population[best_idx].copy()
    best_val = fitness[best_idx]
    
    for gen in range(generations):
        if time.time() - start_time > time_budget:
            break
        
        # Select parents (tournament selection)
        offspring = []
        while len(offspring) < pop_size:
            # Tournament of 3
            idxs = rng.choice(len(population), size=3, replace=False)
            parent1 = population[idxs[np.argmin([fitness[i] for i in idxs])]]
            idxs2 = rng.choice(len(population), size=3, replace=False)
            parent2 = population[idxs2[np.argmin([fitness[i] for i in idxs2])]]
            
            # Crossover
            if rng.random() < 0.7:
                crossover_point = rng.randint(1, N)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            else:
                child = parent1.copy()
            
            # Mutation
            n_mut = rng.randint(1, 10)
            mut_idx = rng.choice(N, size=n_mut, replace=False)
            child[mut_idx] *= -1
            
            offspring.append(child)
        
        # Evaluate offspring
        offspring_fitness = [evaluate(h) for h in offspring]
        
        # Elitism: keep the best
        all_individuals = population + offspring
        all_fitness = fitness + offspring_fitness
        sort_idx = np.argsort(all_fitness)
        population = [all_individuals[i] for i in sort_idx[:pop_size]]
        fitness = [all_fitness[i] for i in sort_idx[:pop_size]]
        
        if fitness[0] < best_val:
            best_val = fitness[0]
            best_h = population[0].copy()
    
    return best_h, best_val


def construct():
    """
    Construct a signed step function that minimizes C3' (maximum absolute 
    autoconvolution divided by squared integral).
    """
    global _DEADLINE
    
    # Phase 1: Generate diverse candidates
    candidates = []
    
    # M-sequences and pseudo-random sequences
    for i in range(20):
        candidates.append(_m_sequence(N, i))
    
    # Balanced random sequences
    for i in range(30):
        candidates.append(_balanced_sequence(N, 5000 + i * 7))
    
    # Sinusoidal sequences
    for freq in range(1, 25):
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            candidates.append(_sinusoidal_balance(N, freq, phase))
    
    # Chirp sequences
    for f0 in [1, 2, 3, 5]:
        for f1 in [5, 10, 20, 30, 50]:
            if f1 != f0:
                candidates.append(_chirp_sequence(N, f0, f1))
    
    # All ones and all minus ones
    candidates.append(np.ones(N))
    candidates.append(-np.ones(N))
    
    # Evaluate all candidates
    best_h = None
    best_val = np.inf
    
    for h in candidates:
        if time.time() > _DEADLINE - 30:
            break
        try:
            val = _c3_prime_val(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
        except Exception:
            continue
    
    # Phase 2: Genetic algorithm exploration
    if time.time() < _DEADLINE - 40:
        ga_h, ga_val = _genetic_algorithm(pop_size=60, generations=80, time_budget=35.0)
        if ga_val < best_val:
            best_val = ga_val
            best_h = ga_h
    
    # Phase 3: Local refinement from best
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_refined, val_refined = _local_search_flip(best_h, max_sweeps=80, time_budget=15.0)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined
    
    # Phase 4: Segment flip refinement
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_seg, val_seg = _local_search_segment_flip(best_h, seg_lengths=[2, 3, 4], max_sweeps=30, time_budget=8.0)
        if val_seg < best_val:
            best_val = val_seg
            best_h = h_seg
    
    # Final fallback
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
