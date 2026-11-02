"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_overlaps(h):
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        overlaps[k] = np.dot(h, 1.0 - np.roll(h, -k))
    return overlaps

def _max_overlap(h):
    return np.max(_compute_overlaps(h))

def _project_to_feasible(h):
    """Project h onto {x in [0,1]^N : sum(x) = N/2} via bisection on offset."""
    lo = -1.0
    hi = 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        h_proj = np.clip(h - mid, 0, 1)
        s = np.sum(h_proj)
        if s > N / 2:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    h_proj = np.clip(h - alpha, 0, 1)
    
    # Fine tuning for exact sum
    for _ in range(20):
        s = np.sum(h_proj)
        diff = N / 2 - s
        if abs(diff) < 1e-12:
            break
        idx = np.where((h_proj > 1e-9) & (h_proj < 1 - 1e-9))[0]
        if len(idx) == 0:
            break
        h_proj[idx] += diff / len(idx)
        h_proj = np.clip(h_proj, 0, 1)
    return h_proj

def _phase_to_signal(phases, amplitudes):
    """Convert phase spectrum to real signal via inverse FFT."""
    n = len(amplitudes)
    # Build complex spectrum: H[0] is real (DC), H[N/2] is real (Nyquist)
    H = np.zeros(n, dtype=complex)
    H[0] = amplitudes[0]  # DC component (real)
    for k in range(1, n // 2):
        H[k] = amplitudes[k] * np.exp(1j * phases[k])
        H[n - k] = amplitudes[k] * np.exp(-1j * phases[k])
    if n % 2 == 0:
        H[n // 2] = amplitudes[n // 2]  # Nyquist (real)
    
    h = np.fft.ifft(H).real
    return h

def _design_amplitudes(n):
    """Design amplitude spectrum for flat autocorrelation target.
    
    For h with sum=N/2, the overlap at shift k is:
    overlap_k = sum_i h[i]*(1-h[i-k]) = N/2 - autocorr_k
    
    where autocorr_k = sum_i h[i]*h[i-k]
    
    We want overlap_k ≈ constant = target.
    So autocorr_k ≈ N/2 - target for all k != 0, and autocorr_0 = sum h^2.
    
    In Fourier domain, autocorrelation's DFT is |H|^2.
    For flat autocorrelation (except DC), we need |H[k]|^2 to be constant for k != 0.
    
    Let |H[k]|^2 = A for k=1,...,N/2 (symmetric).
    Then autocorr_k (k!=0) = (1/N) * sum_{j=1}^{N/2} A * 2*cos(2*pi*j*k/N)
    For this to be constant, we need A=0 or N=1... 
    Actually, a constant autocorrelation for all k!=0 is only achievable with 
    specific designs. Let's use a different approach: minimize variance of overlaps.
    
    Instead, let's parameterize and optimize directly.
    """
    # Use equal amplitudes as starting point
    amps = np.ones(n)
    return amps

def _local_refine(h, deadline, max_iter=500):
    """Local gradient-based refinement of h."""
    h = _project_to_feasible(h.copy())
    best_val = _max_overlap(h)
    
    beta = 50.0
    lr = 0.02
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        overlaps = _compute_overlaps(h)
        m = np.max(overlaps)
        exp_overlaps = np.exp((overlaps - m) * beta)
        sum_exp = np.sum(exp_overlaps)
        weights = exp_overlaps / sum_exp
        
        grad = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-10:
                g_k = (1.0 - np.roll(h, -k)) - np.roll(h, k)
                grad += weights[k] * g_k
        
        # Line search
        step = lr
        improved = False
        for _ in range(20):
            h_new = h - step * grad
            h_new = _project_to_feasible(h_new)
            val_new = _max_overlap(h_new)
            if val_new < best_val - 1e-12:
                h = h_new
                best_val = val_new
                improved = True
                break
            step *= 0.5
        
        if not improved:
            beta *= 1.1
            lr *= 0.95
        else:
            beta *= 1.01
            lr *= 0.995
    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initialize with uniform
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _max_overlap(best_h)
    
    # Phase 1: Genetic algorithm over phase spectra
    n_freqs = N // 2 + 1  # Number of unique frequency bins
    # Amplitudes: use a fixed design that gives sum = N/2
    # h_mean = 0.5, so DC component H[0] = N * 0.5 = N/2
    # For the rest, we'll use small amplitudes to create variation
    base_amp = 0.0  # Start with flat (all 0.5)
    
    # Create initial population of random phase spectra
    pop_size = 40
    population = []
    
    for i in range(pop_size):
        if i == 0:
            phases = np.zeros(n_freqs)
        else:
            phases = np.random.uniform(0, 2 * np.pi, n_freqs)
        population.append(phases)
    
    # Evaluate initial population
    pop_scores = []
    for phases in population:
        # Build signal: H[0] = N/2 (to get mean 0.5), others from phases with small amp
        H = np.zeros(N, dtype=complex)
        H[0] = N / 2.0  # DC: ensures mean = 0.5
        # Use small amplitudes for other frequencies
        amp_scale = 0.3
        for k in range(1, N // 2):
            H[k] = amp_scale * np.exp(1j * phases[k])
            H[N - k] = amp_scale * np.exp(-1j * phases[k])
        if N % 2 == 0:
            H[N // 2] = amp_scale * np.cos(phases[N // 2])
        
        h_raw = np.fft.ifft(H).real
        h = _project_to_feasible(h_raw)
        val = _max_overlap(h)
        pop_scores.append(val)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # GA loop
    gen = 0
    max_gens = 300
    while time.time() < deadline - 30.0 and gen < max_gens:
        gen += 1
        
        # Sort by score
        sorted_idx = np.argsort(pop_scores)
        population = [population[i] for i in sorted_idx]
        pop_scores = [pop_scores[i] for i in sorted_idx]
        
        # Elitism: keep top 5
        elite = population[:5]
        
        # Generate new population
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            # Tournament selection
            t1 = random.randint(0, min(10, pop_size - 1))
            t2 = random.randint(0, min(10, pop_size - 1))
            if pop_scores[t1] <= pop_scores[t2]:
                parent = population[t1]
            else:
                parent = population[t2]
            
            # Mutation
            if random.random() < 0.7:
                # Gaussian mutation on phases
                mutation_rate = 0.3 / (1.0 + 0.05 * gen)
                child = parent + np.random.normal(0, mutation_rate, n_freqs)
                child = np.mod(child, 2 * np.pi)
            else:
                # Crossover
                t3 = random.randint(0, min(10, pop_size - 1))
                if pop_scores[t3] <= pop_scores[t2 if t2 != t3 else t1]:
                    parent2 = population[t3]
                else:
                    parent2 = population[t2 if t2 != t3 else t1]
                cross_point = random.randint(1, n_freqs - 1)
                child = np.concatenate([parent[:cross_point], parent2[cross_point:]])
            
            new_pop.append(child)
        
        population = new_pop
        
        # Evaluate new population
        for i, phases in enumerate(population):
            H = np.zeros(N, dtype=complex)
            H[0] = N / 2.0
            amp_scale = 0.3
            for k in range(1, N // 2):
                H[k] = amp_scale * np.exp(1j * phases[k])
                H[N - k] = amp_scale * np.exp(-1j * phases[k])
            if N % 2 == 0:
                H[N // 2] = amp_scale * np.cos(phases[N // 2])
            
            h_raw = np.fft.ifft(H).real
            h = _project_to_feasible(h_raw)
            val = _max_overlap(h)
            pop_scores[i] = val
            if val < best_val:
                best_val = val
                best_h = h.copy()
        
        # Adaptive amplitude scaling: try different amp scales
        if gen % 10 == 0:
            # Try a few different amplitude scales
            for amp_try in [0.1, 0.5, 0.7, 1.0]:
                if time.time() > deadline - 30.0:
                    break
                for trial in range(3):
                    phases = np.random.uniform(0, 2 * np.pi, n_freqs)
                    H = np.zeros(N, dtype=complex)
                    H[0] = N / 2.0
                    for k in range(1, N // 2):
                        H[k] = amp_try * np.exp(1j * phases[k])
                        H[N - k] = amp_try * np.exp(-1j * phases[k])
                    if N % 2 == 0:
                        H[N // 2] = amp_try * np.cos(phases[N // 2])
                    
                    h_raw = np.fft.ifft(H).real
                    h = _project_to_feasible(h_raw)
                    val = _max_overlap(h)
                    if val < best_val:
                        best_val = val
                        best_h = h.copy()
    
    # Phase 2: Local refinement from best
    if time.time() < deadline - 5.0:
        refined_h, refined_val = _local_refine(best_h, deadline - 2.0, max_iter=300)
        if refined_val < best_val:
            best_val = refined_val
            best_h = refined_h
    
    return best_h
# EVOLVE-BLOCK-END
