import numpy as np
import time

N = 400

def construct():
    """
    Construct a signed step function (400 intervals) to minimize the upper bound on C3.
    We assume the metric is related to the peak sidelobe of the linear autocorrelation.
    We use a Genetic Algorithm to minimize max_{tau != 0} |r(tau)|.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    rng = np.random.RandomState(42)
    
    # Population size
    pop_size = 100
    # Initialize population: mix of random and structured (ones, alternating)
    pop = np.zeros((pop_size, N), dtype=np.float64)
    pop[0] = 1.0  # np.ones
    pop[1] = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    for i in range(2, pop_size):
        pop[i] = rng.choice([-1.0, 1.0], size=N)
    
    def get_autocorr_peak(h):
        # Linear autocorrelation
        # r(tau) = sum h[i] h[i+tau]
        # We want max |r(tau)| for tau != 0
        # Using FFT for speed
        H = np.fft.fft(h, n=2*N)
        R = np.fft.ifft(H * np.conj(H)).real
        # R[0] is r(0) = N
        # R[1:N] are r(1) to r(N-1)
        # R[N+1:] are r(-(N-1)) to r(-1)
        sidelobes = np.abs(np.concatenate([R[1:N], R[N+1:]]))
        return np.max(sidelobes) if len(sidelobes) > 0 else 0.0

    def get_score(h):
        return get_autocorr_peak(h)

    # Evaluate initial population
    scores = np.array([get_score(h) for h in pop])
    best_idx = np.argmin(scores)
    best_h = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    # Genetic Algorithm Loop
    generations = 0
    while time.time() < deadline - 5.0:
        generations += 1
        new_pop = np.zeros_like(pop)
        
        # Elitism: keep top 10%
        elite_size = pop_size // 10
        elite_indices = np.argsort(scores)[:elite_size]
        new_pop[:elite_size] = pop[elite_indices]
        
        # Selection and Crossover
        for i in range(elite_size, pop_size):
            # Tournament selection
            idx1 = rng.randint(0, pop_size)
            idx2 = rng.randint(0, pop_size)
            if scores[idx1] < scores[idx2]:
                parent1 = pop[idx1]
            else:
                parent1 = pop[idx2]
                
            idx3 = rng.randint(0, pop_size)
            idx4 = rng.randint(0, pop_size)
            if scores[idx3] < scores[idx4]:
                parent2 = pop[idx3]
            else:
                parent2 = pop[idx4]
            
            # Uniform Crossover
            mask = rng.rand(N) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            mutation_rate = 0.05
            mut_mask = rng.rand(N) < mutation_rate
            child[mut_mask] *= -1.0
            
            new_pop[i] = child
        
        pop = new_pop
        scores = np.array([get_score(h) for h in pop])
        
        new_best_idx = np.argmin(scores)
        if scores[new_best_idx] < best_score:
            best_score = scores[new_best_idx]
            best_h = pop[new_best_idx].copy()
            
        # Adaptive mutation rate
        if generations % 10 == 0:
            # Check for stagnation
            pass

    return best_h
