import numpy as np

N = 400

def construct():
    """
    Genetic Algorithm to minimize the sum of squares of non-zero autocorrelations.
    """
    
    def compute_autocorr_sum_sq(x):
        """
        Compute sum of squares of non-zero lags of linear autocorrelation using FFT.
        x is a vector of length N with values in {-1, 1}.
        """
        # Zero pad to 2*N to compute linear autocorrelation via circular autocorrelation
        x_pad = np.zeros(2 * N)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        # Power spectrum
        power = X * np.conj(X)
        # Inverse FFT to get circular autocorrelation
        R_circ = np.fft.ifft(power).real
        # Linear autocorrelation is the first N elements of R_circ (lag 0 to N-1)
        # R_circ[0] is lag 0 (energy), R_circ[1] is lag 1, ..., R_circ[N-1] is lag N-1
        # We want to minimize sum of squares of lags 1 to N-1.
        # Note: The linear autocorrelation for lags 1..N-1 corresponds to R_circ[1:N].
        # The lags -1..-(N-1) are symmetric, so their squares are the same.
        # The task says "sum of squares of non-zero autocorrelations".
        # Usually, this implies the full autocorrelation sequence excluding lag 0.
        # R[k] for k=1..N-1 and R[k] for k=1-(N-1).
        # Since R is even, sum_{k=1}^{N-1} R[k]^2 + sum_{k=-(N-1)}^{-1} R[k]^2 = 2 * sum_{k=1}^{N-1} R[k]^2.
        # Minimizing one is equivalent to minimizing the other.
        non_central = R_circ[1:N]
        return np.sum(non_central**2)

    # Initialize population
    pop_size = 100
    generations = 200
    np.random.seed(42)
    
    # Start with random population
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Compute initial scores
    scores = np.array([compute_autocorr_sum_sq(ind) for ind in population])
    
    best_x = population[np.argmin(scores)]
    best_score = np.min(scores)
    
    # GA Loop
    for gen in range(generations):
        # Sort population by score
        order = np.argsort(scores)
        population = population[order]
        scores = scores[order]
        
        # Elitism: keep top 2
        new_population = [population[0].copy(), population[1].copy()]
        
        # Fill rest of population
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                # Pick 3 random individuals
                idxs = np.random.choice(pop_size, size=3, replace=False)
                # Pick the best (lowest score) among them
                best_idx = idxs[np.argmin(scores[idxs])]
                return population[best_idx]
            
            p1 = select()
            p2 = select()
            
            # Uniform Crossover
            mask = np.random.rand(N) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation: flip bits with probability 0.01
            mut_mask = np.random.rand(N) < 0.01
            child[mut_mask] *= -1
            
            new_population.append(child)
        
        population = np.array(new_population)
        scores = np.array([compute_autocorr_sum_sq(ind) for ind in population])
        
        # Update best
        gen_best_idx = np.argmin(scores)
        if scores[gen_best_idx] < best_score:
            best_score = scores[gen_best_idx]
            best_x = population[gen_best_idx].copy()

    return best_x

# EVOLVE-BLOCK-END
