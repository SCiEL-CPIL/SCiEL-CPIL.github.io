"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Uses a Genetic Algorithm to find a binary sequence {-1, 1}^N
    that minimizes the Integrated Sidelobe Level (ISL) of its autocorrelation.
    ISL = sum_{k=1}^{N-1} R(k)^2.
    """
    
    # Population size
    pop_size = 200
    # Number of generations
    generations = 100
    # Crossover rate
    crossover_rate = 0.8
    # Mutation rate (probability per bit)
    mutation_rate = 0.01
    
    # Initialize population with random binary sequences
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    def compute_fitness(pop):
        """
        Compute ISL for each sequence in the population.
        Uses FFT for efficiency.
        ISL = sum(R[k]^2 for k in 1..N-1)
        Note: R = ifft(|fft(f)|^2)
        Sum of squares of R is (1/N) * sum(|fft(R)|^2) = (1/N) * sum(|fft(ifft(|F|^2))|^2)
        fft(ifft(X)) = X. So fft(R) = |F|^2.
        Sum(R^2) = (1/N) * sum(| |F|^2 |^2) = (1/N) * sum(|F|^4).
        R(0) = N.
        ISL = Sum(R^2) - R(0)^2 = (1/N) * sum(|F|^4) - N^2.
        """
        F = np.fft.fft(pop, axis=1)
        power = F * np.conj(F)
        # sum of |F|^4
        sum_f4 = np.sum(np.abs(power)**2, axis=1)
        total_energy = sum_f4 / N
        isls = total_energy - N**2
        return isls

    # Initial fitness
    fitness = compute_fitness(population)
    best_idx = np.argmin(fitness)
    best_seq = population[best_idx].copy()
    best_score = fitness[best_idx]
    
    for gen in range(generations):
        # Selection: Tournament selection
        new_population = np.empty_like(population)
        
        # Elitism: Keep the best individual
        new_population[0] = best_seq
        
        for i in range(1, pop_size):
            # Select parents via tournament
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if fitness[idx1] < fitness[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
                
            idx3 = np.random.randint(0, pop_size)
            idx4 = np.random.randint(0, pop_size)
            if fitness[idx3] < fitness[idx4]:
                parent2 = population[idx3]
            else:
                parent2 = population[idx4]
            
            # Crossover: Single point
            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, N)
                child = np.concatenate([parent1[:point], parent2[point:]])
            else:
                child = parent1.copy()
            
            # Mutation: Bit flip
            if np.random.rand() < 0.9: # Apply mutation with high prob to maintain diversity
                mask = np.random.rand(N) < mutation_rate
                child[mask] *= -1
                
            new_population[i] = child
        
        population = new_population
        fitness = compute_fitness(population)
        
        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_score:
            best_score = fitness[gen_best_idx]
            best_seq = population[gen_best_idx].copy()
            
    return best_seq
# EVOLVE-BLOCK-END
