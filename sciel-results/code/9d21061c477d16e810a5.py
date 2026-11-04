import numpy as np
import time

N = 400

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    def score(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        # np.convolve(h, h) is linear convolution
        z = np.convolve(h, h)
        # The metric takes max of signed entries
        # We want to minimize this value.
        # Note: If max(z) is negative, abs() makes it positive. 
        # Ideally we want max(z) to be close to 0.
        return abs(2 * N * np.max(z) / (s * s))

    # Strategy: 
    # Use a Genetic Algorithm to evolve a signal h with negative lobes.
    # The idea is that h(x) = Center(x) - Sides(x) can create destructive 
    # interference in the autocorrelation at the center lag.
    
    # Initial population: Mix of random and structured signals
    def generate_individual():
        # Create a signal with a positive center and negative sides
        # This structure is likely to reduce the central autocorrelation peak.
        
        # Random parameters for the "shape"
        center_width = np.random.uniform(0.1, 0.4)
        side_width = np.random.uniform(0.1, 0.4)
        side_distance = np.random.uniform(0.1, 0.4)
        center_amp = np.random.uniform(0.5, 1.5)
        side_amp = np.random.uniform(0.2, 1.0)
        
        x = np.linspace(0, 1, N)
        
        # Center bump (positive)
        h_center = center_amp * np.exp(-((x - 0.5) / center_width) ** 2)
        
        # Side bumps (negative)
        # Symmetric placement to maintain some structure
        h_side1 = -side_amp * np.exp(-((x - (0.5 - side_distance)) / side_width) ** 2)
        h_side2 = -side_amp * np.exp(-((x - (0.5 + side_distance)) / side_width) ** 2)
        
        h = h_center + h_side1 + h_side2
        
        # Add some random noise to break symmetry and explore
        noise_std = np.random.uniform(0, 0.1)
        h += noise_std * np.random.randn(N)
        
        return h

    def evaluate_population(pop):
        scores = []
        for h in pop:
            scores.append(score(h))
        return np.array(scores)

    def crossover(parent1, parent2):
        # Simple uniform crossover
        mask = np.random.rand(N) > 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def mutate(h, rate=0.1, sigma=0.05):
        # Gaussian mutation
        if np.random.rand() < rate:
            h = h + sigma * np.random.randn(N)
        return h

    # GA Parameters
    pop_size = 50
    num_generations = 200  # Adjust based on time
    elite_size = 5
    
    # Initialize population
    population = [generate_individual() for _ in range(pop_size)]
    
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Track time
    start_time = time.time()
    
    for gen in range(num_generations):
        # Check time
        if time.time() > deadline:
            break
            
        # Evaluate
        scores = evaluate_population(population)
        
        # Update best
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_h = population[min_idx].copy()
            
        # Sort population by score (lower is better)
        sorted_indices = np.argsort(scores)
        population = [population[i] for i in sorted_indices]
        scores = scores[sorted_indices]
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(population[i].copy())
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection (size 3)
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            idx3 = np.random.randint(0, pop_size)
            
            # Pick the best of the three
            if scores[idx1] < scores[idx2]:
                p1 = population[idx1]
            else:
                p1 = population[idx2]
                
            if scores[idx1] < scores[idx3]:
                p1 = population[idx1]
            elif scores[idx3] < scores[idx2]:
                p1 = population[idx3]
            else:
                p1 = population[idx2]
                
            # Second parent
            idx4 = np.random.randint(0, pop_size)
            idx5 = np.random.randint(0, pop_size)
            idx6 = np.random.randint(0, pop_size)
            
            if scores[idx4] < scores[idx5]:
                p2 = population[idx4]
            else:
                p2 = population[idx5]
                
            if scores[idx4] < scores[idx6]:
                p2 = population[idx4]
            elif scores[idx6] < scores[idx5]:
                p2 = population[idx6]
            else:
                p2 = population[idx5]
                
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            child = mutate(child, rate=0.3, sigma=0.02)
            
            new_population.append(child)
            
        population = new_population
        
        # Occasionally inject a new random individual to maintain diversity
        if gen % 10 == 0:
            population[-1] = generate_individual()
            
    return best_h
