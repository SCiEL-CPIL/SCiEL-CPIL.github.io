import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute indices for target sums
    sum_to_idx = {s: i for i, s in enumerate(target_sums)}

    def get_counts(A_set):
        """Compute representation counts for target sums given set A using convolution."""
        # Vectorized convolution is much faster than O(k^2) loops for large k
        # A is a subset of [0, 128]. We represent it as a binary vector.
        vec = np.zeros(129, dtype=np.int32)
        for a in A_set:
            vec[a] = 1
        
        # Convolution of vec with itself gives counts of ordered pairs (i, j)
        # such that i + j = s.
        # We want unordered pairs {i, j} with i <= j.
        # Let C[s] be the count from convolution (ordered).
        # C[s] = 2 * (# pairs i < j) + (# pairs i = j)
        # We want U[s] = (# pairs i < j) + (# pairs i = j)
        # Note: # pairs i = j is 1 if s is even and s/2 in A, else 0.
        # Let D[s] be the number of diagonal pairs (i=i). D[s] is 1 if s%2==0 and s/2 in A, else 0.
        # Then C[s] = 2 * (U[s] - D[s]) + D[s] = 2*U[s] - D[s].
        # So U[s] = (C[s] + D[s]) / 2.
        
        conv = np.convolve(vec, vec, mode='full')
        # conv index k corresponds to sum k.
        # We care about sums in [128, 256].
        counts = np.zeros(num_targets, dtype=np.int32)
        
        for idx, s in enumerate(target_sums):
            c_val = conv[s]
            # Calculate D[s]
            d_val = 0
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and vec[half] == 1:
                    d_val = 1
            counts[idx] = (c_val + d_val) // 2
            
        return counts

    def calculate_peak(counts):
        """Return the maximum multiplicity in the target interval, with penalty for invalidity."""
        if np.any(counts == 0):
            return 1000 + np.sum(counts == 0)
        return np.max(counts)

    def cost(A_set):
        counts = get_counts(A_set)
        return calculate_peak(counts)

    def initial_solution():
        """Create a valid solution using a greedy approach."""
        # Start with a dense set
        A = set(random.sample(range(129), random.randint(40, 60)))
        counts = get_counts(A)
        
        # Greedy fill if invalid
        if np.any(counts == 0):
            uncovered = [i for i in range(num_targets) if counts[i] == 0]
            candidates = [x for x in range(129) if x not in A]
            random.shuffle(candidates)
            
            for x in candidates:
                if not uncovered:
                    break
                A.add(x)
                new_counts = get_counts(A)
                new_uncovered = [i for i in range(num_targets) if new_counts[i] == 0]
                if len(new_uncovered) < len(uncovered):
                    uncovered = new_uncovered
                    counts = new_counts
                else:
                    A.remove(x)
        
        # If still invalid, add everything (fallback)
        if np.any(get_counts(A) == 0):
             A = set(range(129))
             
        return A

    # Initialize Population
    pop_size = 10
    population = []
    for _ in range(pop_size):
        A = initial_solution()
        c = cost(A)
        population.append((A, c))
        
    # Sort initial population
    population.sort(key=lambda x: x[1])
    best_A, best_cost = population[0]
    
    # Genetic Algorithm Loop
    T_start = 50.0
    T_end = 1.0
    alpha = 0.9995
    
    T = T_start
    gen = 0
    max_gens = 5000
    
    while time.time() < deadline - 2.0 and T > T_end:
        gen += 1
        
        # Selection: Roulette wheel or Tournament
        # We use Tournament Selection (k=2)
        def select():
            i = random.randint(0, len(population)-1)
            j = random.randint(0, len(population)-1)
            if population[i][1] <= population[j][1]:
                return population[i][0]
            else:
                return population[j][0]
                
        new_population = []
        
        # Elitism: Keep the best few
        for i in range(min(3, len(population))):
            new_population.append(population[i])
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            p1 = select()
            p2 = select()
            
            # Crossover: Union of parents
            # This ensures validity is more likely preserved if parents are valid
            child_A = set(p1) | set(p2)
            
            # Mutation
            # Strategy: 
            # 1. Try to remove elements that contribute to the peak.
            # 2. Random add/remove.
            
            counts = get_counts(child_A)
            peak_val = np.max(counts) if not np.any(counts==0) else 1000
            
            # Identify elements contributing to the peak sums
            # If peak is high, try removing elements from sums that have peak count.
            # This is heuristic.
            
            if random.random() < 0.7:
                # Peak-directed mutation
                # Find sums with max count
                if not np.any(counts == 0):
                    max_count = np.max(counts)
                    # Get indices of sums with max count
                    peak_indices = np.where(counts == max_count)[0]
                    peak_sums = [target_sums[i] for i in peak_indices]
                    
                    # For each peak sum, find pairs in child_A that sum to it
                    # If we remove one element from a pair, we reduce count by 1 (for that sum)
                    # But it might increase count for other sums? No, removing an element only decreases counts.
                    # So removing an element from a peak sum is always safe for validity (if count > 1) 
                    # and reduces peak (if it was the only pair? No, reduces count).
                    
                    # Heuristic: Pick a random element from child_A that is part of a peak sum pair.
                    candidates_to_remove = []
                    for s in peak_sums:
                        # Find a in child_A such that s-a in child_A and a <= s-a
                        for a in child_A:
                            b = s - a
                            if a <= b and b in child_A:
                                candidates_to_remove.append(a)
                                candidates_to_remove.append(b)
                                break # Just find one pair for this sum to save time? 
                                # Actually, adding all is better for diversity.
                    
                    if candidates_to_remove:
                        # Remove one
                        x = random.choice(candidates_to_remove)
                        if x in child_A:
                            # Check if removal keeps it valid
                            child_A.remove(x)
                            new_counts = get_counts(child_A)
                            if np.any(new_counts == 0):
                                # Restore if invalid
                                child_A.add(x)
                            else:
                                # Valid, keep removal
                                pass
                        else:
                            # x was not in child_A? Shouldn't happen.
                            pass
                    else:
                        # No peak pairs found? (Shouldn't happen if valid)
                        # Random removal
                        if len(child_A) > 20:
                            x = random.choice(list(child_A))
                            child_A.remove(x)
                            new_counts = get_counts(child_A)
                            if np.any(new_counts == 0):
                                child_A.add(x)
                else:
                    # Invalid, try to fix by adding
                    # Find uncovered sums
                    uncovered = [i for i in range(num_targets) if counts[i] == 0]
                    if uncovered:
                         target_s = target_sums[uncovered[0]]
                         # Find a, b not in child_A such that a+b = target_s?
                         # Or a in child_A, b not in child_A?
                         # Simple: add a random element
                         if len(child_A) < 128:
                             x = random.randint(0, 128)
                             if x not in child_A:
                                 child_A.add(x)
            else:
                # Random mutation
                move_type = random.random()
                if move_type < 0.3:
                    x = random.randint(0, 128)
                    child_A.add(x)
                elif move_type < 0.6 and len(child_A) > 10:
                    x = random.choice(list(child_A))
                    child_A.remove(x)
                else:
                    if len(child_A) > 1:
                        x = random.choice(list(child_A))
                        child_A.remove(x)
                        y = random.randint(0, 128)
                        if y not in child_A:
                            child_A.add(y)
            
            child_cost = cost(child_A)
            new_population.append((child_A, child_cost))
            
        # Update best
        new_population.sort(key=lambda x: x[1])
        if new_population[0][1] < best_cost:
            best_cost = new_population[0][1]
            best_A = new_population[0][0]
            
        population = new_population
        T *= alpha
        
    # Final verification
    final_counts = get_counts(best_A)
    if np.any(final_counts == 0):
        # Fallback
        return list(range(64, 129))
        
    return list(best_A)
# EVOLVE-BLOCK-END
