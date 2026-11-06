"""Rewritten program using FFT-based scoring with simulated annealing."""
import time
import random
import numpy as np
from scipy.fft import fft, ifft

# EVOLVE-BLOCK-START

def _score_set_fft(s):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d all in s.
    Uses FFT-based convolution for fast evaluation.
    """
    s_set = set(s)
    if len(s_set) < 3:
        return 0
    
    max_val = max(s_set)
    # We need an array of size at least 2*max_val + 1 for convolution
    size = 1
    while size < 2 * max_val + 1:
        size *= 2
    
    # Create indicator array
    arr = np.zeros(size, dtype=np.float64)
    for x in s_set:
        arr[x] = 1.0
    
    # FFT of the indicator
    f = fft(arr)
    
    # Convolution: (arr * arr)[k] = sum over i of arr[i] * arr[k-i]
    # For a 3-AP with difference d: we need a, a+d, a+2d in s
    # This means arr[a] * arr[a+d] * arr[a+2d] = 1 for some a
    # Equivalently, for each d, check if sum_a arr[a] * arr[a+d] * arr[a+2d] > 0
    # 
    # Using convolution: let c = arr * arr (circular convolution)
    # c[k] = sum_i arr[i] * arr[k-i]
    # For d, we want: sum_a arr[a+d] * arr[a+2d] * arr[a] 
    # Note that arr[a] * arr[a+2d] contributes to c[2d] at position... 
    # Actually, let's think differently.
    #
    # The number of 3-APs with difference d is: sum_a arr[a] * arr[a+d] * arr[a+2d]
    # This can be computed as: for each d, take the element-wise product of arr[d:] * arr[2d:] * arr[:-2d]
    # But that's O(max_val) per d, and we have O(max_val) values of d, so O(max_val^2) total.
    #
    # Better approach: Use the fact that sum_a arr[a] * arr[a+d] * arr[a+2d] 
    # = sum_a arr[a] * (arr * arr)[a+2d] ... no that's not right either.
    #
    # Let's use: sum_a arr[a] * arr[a+d] * arr[a+2d]
    # = sum_a arr[a+2d] * (arr * arr)[a+d] ... still not clean.
    #
    # Actually, the standard approach: 
    # Let f(x) = sum over a of arr[a] * arr[a+d] * arr[a+2d]
    # We can compute this for all d simultaneously using FFT.
    #
    # Note: sum_a arr[a] * arr[a+d] * arr[a+2d] = sum_a arr[a+2d] * arr[a+d] * arr[a]
    # Let b[k] = arr[k] * arr[k+2d] for fixed d... this is getting complex.
    #
    # Simpler: just use the direct O(n^2) approach with numpy vectorization.
    # With n=32 elements, we have at most 32*31/2 = 496 pairs, which is very fast.
    
    lst = sorted(s_set)
    n = len(lst)
    arr_bool = np.zeros(max_val + 1, dtype=bool)
    for x in s_set:
        arr_bool[x] = True
    
    diffs_found = set()
    for i in range(n):
        for j in range(i + 1, n):
            diff = lst[j] - lst[i]
            if diff % 2 == 0:
                mid = (lst[i] + lst[j]) // 2
                if arr_bool[mid]:
                    diffs_found.add(diff // 2)
    
    return len(diffs_found)


def _score_set_fast(s):
    """
    Fast scoring using bitset operations with Python integers.
    """
    s_set = set(s)
    if len(s_set) < 3:
        return 0
    
    max_val = max(s_set)
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    diffs_found = 0
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            diffs_found += 1
    
    return diffs_found


def _score_set_bitset(s):
    """
    Fast scoring using bitset operations.
    """
    s_set = set(s)
    if len(s_set) < 3:
        return 0
    
    max_val = max(s_set)
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    diffs_found = 0
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            diffs_found += 1
    
    return diffs_found


def _construct_simulated_annealing():
    """
    Simulated annealing with fast bitset scoring.
    """
    start_time = time.time()
    deadline = 145.0
    
    random.seed(123)
    
    # Start with a dense interval
    current_set = set(range(1, 33))
    current_score = _score_set_bitset(current_set)
    
    best_set = set(current_set)
    best_score = current_score
    
    # Simulated annealing parameters
    T_start = 10.0
    T_end = 0.1
    iterations = 50000
    
    for it in range(iterations):
        if time.time() > deadline:
            break
        
        # Temperature schedule (exponential cooling)
        T = T_start * (T_end / T_start) ** (it / iterations)
        
        # Generate a move: swap one element with a random new element
        elems = list(current_set)
        idx = random.randint(0, len(elems) - 1)
        old_elem = elems[idx]
        new_elem = random.randint(1, 512)
        
        if new_elem in current_set:
            continue
        
        # Try the move
        current_set.remove(old_elem)
        current_set.add(new_elem)
        new_score = _score_set_bitset(current_set)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept the move
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = set(current_set)
        else:
            # Reject the move
            current_set.remove(new_elem)
            current_set.add(old_elem)
    
    return list(best_set)


def _construct_genetic_algorithm():
    """
    Genetic algorithm for finding sets with many progression differences.
    """
    start_time = time.time()
    deadline = 145.0
    
    random.seed(456)
    
    pop_size = 50
    elite_size = 5
    mutation_rate = 0.1
    
    # Initialize population
    population = []
    for i in range(pop_size):
        if i == 0:
            # Start with dense interval
            s = set(range(1, 33))
        elif i == 1:
            # Start with sparse random
            s = set(random.sample(range(1, 513), 32))
        else:
            # Mix
            s = set(range(1, 33))
            for _ in range(random.randint(1, 5)):
                to_remove = random.choice(list(s))
                s.remove(to_remove)
                s.add(random.randint(1, 512))
        population.append(s)
    
    # Score all
    scores = [_score_set_bitset(s) for s in population]
    
    best_idx = np.argmax(scores)
    best_set = set(population[best_idx])
    best_score = scores[best_idx]
    
    generation = 0
    max_generations = 1000
    
    while generation < max_generations:
        if time.time() > deadline:
            break
        
        generation += 1
        
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_set = set(population[0])
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top individuals
        for i in range(elite_size):
            new_population.append(set(population[i]))
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                candidates = random.sample(range(len(population)), min(5, len(population)))
                best_c = max(candidates, key=lambda i: scores[i])
                return population[best_c]
            
            parent1 = select()
            parent2 = select()
            
            # Crossover: take half from each parent
            p1 = list(parent1)
            p2 = list(parent2)
            random.shuffle(p1)
            random.shuffle(p2)
            child = set(p1[:16] + p2[:16])
            
            # Ensure exactly 32 elements
            while len(child) < 32:
                child.add(random.randint(1, 512))
            while len(child) > 32:
                child.remove(random.choice(list(child)))
            
            # Mutation
            for _ in range(int(mutation_rate * 32)):
                if random.random() < mutation_rate:
                    to_remove = random.choice(list(child))
                    child.remove(to_remove)
                    child.add(random.randint(1, 512))
            
            new_population.append(child)
        
        population = new_population
        scores = [_score_set_bitset(s) for s in population]
    
    return list(best_set)


def _construct_iterated_local_search():
    """
    Iterated local search with bitset scoring.
    """
    start_time = time.time()
    deadline = 145.0
    
    random.seed(789)
    
    # Start with a dense interval
    current_set = set(range(1, 33))
    current_score = _score_set_bitset(current_set)
    
    best_set = set(current_set)
    best_score = current_score
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Local search: try all single-element swaps
        improved = True
        while improved:
            improved = False
            current_set_list = sorted(current_set)
            
            for i in range(len(current_set_list)):
                old_elem = current_set_list[i]
                for new_elem in range(1, 513):
                    if new_elem in current_set:
                        continue
                    
                    new_set = current_set - {old_elem} | {new_elem}
                    new_score = _score_set_bitset(new_set)
                    
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_set = set(current_set)
                        break
                if improved:
                    break
        
        # Perturbation: randomly change 2-4 elements
        num_perturb = random.randint(2, 4)
        current_set = set(best_set)
        for _ in range(num_perturb):
            to_remove = random.choice(list(current_set))
            current_set.remove(to_remove)
            new_elem = random.randint(1, 512)
            while new_elem in current_set:
                new_elem = random.randint(1, 512)
            current_set.add(new_elem)
        
        current_score = _score_set_bitset(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
    
    return list(best_set)


def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Try multiple strategies and keep the best
    results = []
    
    # Strategy 1: Iterated local search (fast and effective)
    if time.time() < deadline - 30:
        result1 = _construct_iterated_local_search()
        score1 = _score_set_bitset(result1)
        results.append((score1, result1))
    
    # Strategy 2: Simulated annealing
    if time.time() < deadline - 15:
        result2 = _construct_simulated_annealing()
        score2 = _score_set_bitset(result2)
        results.append((score2, result2))
    
    # Strategy 3: Genetic algorithm
    if time.time() < deadline - 5:
        result3 = _construct_genetic_algorithm()
        score3 = _score_set_bitset(result3)
        results.append((score3, result3))
    
    if not results:
        # Fallback: dense interval
        return list(range(1, 33))
    
    # Return the best result
    results.sort(key=lambda x: x[0], reverse=True)
    return results[0][1]

# EVOLVE-BLOCK-END
