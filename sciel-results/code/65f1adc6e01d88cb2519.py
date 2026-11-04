import time
import random
import math
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0.0
    max_u = max(U_set)
    # Compute sumset and diffset efficiently
    sums = set()
    diffs = set()
    diffs.add(0)
    U_list = sorted(U_set)
    for i in range(n):
        ui = U_list[i]
        for j in range(i, n):
            sums.add(ui + U_list[j])
        for j in range(i + 1, n):
            d = U_list[j] - ui
            diffs.add(d)
            diffs.add(-d)
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max U + 1)
    if len(sums) == 0:
        return 0.0
    ratio = len(diffs) / len(sums)
    if ratio <= 1.0:
        return 1.0
    if 2 * max_u + 1 <= 1:
        return 1.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _fast_score_set(U):
    """Fast score computation using numpy for larger sets."""
    if not U or 0 not in U:
        return 0.0
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n < 2:
        return 0.0
    max_u = U_arr[-1]
    
    # Compute sumset and diffset using broadcasting
    sums = np.add.outer(U_arr, U_arr)
    diffs = np.subtract.outer(U_arr, U_arr)
    
    sums_unique = np.unique(sums)
    diffs_unique = np.unique(diffs)
    
    ratio = len(diffs_unique) / len(sums_unique)
    if ratio <= 1.0:
        return 1.0
    if 2 * max_u + 1 <= 1:
        return 1.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _generate_candidates(U, limit_val, random_small_count=15):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    Also adds some random small integers for diversity.
    """
    U_set = set(U)
    candidates = set()
    U_list = sorted(U_set)
    n = len(U_list)
    for i in range(n):
        a = U_list[i]
        for j in range(i, n):
            b = U_list[j]
            s = a + b
            d = b - a
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
    # Add random small integers for diversity
    for _ in range(random_small_count):
        r = random.randint(1, min(50, limit_val))
        if r not in U_set:
            candidates.add(r)
    return [c for c in candidates if 0 < c <= limit_val]

def _generate_difference_set_seeds():
    """
    Generate seeds based on difference set theory.
    Perfect difference sets have |U-U| = n(n-1)+1 and |U+U| is relatively small.
    We use cyclic difference sets and related constructions.
    """
    seeds = []
    
    # Cyclic difference sets for small prime powers
    # For q = p^k, there exist difference sets with parameters (v, k, lambda)
    # where v = q^2 + q + 1, k = q + 1, lambda = 1
    
    for q in [2, 3, 4, 5, 7, 8, 9, 11, 13, 16]:
        v = q * q + q + 1
        k = q + 1
        # Construct a cyclic difference set
        # Use quadratic residues or simple construction
        if q % 2 == 1:
            # Use quadratic residues mod v
            residues = set()
            for x in range(1, v):
                residues.add((x * x) % v)
            U = sorted([0] + list(residues)[:k])
            if len(U) >= 2:
                seeds.append(U)
        else:
            # Use a simple pattern
            U = [0]
            val = 1
            for i in range(k - 1):
                val = (val * q) % v
                if val not in U and val > 0:
                    U.append(val)
            U = sorted(U)
            if len(U) >= 2:
                seeds.append(U)
    
    # Geometric progressions with various bases
    for base in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        for m in range(3, 12):
            U = [0]
            val = 1
            for i in range(m):
                val *= base
                if val > 10000:
                    break
                U.append(val)
            if len(U) >= 2:
                seeds.append(U)
    
    # Fibonacci-like sequences
    fib = [0, 1, 1]
    while len(fib) < 15:
        fib.append(fib[-1] + fib[-2])
    for start in range(0, 5):
        for length in range(4, 12):
            U = fib[start:start+length]
            if 0 not in U:
                U = [0] + U
            U = sorted(set(U))
            if len(U) >= 2 and max(U) < 10000:
                seeds.append(U)
    
    # Prime-based constructions
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for subset_size in range(3, 8):
        for combo in combinations(primes, subset_size):
            U = sorted([0] + list(combo))
            if max(U) < 500:
                seeds.append(U)
    
    # Remove duplicates
    unique_seeds = []
    seen = set()
    for s in seeds:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)
    
    return unique_seeds

def _local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = _score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = _generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = _generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:10]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def _genetic_crossover(U1, U2, max_val_limit):
    """
    Crossover operator for genetic algorithm.
    Combines elements from two parent sets.
    """
    U1_set = set(U1)
    U2_set = set(U2)
    
    # Select a random subset from each parent
    n1 = len(U1_set)
    n2 = len(U2_set)
    
    # Keep 0 always
    child = {0}
    
    # Randomly select elements from parent 1
    if n1 > 1:
        k1 = random.randint(max(1, n1 // 3), max(2, n1 // 2))
        elements1 = random.sample([x for x in U1_set if x != 0], min(k1, n1 - 1))
        child.update(elements1)
    
    # Randomly select elements from parent 2
    if n2 > 1:
        k2 = random.randint(max(1, n2 // 3), max(2, n2 // 2))
        elements2 = random.sample([x for x in U2_set if x != 0], min(k2, n2 - 1))
        child.update(elements2)
    
    # Add some new candidates from sums/diffs
    candidates = _generate_candidates(list(child), max_val_limit, random_small_count=5)
    if candidates and random.random() < 0.5:
        new_elem = random.choice(candidates[:5])
        child.add(new_elem)
    
    return sorted(child)

def _genetic_mutation(U, max_val_limit, mutation_rate=0.3):
    """
    Mutation operator for genetic algorithm.
    """
    U = list(U)
    U_set = set(U)
    
    # Randomly add or remove elements
    if random.random() < mutation_rate and len(U) > 2:
        # Remove a random element (not 0)
        removable = [x for x in U if x != 0]
        if removable:
            to_remove = random.choice(removable)
            U = [x for x in U if x != to_remove]
    
    if random.random() < mutation_rate:
        # Add a random candidate
        candidates = _generate_candidates(U, max_val_limit, random_small_count=3)
        if candidates:
            new_elem = random.choice(candidates[:5])
            if new_elem not in U:
                U.append(new_elem)
    
    # Occasionally replace an element
    if random.random() < mutation_rate * 0.5 and len(U) > 2:
        idx = random.randint(1, len(U) - 1)
        candidates = _generate_candidates(U, max_val_limit, random_small_count=3)
        if candidates:
            new_val = random.choice(candidates[:5])
            if new_val not in U:
                U[idx] = new_val
    
    return sorted(set(U))

def _genetic_algorithm(deadline, initial_population, pop_size=20, max_val_limit=1000, max_n_limit=40):
    """
    Genetic algorithm to optimize the set.
    """
    start_time = time.time()
    
    # Initialize population
    population = []
    for i in range(min(pop_size, len(initial_population))):
        population.append(list(initial_population[i]))
    
    # Fill remaining with random sets
    while len(population) < pop_size:
        n = random.randint(3, min(15, max_n_limit))
        elements = random.sample(range(1, max_val_limit + 1), n - 1)
        population.append(sorted([0] + elements))
    
    # Evaluate initial population
    scores = [_score_set(ind) for ind in population]
    
    best_idx = scores.index(max(scores))
    best_score = scores[best_idx]
    best_individual = list(population[best_idx])
    
    generation = 0
    no_improvement_gen = 0
    max_no_improvement = 10
    
    while time.time() < deadline - 5:
        generation += 1
        no_improvement_gen += 1
        
        # Selection: keep top half
        sorted_indices = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        survivors = [population[i] for i in sorted_indices[:pop_size // 2]]
        
        # Generate offspring
        new_population = list(survivors)
        
        while len(new_population) < pop_size:
            if random.random() < 0.7:
                # Crossover
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                child = _genetic_crossover(p1, p2, max_val_limit)
            else:
                # Mutation
                parent = random.choice(survivors)
                child = _genetic_mutation(parent, max_val_limit)
            
            # Enforce constraints
            if len(child) > max_n_limit:
                child = child[:max_n_limit]
            if max(child) > max_val_limit:
                child = [x for x in child if x <= max_val_limit]
            if 0 not in child:
                child = [0] + child
            child = sorted(set(child))
            
            if len(child) >= 2:
                new_population.append(child)
        
        # Evaluate new population
        new_scores = [_score_set(ind) for ind in new_population]
        
        # Check for improvement
        gen_best_score = max(new_scores)
        if gen_best_score > best_score + 1e-10:
            best_score = gen_best_score
            best_idx = new_scores.index(gen_best_score)
            best_individual = list(new_population[best_idx])
            no_improvement_gen = 0
        
        population = new_population
        scores = new_scores
        
        if no_improvement_gen > max_no_improvement:
            break
    
    return best_score, best_individual

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Initial seeds
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Structured seeds
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 4, 6, 10, 15, 21, 28], # Triangular + 1?
        [0, 1, 3, 6, 10, 15, 21, 28], # Triangular
    ]
    
    for cand in structured_candidates:
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Add difference set seeds
    diff_seeds = _generate_difference_set_seeds()
    for cand in diff_seeds:
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Dynamic Local Search with multiple restarts
    seeds = [best_set]
    # Add a few other good structured seeds for diversity
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    # Add some difference set seeds
    for cand in diff_seeds[:5]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        # Cycle through seeds and random starts
        if restarts % 3 == 0:
            base = seeds[restarts % len(seeds)]
        else:
            # Random start
            n = random.randint(4, 15)
            if n - 1 > max_val_limit:
                n = max_val_limit
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            base = [0] + elements
            base.sort()
        
        U = list(base)
        
        # Run local search
        local_score, local_set = _local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts
            if len(seeds) < 10:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set
    
    # Run genetic algorithm with diverse initial population
    if time.time() < deadline - 20:
        # Create diverse initial population
        ga_population = []
        
        # Add current best
        ga_population.append(list(best_set))
        
        # Add structured seeds
        for cand in structured_candidates[:5]:
            ga_population.append(list(cand))
        
        # Add difference set seeds
        for cand in diff_seeds[:5]:
            ga_population.append(list(cand))
        
        # Add random sets
        for _ in range(5):
            n = random.randint(4, 12)
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            ga_population.append(sorted([0] + elements))
        
        # Run genetic algorithm
        ga_score, ga_set = _genetic_algorithm(deadline, ga_population, pop_size=25, max_val_limit=max_val_limit, max_n_limit=max_n_limit)
        
        if ga_score > best_score:
            best_score = ga_score
            best_set = ga_set
    
    # Final Polish: High-precision local search on the best set
    # Use a tighter limit and more iterations if time permits
    if time.time() < deadline - 2:
        final_score, final_set = _local_search_dynamic(best_set, deadline, max_val_limit=2000, max_n_limit=50)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set

    return best_set
