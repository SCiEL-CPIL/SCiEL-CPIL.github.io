"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from math import log
import numpy as np

# EVOLVE-BLOCK-START

def score_set(U):
    """Fast scoring for small sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    # For small sets, use direct computation which is faster than numpy overhead
    if n <= 30:
        sums_set = set()
        diffs_set = set()
        for i in range(n):
            for j in range(i, n):
                sums_set.add(U[i] + U[j])
            for j in range(i + 1, n):
                d = U[j] - U[i]
                diffs_set.add(d)
                diffs_set.add(-d)
        diffs_set.add(0)
        
        ratio = len(diffs_set) / len(sums_set)
        if ratio <= 1:
            return -1.0
        max_u = U[-1]
        D = 2 * max_u + 1
        if D <= 1:
            return -1.0
        return 1.0 + log(ratio) / log(D)
    else:
        arr = np.array(U, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n)
        sums = arr[i_idx] + arr[j_idx]
        unique_sums = np.unique(sums)
        i_idx2, j_idx2 = np.triu_indices(n, k=1)
        diffs = arr[j_idx2] - arr[i_idx2]
        all_diffs = np.concatenate([diffs, -diffs, np.array([0], dtype=np.int64)])
        unique_diffs = np.unique(all_diffs)
        ratio = len(unique_diffs) / len(unique_sums)
        if ratio <= 1:
            return -1.0
        max_u = U[-1]
        D = 2 * max_u + 1
        if D <= 1:
            return -1.0
        return 1.0 + log(ratio) / log(D)

def generate_difference_set_candidates(max_val=150):
    """Generate candidates inspired by difference sets and combinatorial designs."""
    candidates = []
    
    # Cyclic difference set inspired: for prime p, use quadratic residues mod p
    # This gives good difference coverage
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        # Quadratic residues mod p
        residues = sorted(set([i*i % p for i in range(1, p)]))
        # Scale to fit within max_val
        scale = max_val // (2 * p) if p > 0 else 1
        if scale < 1:
            scale = 1
        cand = [0] + [r * scale for r in residues]
        if len(cand) >= 3 and max(cand) <= max_val:
            candidates.append(sorted(set(cand)))
        
        # Also try with offset
        for offset in [1, 2]:
            cand = [0] + [(r + offset) * scale for r in residues if (r + offset) * scale <= max_val]
            if len(cand) >= 3 and max(cand) <= max_val:
                candidates.append(sorted(set(cand)))
    
    # Bose-Chowla type: {0} ∪ {a^i - 1 : i=1..n} for suitable a
    for a in [2, 3, 4, 5]:
        for n in range(3, 12):
            cand = [0]
            val = 1
            for i in range(n):
                val = val * a
                if val - 1 > max_val:
                    break
                cand.append(val - 1)
            if len(cand) >= 3 and max(cand) <= max_val:
                candidates.append(sorted(set(cand)))
    
    # Perfect difference set approximations: try to find sets where differences are well-distributed
    # Use greedy construction
    random.seed(42)
    for _ in range(100):
        n = random.randint(4, 12)
        max_v = random.randint(30, max_val)
        cand = [0]
        used_diffs = set()
        for _ in range(n - 1):
            best_val = None
            best_score = -1
            for val in range(1, max_v + 1):
                if val in cand:
                    continue
                # Check how many new differences this creates
                new_diffs = set()
                for u in cand:
                    d = abs(val - u)
                    if d not in used_diffs:
                        new_diffs.add(d)
                if len(new_diffs) > best_score:
                    best_score = len(new_diffs)
                    best_val = val
            if best_val is not None and best_val not in cand:
                cand.append(best_val)
                for u in cand[:-1]:
                    used_diffs.add(abs(best_val - u))
        if len(cand) >= 3 and max(cand) <= max_val:
            candidates.append(sorted(set(cand)))
    
    # Known good sets from the previous program
    known = [
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 8, 13],
        [0, 1, 4, 11, 13],
        [0, 1, 5, 10, 15],
        [0, 1, 6, 12, 18],
        [0, 1, 4, 7, 10],
        [0, 1, 5, 9, 13],
        [0, 1, 4, 8, 12],
        [0, 1, 3, 6, 9, 12],
        [0, 1, 4, 7, 10, 13],
        [0, 1, 5, 10, 15, 20],
        [0, 1, 4, 9, 14, 19],
        [0, 1, 3, 8, 13, 18],
        [0, 1, 4, 10, 16, 22],
        [0, 1, 5, 11, 17, 23],
        [0, 1, 4, 8, 12, 16],
        [0, 1, 6, 11, 16, 21],
        [0, 1, 4, 9, 14, 19, 24],
        [0, 1, 5, 10, 15, 20, 25],
        [0, 1, 3, 7, 11, 15, 19],
        [0, 1, 4, 8, 12, 16, 20],
        [0, 2, 5, 8, 11, 14],
        [0, 1, 4, 7, 10, 13, 16],
        [0, 1, 5, 9, 13, 17, 21],
        [0, 1, 3, 6, 9, 12, 15],
        [0, 1, 4, 8, 12, 16, 20, 24],
    ]
    candidates.extend(known)
    
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates

def local_search_improved(start_set, max_val, max_size, deadline, max_iterations=5000):
    """Improved local search with more aggressive neighborhood exploration."""
    current_set = sorted(set(start_set))
    current_score = score_set(current_set)
    
    if current_score <= 0:
        return current_set, current_score
    
    improved = True
    iteration = 0
    stall_count = 0
    
    while improved and time.time() < deadline - 2 and iteration < max_iterations:
        iteration += 1
        improved = False
        
        # Try moving each element
        for i in range(len(current_set)):
            if i == 0:
                continue
            old_val = current_set[i]
            # Try a wider range of moves
            for delta in range(-30, 31):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0 or new_val > max_val:
                    continue
                # Check for duplicates
                if any(current_set[j] == new_val for j in range(len(current_set)) if j != i):
                    continue
                new_set = sorted([current_set[j] if j != i else new_val for j in range(len(current_set))])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved and len(current_set) < max_size:
            # Sample candidate values rather than trying all
            candidates = list(set(random.sample(range(1, max_val + 1), min(50, max_val))))
            for new_val in candidates:
                if new_val not in current_set:
                    new_set = sorted(current_set + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
        
        # Try removing an element
        if not improved and len(current_set) > 3:
            for i in range(1, len(current_set)):
                new_set = [current_set[j] for j in range(len(current_set)) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Try swapping two elements
        if not improved and len(current_set) > 3:
            i, j = random.sample(range(1, len(current_set)), 2)
            new_set = sorted([current_set[j] if idx == i else (current_set[i] if idx == j else current_set[idx]) for idx in range(len(current_set))])
            if len(set(new_set)) == len(current_set):
                new_score = score_set(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
        
        if not improved:
            stall_count += 1
            if stall_count > 50:
                break
        else:
            stall_count = 0
    
    return current_set, current_score


def genetic_algorithm(deadline):
    """Genetic algorithm with improved local search and multi-start."""
    random.seed(12345)
    
    # Initialize population
    max_val = 200
    pop_size = 80
    max_size = 15
    
    # Start with difference-set-inspired candidates
    candidates = generate_difference_set_candidates(max_val)
    
    # Score all candidates and take the best ones
    scored = [(score_set(c), c) for c in candidates]
    scored = [(s, c) for s, c in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Take top candidates for initial population
    population = [c for _, c in scored[:min(pop_size, len(scored))]]
    
    # Fill remaining with random sets
    while len(population) < pop_size:
        n = random.randint(4, max_size)
        other = random.sample(range(1, max_val), min(n - 1, max_val - 1))
        population.append(sorted([0] + other))
    
    best_set = max(population, key=lambda s: score_set(s))
    best_score = score_set(best_set)
    
    # First, do local search on the best initial candidate
    if time.time() < deadline - 30:
        polished, polished_score = local_search_improved(best_set, max_val, max_size, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished
    
    # Genetic algorithm loop
    generation = 0
    start_time = time.time()
    
    while time.time() < deadline - 10:
        generation += 1
        
        # Evaluate population
        scored_pop = [(score_set(ind), ind) for ind in population]
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if scored_pop[0][0] > best_score:
            best_score = scored_pop[0][0]
            best_set = scored_pop[0][1][:]
        
        # Elitism: keep top individuals
        elite_count = max(8, pop_size // 4)
        new_population = [ind[:] for _, ind in scored_pop[:elite_count]]
        
        # Crossover and mutation
        while len(new_population) < pop_size and time.time() < deadline - 10:
            # Select parents using tournament selection
            k = min(5, len(population))
            candidates_sel = random.sample(population, k)
            parent1 = max(candidates_sel, key=lambda x: score_set(x))
            candidates_sel2 = random.sample(population, k)
            parent2 = max(candidates_sel2, key=lambda x: score_set(x))
            
            # Uniform crossover
            child = []
            p1_set = set(parent1)
            p2_set = set(parent2)
            all_vals = sorted(p1_set | p2_set)
            
            for v in all_vals:
                if v == 0:
                    child.append(0)
                elif random.random() < 0.5:
                    if v in p1_set:
                        child.append(v)
                else:
                    if v in p2_set:
                        child.append(v)
            
            if 0 not in child:
                child.append(0)
            child = sorted(set(child))
            
            # Mutation: try different types with higher probability of useful mutations
            mutation_type = random.random()
            
            if mutation_type < 0.35:
                # Point mutation: change one element with larger steps
                if len(child) > 1:
                    idx = random.randint(1, len(child) - 1)
                    old_val = child[idx]
                    delta = random.randint(-25, 25)
                    new_val = old_val + delta
                    if new_val > 0 and new_val <= max_val:
                        # Check for duplicates
                        if new_val not in [child[j] for j in range(len(child)) if j != idx]:
                            child[idx] = new_val
                            child = sorted(set(child))
            elif mutation_type < 0.55:
                # Insert mutation: add a new element
                if len(child) < max_size:
                    new_val = random.randint(1, max_val)
                    if new_val not in child:
                        child.append(new_val)
                        child = sorted(child)
            elif mutation_type < 0.70:
                # Delete mutation: remove an element
                if len(child) > 3:
                    idx = random.randint(1, len(child) - 1)
                    child.pop(idx)
            elif mutation_type < 0.85:
                # Swap mutation: swap two elements' values
                if len(child) > 3:
                    i, j = random.sample(range(1, len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                    child = sorted(set(child))
            else:
                # Gaussian mutation: perturb multiple elements
                for i in range(1, len(child)):
                    if random.random() < 0.4:
                        delta = int(random.gauss(0, 8))
                        new_val = child[i] + delta
                        if new_val > 0 and new_val <= max_val:
                            # Check for duplicates
                            if new_val not in [child[j] for j in range(len(child)) if j != i]:
                                child[i] = new_val
                child = sorted(set(child))
            
            # Ensure validity
            if len(child) >= 2 and 0 in child and max(child) <= max_val:
                new_population.append(child)
        
        population = new_population
        
        # Occasionally inject diverse individuals
        if generation % 3 == 0 and len(population) < pop_size:
            n = random.randint(4, max_size)
            other = random.sample(range(1, max_val), min(n - 1, max_val - 1))
            population.append(sorted([0] + other))
        
        # Every 5 generations, do local search on the best individual
        if generation % 5 == 0 and time.time() < deadline - 15:
            polished, polished_score = local_search_improved(population[0], max_val, max_size, deadline - 5, max_iterations=1000)
            if polished_score > best_score:
                best_score = polished_score
                best_set = polished
                # Replace worst individual with polished one
                population[-1] = polished[:]
    
    # Final local search polish on the best set with more iterations
    if best_score > 0 and time.time() < deadline - 3:
        polish_set, polish_score = local_search_improved(best_set, max_val, max_size, deadline, max_iterations=10000)
        
        if polish_score > best_score:
            best_score = polish_score
            best_set = polish_set[:]
    
    return best_set

def construct():
    # Use a safety margin of 115s to stay within the 120s limit
    deadline = time.time() + 115
    try:
        best_set = genetic_algorithm(deadline)
    except Exception:
        # Fallback to a known valid set if search fails
        best_set = [0, 1, 4, 9, 11]
    
    # Ensure the result is valid
    if not isinstance(best_set, list) or 0 not in best_set or len(best_set) < 2:
        best_set = [0, 1, 4, 9, 11]
    
    return best_set

# EVOLVE-BLOCK-END
