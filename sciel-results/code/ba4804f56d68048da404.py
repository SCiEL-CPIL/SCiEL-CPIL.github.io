import time
import numpy as np
import itertools
import random

def _greedy_cover(balls, cover_count, current_uncovered, deadline, margin=5.0):
    """Add codewords greedily to cover all uncovered positions."""
    selected = []
    while np.any(current_uncovered) and time.time() < deadline - margin:
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
    return selected, current_uncovered

def _prune_code(selected, balls, cover_count, deadline, margin=5.0):
    """Remove redundant codewords (those whose removal keeps all words covered)."""
    changed = True
    while changed and time.time() < deadline - margin:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    return selected

def _ga_improve(selected, balls, cover_count, deadline, margin=5.0):
    """
    Genetic algorithm to improve the code.
    Represents code as a set of word indices.
    Fitness: minimize size (penalty for uncovered words).
    """
    total = len(cover_count)
    best_selected = list(selected)
    best_cover = cover_count.copy()
    best_size = len(best_selected)
    best_uncovered = int(np.sum(best_cover == 0))
    
    rng = random.Random(123)
    population_size = 20
    max_generations = 50
    
    # Initialize population with best + some random perturbations
    population = [list(best_selected)]
    for _ in range(population_size - 1):
        # Perturb best
        pop = list(best_selected)
        k = rng.randint(3, min(10, len(pop) // 10))
        if len(pop) > k:
            to_remove = rng.sample(pop, k)
            to_remove_set = set(to_remove)
            pop = [c for c in pop if c not in to_remove_set]
            # Repair
            temp_cover = np.zeros(total, dtype=np.int32)
            for c in pop:
                temp_cover[balls[c]] += 1
            uncovered = (temp_cover == 0)
            _, uncovered = _greedy_cover(balls, temp_cover, uncovered, deadline, margin=margin+5)
            # Prune
            pop = _prune_code(pop, balls, temp_cover, deadline, margin=margin+5)
        population.append(pop)
    
    for gen in range(max_generations):
        if time.time() > deadline - margin:
            break
        
        # Evaluate fitness
        def fitness(code):
            if not code:
                return (10000, 10000) # Size, uncovered
            cover = np.zeros(total, dtype=np.int32)
            for c in code:
                cover[balls[c]] += 1
            unc = int(np.sum(cover == 0))
            # Penalty: if uncovered, size is effectively worse
            if unc > 0:
                return (len(code) + unc * 100, unc)
            return (len(code), 0)
        
        scored_pop = []
        for code in population:
            f, unc = fitness(code)
            scored_pop.append((f, unc, code))
        
        scored_pop.sort(key=lambda x: (x[0], x[1]))
        
        # Update best
        best_f, best_unc, best_code = scored_pop[0]
        if best_f < best_size or (best_f == best_size and best_unc < best_uncovered):
            best_size = best_f
            best_uncovered = best_unc
            best_selected = list(best_code)
            # Recompute cover for best
            best_cover = np.zeros(total, dtype=np.int32)
            for c in best_selected:
                best_cover[balls[c]] += 1
        
        # Selection: Tournament
        def select():
            idxs = rng.sample(range(len(scored_pop)), min(3, len(scored_pop)))
            best_idx = min(idxs, key=lambda i: scored_pop[i][0])
            return scored_pop[best_idx][2]
        
        # Crossover and Mutation
        new_population = []
        # Elitism: keep top 2
        for i in range(min(2, len(scored_pop))):
            new_population.append(list(scored_pop[i][2]))
        
        while len(new_population) < population_size:
            if time.time() > deadline - margin:
                break
            p1 = select()
            p2 = select()
            
            # Uniform crossover
            child = []
            set1 = set(p1)
            set2 = set(p2)
            union = set1 | set2
            for c in union:
                if rng.random() < 0.5:
                    child.append(c)
            
            # Mutation: random add/remove
            # Remove random
            if child and rng.random() < 0.5:
                rm = rng.choice(child)
                child.remove(rm)
            # Add random (from all words, weighted by frequency in parents?)
            if rng.random() < 0.5:
                # Pick from union of parents or random word
                if rng.random() < 0.5 and union:
                    add = rng.choice(list(union))
                else:
                    add = rng.randint(0, total - 1)
                if add not in child:
                    child.append(add)
            
            # Repair and Prune
            temp_cover = np.zeros(total, dtype=np.int32)
            for c in child:
                temp_cover[balls[c]] += 1
            uncovered = (temp_cover == 0)
            _, uncovered = _greedy_cover(balls, temp_cover, uncovered, deadline, margin=margin+10)
            if not uncovered.any():
                child = _prune_code(child, balls, temp_cover, deadline, margin=margin+10)
            
            new_population.append(child)
        
        population = new_population
    
    return best_selected, best_cover

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline, start_time)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _construct_ternary_8_1_optimized(deadline, start_time):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Weights for positions (big-endian)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Precompute a mapping from word index to its position in the balls array
    # For quick lookup: given a word, which codewords cover it?
    # We'll maintain cover_count array dynamically.
    
    # 1. Greedy Construction
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # Main greedy loop
    while np.any(current_uncovered):
        if time.time() > deadline - 40.0:
            break
            
        # Count uncovered neighbors for each word
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        
        if max_count == 0:
            break
        
        # Deterministic: pick smallest index among max-count candidates
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    # Force completion if needed
    if np.any(current_uncovered) and time.time() < deadline - 10.0:
        while np.any(current_uncovered):
            uncovered_indices = np.where(current_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            best_idx = uncovered_indices[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            if time.time() > deadline - 10.0:
                break
    
    # 2. Aggressive Pruning (Greedy Removal)
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    # Try reverse order pass
    if time.time() < deadline - 20.0:
        changed = True
        while changed and time.time() < deadline - 20.0:
            changed = False
            new_selected = []
            for c in reversed(selected):
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = list(reversed(new_selected))
    
    # 3. Genetic Algorithm Improvement
    if time.time() < deadline - 15.0:
        selected, cover_count = _ga_improve(selected, balls, cover_count, deadline, margin=15.0)
    
    # 4. Final Pruning Pass (in case GA didn't prune fully)
    selected = _prune_code(selected, balls, cover_count, deadline, margin=5.0)
    
    # 5. Final Verification and Conversion
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1

    # Convert to digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
