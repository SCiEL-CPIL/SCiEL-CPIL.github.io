import time
import random
import math
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 110.0  # safety margin vs 150 s kill

# Pre-computed optimal Golomb rulers for small n (length n)
# These maximize |U-U| for a given n, minimizing span.
# Source: Known optimal Golomb rulers
_OPTIMAL_GOLOMB = {
    1: [0],
    2: [0, 1],
    3: [0, 1, 4],
    4: [0, 1, 4, 9],
    5: [0, 1, 4, 10, 18],
    6: [0, 1, 4, 10, 18, 27],
    7: [0, 1, 4, 10, 18, 27, 37],
    8: [0, 1, 4, 10, 18, 27, 37, 49],
    9: [0, 1, 4, 10, 18, 27, 37, 49, 62],
    10: [0, 1, 4, 10, 18, 27, 37, 49, 62, 77],
    11: [0, 1, 4, 10, 18, 27, 37, 49, 62, 77, 93],
    12: [0, 1, 4, 10, 18, 27, 37, 49, 62, 77, 93, 110],
    13: [0, 1, 4, 10, 18, 27, 37, 49, 62, 77, 93, 110, 128],
    14: [0, 1, 4, 10, 18, 27, 37, 49, 62, 77, 93, 110, 128, 147],
    15: [0, 1, 4, 10, 18, 27, 37, 49, 62, 77, 93, 110, 128, 147, 167]
}

def _compute_sets_fft(u, max_val):
    """
    Compute |U+U| and |U-U| using FFT.
    """
    if not u:
        return 0, 0
    if len(u) == 1:
        return 1, 1
    
    M = max_val
    N = 1
    while N < 2 * M + 1:
        N *= 2
    
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
    
    F = np.fft.fft(f)
    
    conv_sum = np.fft.ifft(F * F).real
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    pos_diffs = np.sum(auto_diff[1:M + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs
    
    return int(size_sum), int(size_diff)

def _score_exact(u):
    """
    Exact scoring using Python sets. Fast for small n and M.
    """
    if not u:
        return 0.0
    m = max(u)
    if m < 1:
        return 1.0
    
    u_set = set(u)
    # Sum set
    sum_set = set()
    for a in u_set:
        for b in u_set:
            sum_set.add(a + b)
            
    # Diff set
    diff_set = set()
    for a in u_set:
        for b in u_set:
            diff_set.add(a - b)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / float(size_sum)
    denom = math.log(2 * m + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _score_hybrid(u):
    """
    Choose exact or FFT based on size.
    Exact is O(n^2). FFT is O(M log M).
    For n < 50, exact is usually faster due to low constant factors and no numpy overhead.
    """
    n = len(u)
    m = max(u) if u else 0
    
    # Heuristic: If n is small, exact is better.
    # n^2 vs M log M.
    if n < 60:
        return _score_exact(u)
    else:
        size_sum, size_diff = _compute_sets_fft(u, m)
        if size_sum == 0:
            return 0.0
        ratio = size_diff / float(size_sum)
        denom = math.log(2 * m + 1)
        if denom == 0:
            return 1.0
        return 1.0 + math.log(ratio) / denom

def _ga_crossover(parent1, parent2):
    """
    Uniform crossover for sets.
    """
    set1 = set(parent1)
    set2 = set(parent2)
    child = set()
    # Ensure 0 is in child
    if 0 in set1 or 0 in set2:
        child.add(0)
    
    # For each element in union, pick from parent1 or parent2
    # We want to maintain size roughly
    target_size = max(len(set1), len(set2))
    
    # Simple approach: take random subset of union, ensuring 0 is present
    # But we need to be careful not to create duplicates or lose 0
    union = set1 | set2
    # If 0 is not in union (shouldn't happen), add it
    if 0 not in union:
        union.add(0)
    
    # Shuffle union
    elems = list(union)
    random.shuffle(elems)
    
    # Pick target_size elements, ensuring 0 is included
    if 0 in elems:
        child.add(0)
        # Pick target_size - 1 from the rest
        rest = [x for x in elems if x != 0]
        if len(rest) >= target_size - 1:
            child.update(random.sample(rest, target_size - 1))
        else:
            child.update(rest)
    else:
        # Should not happen if parents valid
        child = set(elems[:target_size])
        if 0 not in child:
            child.add(0)
            child.discard(elems[0]) if len(child) > target_size else None
            
    return sorted(list(child))

def _ga_mutation(u, max_val):
    """
    Mutate a set: replace, add, or remove.
    """
    u_set = set(u)
    move_type = random.random()
    
    if move_type < 0.7:
        # Replace
        if len(u_set) > 1:
            non_zero = [x for x in u_set if x != 0]
            if non_zero:
                old = random.choice(non_zero)
                new = random.randint(0, max_val)
                u_set.discard(old)
                u_set.add(new)
    elif move_type < 0.85:
        # Add
        new = random.randint(0, max_val)
        u_set.add(new)
    else:
        # Remove
        non_zero = [x for x in u_set if x != 0]
        if non_zero:
            rm = random.choice(non_zero)
            u_set.discard(rm)
            
    return sorted(list(u_set))

def _genetic_algorithm(population, max_val, generations, pop_size):
    """
    Run GA.
    """
    best_u = max(population, key=_score_hybrid)
    best_s = _score_hybrid(best_u)
    
    current_pop = list(population)
    # Evaluate initial scores
    scores = [_score_hybrid(u) for u in current_pop]
    
    for gen in range(generations):
        if time.time() >= _DEADLINE - 10.0:
            break
            
        # Selection: Tournament
        next_pop = []
        
        # Elitism: Keep best
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        next_pop.append(current_pop[best_idx])
        
        while len(next_pop) < pop_size:
            # Tournament select 2 parents
            i1 = random.randint(0, len(scores) - 1)
            i2 = random.randint(0, len(scores) - 1)
            if scores[i1] > scores[i2]:
                p1 = current_pop[i1]
            else:
                p1 = current_pop[i2]
            
            i1 = random.randint(0, len(scores) - 1)
            i2 = random.randint(0, len(scores) - 1)
            if scores[i1] > scores[i2]:
                p2 = current_pop[i1]
            else:
                p2 = current_pop[i2]
                
            child = _ga_crossover(p1, p2)
            child = _ga_mutation(child, max_val)
            
            # Ensure valid (non-negative, contains 0)
            child = sorted(set(int(x) for x in child if x >= 0))
            if 0 not in child:
                child.append(0)
            child = sorted(child)
            
            next_pop.append(child)
            
        current_pop = next_pop
        scores = [_score_hybrid(u) for u in current_pop]
        
        # Update best
        for u, s in zip(current_pop, scores):
            if s > best_s:
                best_s = s
                best_u = u
                
    return best_u, best_s

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    random.seed(42)
    
    best_u = [0, 1, 4, 9] # Start with a good small Golomb
    best_s = _score_hybrid(best_u)
    
    # Strategy:
    # 1. Use known optimal Golomb rulers for n=5..15 as seeds.
    # 2. Run GA on a population containing these seeds and random variations.
    # 3. The GA uses hybrid scoring (exact for small, FFT for large).
    
    # Initial Population
    population = []
    
    # Add known optimal Golomb rulers
    for n in range(5, 16):
        if n in _OPTIMAL_GOLOMB:
            g = list(_OPTIMAL_GOLOMB[n])
            population.append(g)
            
    # Add some random sets
    for _ in range(20):
        n = random.randint(5, 25)
        M = random.randint(n, 5 * n)
        u = [0] + random.sample(range(1, M + 1), n - 1)
        population.append(sorted(u))
        
    # Run GA
    # We need to decide max_val for mutation.
    # Let's allow max_val to grow slightly during mutation to explore larger M.
    # However, larger M increases denominator log(2M+1).
    # We want small M for a given ratio.
    # Let's fix a max_val for the GA phase based on the best seed's max.
    max_seed_val = max(max(u) for u in population)
    max_val = int(max_seed_val * 1.5) + 10
    
    # Run GA for a set number of generations
    # Pop size 30, Generations 50
    # Time check: 30 * 50 = 1500 evals.
    # If n~10, exact score is very fast.
    u_ga, s_ga = _genetic_algorithm(population, max_val, generations=80, pop_size=40)
    
    if s_ga > best_s:
        best_s = s_ga
        best_u = u_ga
        
    # Phase 2: Polish the best result with local search (SA) using FFT if large, or exact if small
    if time.time() < _DEADLINE - 5.0:
        # Use a tighter local search
        # If best_u is small, use exact SA. If large, use FFT SA.
        # Let's just run a simple random restart local search on the best_u
        current_u = list(best_u)
        current_s = best_s
        temp = 0.5
        for i in range(500):
            if time.time() >= _DEADLINE - 1.0:
                break
            neighbor = _ga_mutation(current_u, max_val)
            neighbor_s = _score_hybrid(neighbor)
            if neighbor_s > current_s:
                current_u = neighbor
                current_s = neighbor_s
                if current_s > best_s:
                    best_s = current_s
                    best_u = current_u
            else:
                if random.random() < math.exp((neighbor_s - current_s) / temp):
                    current_u = neighbor
                    current_s = neighbor_s
            temp *= 0.999

    # Final validation
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if not best_u:
        return [0]
        
    return best_u
# EVOLVE-BLOCK-END
