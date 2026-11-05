"""Rewritten program: genetic algorithm + local search for 4-AP-free subsets maximizing 3-APs."""
import time
import random
import numpy as np

N = 256
DEADLINE_FRACTION = 0.85  # safety margin for 90s wall-clock

# Precompute 4-AP masks: for each position i, mask of positions that would form a 4-AP with i
# A 4-AP is a, a+d, a+2d, a+3d with d>0.
# For each element x in [1..256], we need to check if adding x creates a 4-AP.
# Precompute for each x, the set of "forbidden" pairs or the set of existing elements that would complete a 4-AP.

# Better approach: precompute all 4-APs in [1,256].
# A 4-AP is (a, a+d, a+2d, a+3d) where a >= 1, a+3d <= 256, d >= 1.
# For a subset S to be 4-AP-free, no 4-AP can be fully contained in S.

# Precompute list of all 4-APs as tuples of 4 indices (0-based: a-1, a+d-1, a+2d-1, a+3d-1)
def _precompute_4aps():
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    return aps

# Precompute 3-APs: (a, a+d, a+2d) with d > 0
def _precompute_3aps():
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 2
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1))
    return aps

# For fast 4-AP checking: for each element i, precompute the list of 4-APs that contain i
# and for each 4-AP, store the other 3 elements. When checking if adding i to S creates a 4-AP,
# we check if any 4-AP containing i has all other 3 elements in S.

def _build_4ap_index():
    """Returns a dict: element_index -> list of (other3_indices, ap_index)"""
    all_4aps = _precompute_4aps()
    index = {i: [] for i in range(N)}
    for ap_idx, ap in enumerate(all_4aps):
        for pos in range(4):
            elem = ap[pos]
            others = tuple(ap[j] for j in range(4) if j != pos)
            index[elem].append(others)
    return index

def _build_3ap_count_index():
    """Returns for each element i, the list of 3-APs (as frozensets or tuples) containing i.
    For counting 3-APs in a subset, we can use: count = sum over 3-APs of [all 3 in S]
    But that's O(#3APs) per evaluation. Better: precompute for each pair (i,j) the third element.
    Actually, let's precompute: for each element i, the list of (j, k) such that {i,j,k} forms a 3-AP.
    Then count_3aps(S) = (1/3) * sum_{i in S} |{j in S : exists k in S, {i,j,k} is 3-AP}|... 
    Hmm, each 3-AP has 3 elements, so if we count for each i in S the number of pairs (j,k) in S completing a 3-AP with i, we count each 3-AP 3 times.
    So count = (1/3) * sum_{i in S} (number of 3-APs containing i that are fully in S).
    
    Alternatively, precompute for each i, a bitmask of elements j such that there exists k making a 3-AP.
    But that's not quite right either.
    
    Let's just precompute all 3-APs and for counting, iterate over them. #3APs is about N^2/4 ~ 16384, which is manageable.
    """
    return _precompute_3aps()

# For faster fitness: precompute for each element i, a list of bitmasks (as Python ints) representing 
# the other two elements of each 3-AP containing i. Then for a bitmask S, the number of 3-APs 
# fully in S can be computed as:
# For each i in S, count how many 3-APs containing i have both other elements in S.
# Each 3-AP is counted 3 times (once per element), so divide by 3.

def _build_3ap_pair_index():
    """For each element i, list of (j, k) where {i,j,k} is a 3-AP, j<k, j!=i, k!=i."""
    all_3aps = _precompute_3aps()
    index = {i: [] for i in range(N)}
    for a, b, c in all_3aps:
        index[a].append((b, c))
        index[b].append((a, c))
        index[c].append((a, b))
    return index

# Precomputation done once
_4AP_INDEX = _build_4ap_index()
_3AP_PAIR_INDEX = _build_3ap_pair_index()
_ALL_3APS = _precompute_3aps()

def is_4ap_free(mask):
    """Check if bitmask (int) represents a 4-AP-free subset."""
    # For each element in mask, check if any 4-AP containing it is fully in mask
    for i in range(N):
        if not (mask >> i) & 1:
            continue
        for others in _4AP_INDEX[i]:
            # Check if all others are in mask
            ok = True
            for j in others:
                if not (mask >> j) & 1:
                    ok = False
                    break
            if ok:
                return False
    return True

def count_3aps(mask):
    """Count distinct 3-APs fully contained in the subset represented by mask."""
    count = 0
    for a, b, c in _ALL_3APS:
        if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
            count += 1
    return count

def greedy_construct():
    """Greedy: add elements in order if they don't create a 4-AP. Then try to add more."""
    mask = 0
    for x in range(N):
        # Check if adding x creates a 4-AP
        new_mask = mask | (1 << x)
        # Check only 4-APs containing x
        valid = True
        for others in _4AP_INDEX[x]:
            all_in = True
            for j in others:
                if not (mask >> j) & 1:
                    all_in = False
                    break
            if all_in:
                valid = False
                break
        if valid:
            mask = new_mask
    return mask

def local_search(mask, max_iter=2000):
    """Try to add elements or swap to improve 3-AP count while maintaining 4-AP-free."""
    best_mask = mask
    best_score = count_3aps(mask)
    
    for _ in range(max_iter):
        improved = False
        # Try adding each element not in mask
        for x in range(N):
            if (mask >> x) & 1:
                continue
            new_mask = mask | (1 << x)
            # Check 4-AP-free
            valid = True
            for others in _4AP_INDEX[x]:
                all_in = True
                for j in others:
                    if not (mask >> j) & 1:
                        all_in = False
                        break
                if all_in:
                    valid = False
                    break
            if not valid:
                continue
            score = count_3aps(new_mask)
            if score > best_score:
                best_mask = new_mask
                best_score = score
                mask = new_mask
                improved = True
                break
        if not improved:
            break
    
    return best_mask, best_score

def genetic_algorithm(time_budget):
    """GA with bitmask representation."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initialize population
    pop_size = 50
    population = []
    
    # Seed with greedy and some random valid sets
    greedy_mask = greedy_construct()
    population.append(greedy_mask)
    
    # Add random valid sets
    random.seed(42)
    while len(population) < pop_size:
        # Random subset, then remove 4-APs
        mask = 0
        for x in range(N):
            if random.random() < 0.4:
                mask |= (1 << x)
        # Remove elements to make 4-AP-free
        for x in range(N):
            if (mask >> x) & 1:
                valid = True
                for others in _4AP_INDEX[x]:
                    all_in = True
                    for j in others:
                        if not (mask >> j) & 1:
                            all_in = False
                            break
                    if all_in:
                        valid = False
                        break
                if not valid:
                    mask &= ~(1 << x)
        if mask != 0:
            population.append(mask)
    
    # Evaluate initial population
    scores = [count_3aps(m) for m in population]
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_mask = population[best_idx]
    best_score = scores[best_idx]
    
    # Local search on best
    best_mask, best_score = local_search(best_mask, max_iter=500)
    
    while time.time() < deadline:
        # Selection: tournament
        def select():
            candidates = random.sample(range(pop_size), min(3, pop_size))
            return population[max(candidates, key=lambda i: scores[i])]
        
        # Crossover: uniform
        def crossover(p1, p2):
            child = 0
            for i in range(N):
                if random.random() < 0.5:
                    if (p1 >> i) & 1:
                        child |= (1 << i)
                else:
                    if (p2 >> i) & 1:
                        child |= (1 << i)
            return child
        
        # Mutation: flip random bits
        def mutate(mask, rate=0.05):
            m = mask
            for i in range(N):
                if random.random() < rate:
                    m ^= (1 << i)
            return m
        
        # Create new generation
        new_population = []
        new_scores = []
        
        # Keep best (elitism)
        new_population.append(best_mask)
        new_scores.append(best_score)
        
        while len(new_population) < pop_size:
            if time.time() >= deadline:
                break
            p1 = select()
            p2 = select()
            child = crossover(p1, p2)
            child = mutate(child)
            
            # Make 4-AP-free
            for x in range(N):
                if (child >> x) & 1:
                    valid = True
                    for others in _4AP_INDEX[x]:
                        all_in = True
                        for j in others:
                            if not (child >> j) & 1:
                                all_in = False
                                break
                        if all_in:
                            valid = False
                            break
                    if not valid:
                        child &= ~(1 << x)
            
            if child == 0:
                continue
            
            score = count_3aps(child)
            new_population.append(child)
            new_scores.append(score)
        
        # Update population
        population = new_population
        scores = new_scores
        
        # Update best
        for i in range(len(population)):
            if scores[i] > best_score:
                best_mask = population[i]
                best_score = scores[i]
        
        # Occasional local search on best
        if random.random() < 0.3:
            lm, ls = local_search(best_mask, max_iter=200)
            if ls > best_score:
                best_mask = lm
                best_score = ls
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    total_budget = 75.0  # 90s * 0.85 safety margin
    
    # Phase 1: Greedy + local search
    mask = greedy_construct()
    mask, score = local_search(mask, max_iter=1000)
    best_mask = mask
    best_score = score
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    # Phase 2: Genetic algorithm if time permits
    if remaining > 5.0:
        ga_mask, ga_score = genetic_algorithm(remaining - 2.0)
        if ga_score > best_score:
            best_mask = ga_mask
            best_score = ga_score
    
    # Final local search
    elapsed = time.time() - start_time
    if time.time() - start_time < total_budget - 1.0:
        best_mask, best_score = local_search(best_mask, max_iter=500)
    
    # Convert mask to sorted list
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return result

# Precompute at module load time to avoid repeated work
_4AP_INDEX = _build_4ap_index()
_3AP_PAIR_INDEX = _build_3ap_pair_index()
_ALL_3APS = _precompute_3aps()
