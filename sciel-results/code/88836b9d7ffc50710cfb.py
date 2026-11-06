"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

# Precompute all 4-subsets of [1,16] as bitmasks (0-indexed bits for 0..15)
_ALL_MASKS = []
_MASK_TO_IDX = {}
for c in combinations(range(16), 4):
    mask = 0
    for i in c:
        mask |= (1 << i)
    _ALL_MASKS.append(mask)
    _MASK_TO_IDX[mask] = len(_ALL_MASKS) - 1

_N = len(_ALL_MASKS)  # 1820

# Precompute pairwise intersection masks
_PAIR_INTERSECT = np.zeros((_N, _N), dtype=np.int32)
for i in range(_N):
    mi = _ALL_MASKS[i]
    for j in range(_N):
        mj = _ALL_MASKS[j]
        _PAIR_INTERSECT[i, j] = mi & mj

def _is_sunflower(a, b, c):
    """Check if three 4-element sets (as bitmasks) form a sunflower."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(candidate, chosen_indices):
    """Check if candidate bitmask can be added to family of chosen_indices."""
    n = len(chosen_indices)
    for i in range(n):
        ai = _ALL_MASKS[chosen_indices[i]]
        for j in range(i + 1, n):
            aj = _ALL_MASKS[chosen_indices[j]]
            if _is_sunflower(ai, aj, candidate):
                return False
    return True

def _can_add_fast(candidate_idx, chosen_indices):
    """Faster check using precomputed intersections."""
    n = len(chosen_indices)
    cand = _ALL_MASKS[candidate_idx]
    for i in range(n):
        ai = _ALL_MASKS[chosen_indices[i]]
        ac = ai & cand
        for j in range(i + 1, n):
            aj = _ALL_MASKS[chosen_indices[j]]
            ab = ai & aj
            bc = aj & cand
            if ab == ac == bc:
                return False
    return True

def _greedy_build(order, deadline):
    """Build a family using greedy addition in the given order."""
    chosen = []
    for idx in order:
        if time.time() > deadline:
            break
        mask = _ALL_MASKS[idx]
        if _can_add(mask, chosen):
            chosen.append(idx)
    return chosen

def _try_add_any(family_indices, deadline):
    """Try to add any remaining subset to the family."""
    family_set = set(family_indices)
    for idx in range(_N):
        if time.time() > deadline:
            break
        if idx in family_set:
            continue
        if _can_add(_ALL_MASKS[idx], family_indices):
            return family_indices + [idx], idx
    return family_indices, None

def _local_improve(family_indices, deadline):
    """Try to improve the family by swapping out sets to allow more additions."""
    family_set = set(family_indices)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(family_indices)):
            if time.time() > deadline:
                break
            removed = family_indices[i]
            remaining = family_indices[:i] + family_indices[i+1:]
            best_candidate = None
            for idx in range(_N):
                if idx in family_set:
                    continue
                if _can_add(_ALL_MASKS[idx], remaining):
                    best_candidate = idx
                    break
            if best_candidate is not None:
                family_indices[i] = best_candidate
                family_set.remove(removed)
                family_set.add(best_candidate)
                improved = True
    return family_indices

def _simulated_annealing(initial_family, deadline, rng):
    """Improve family using simulated annealing with add/swap/remove moves."""
    family = list(initial_family)
    family_set = set(family)
    current_size = len(family)
    best_family = list(family)
    best_size = current_size
    
    t = 1.0
    cooling = 0.9995
    
    while time.time() < deadline and t > 0.005:
        t *= cooling
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a random set
            candidate = rng.randint(0, _N - 1)
            if candidate in family_set:
                continue
            mask = _ALL_MASKS[candidate]
            if _can_add(mask, family):
                family.append(candidate)
                family_set.add(candidate)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_family = list(family)
            else:
                # With some probability, remove a random set and try again
                if rng.random() < t * 0.1 and family:
                    rem_idx = rng.randint(0, len(family) - 1)
                    removed = family.pop(rem_idx)
                    family_set.discard(removed)
                    current_size -= 1
                    if _can_add(mask, family):
                        family.append(candidate)
                        family_set.add(candidate)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_family = list(family)
                    else:
                        # Put it back
                        family.append(removed)
                        family_set.add(removed)
                        current_size += 1
        
        elif move_type < 0.75:
            # Swap: remove one, try to add another
            if not family:
                continue
            rem_idx = rng.randint(0, len(family) - 1)
            removed = family[rem_idx]
            test_family = family[:rem_idx] + family[rem_idx+1:]
            # Try a few random candidates
            for _ in range(12):
                if time.time() > deadline:
                    break
                candidate = rng.randint(0, _N - 1)
                if candidate in family_set:
                    continue
                mask = _ALL_MASKS[candidate]
                if _can_add(mask, test_family):
                    family[rem_idx] = candidate
                    family_set.discard(removed)
                    family_set.add(candidate)
                    break
        
        else:
            # Remove a random set (to escape local optima)
            if family and rng.random() < t * 0.05:
                rem_idx = rng.randint(0, len(family) - 1)
                removed = family.pop(rem_idx)
                family_set.discard(removed)
                current_size -= 1
    
    return best_family

def _genetic_crossover(fam1, fam2, deadline):
    """
    Crossover: merge two valid families by taking their union and greedily
    building a maximal valid subfamily. This leverages the hereditary property:
    any subset of a sunflower-free family is also sunflower-free.
    """
    union_set = set(fam1) | set(fam2)
    # Greedily build from the union
    result = []
    for idx in union_set:
        if time.time() > deadline:
            break
        mask = _ALL_MASKS[idx]
        if _can_add(mask, result):
            result.append(idx)
    return result

def _genetic_algorithm(initial_families, deadline, rng):
    """
    Genetic algorithm with crossover and mutation.
    Population of families, each evolved via add/swap mutations and crossover.
    """
    pop_size = min(len(initial_families), 8)
    if pop_size == 0:
        return []
    
    population = [list(f) for f in initial_families[:pop_size]]
    # Pad population if needed
    while len(population) < pop_size:
        population.append(list(population[0]))
    
    best_family = max(population, key=len)
    best_size = len(best_family)
    
    while time.time() < deadline:
        # Selection: keep top half, replace bottom half with offspring
        population.sort(key=len, reverse=True)
        
        # Create offspring
        new_population = []
        for i in range(pop_size // 2):
            if time.time() > deadline:
                break
            # Crossover between two parents
            p1 = population[i]
            p2 = population[i + pop_size // 2] if i + pop_size // 2 < pop_size else population[0]
            
            child = _genetic_crossover(p1, p2, deadline)
            
            # Mutation: try to add random sets
            for _ in range(20):
                if time.time() > deadline:
                    break
                candidate = rng.randint(0, _N - 1)
                if candidate not in set(child):
                    mask = _ALL_MASKS[candidate]
                    if _can_add(mask, child):
                        child.append(candidate)
            
            # Mutation: try swaps
            for _ in range(10):
                if time.time() > deadline:
                    break
                if not child:
                    break
                rem_idx = rng.randint(0, len(child) - 1)
                removed = child[rem_idx]
                test_child = child[:rem_idx] + child[rem_idx+1:]
                for _ in range(5):
                    candidate = rng.randint(0, _N - 1)
                    if candidate not in set(child):
                        mask = _ALL_MASKS[candidate]
                        if _can_add(mask, test_child):
                            child[rem_idx] = candidate
                            break
            
            new_population.append(child)
        
        # Replace bottom half
        population = population[:pop_size // 2] + new_population
        
        # Track best
        current_best = max(population, key=len)
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_family = list(current_best)
    
    return best_family

def _branch_and_bound(deadline):
    """
    Solve maximum independent set in the sunflower hypergraph using branch-and-bound.
    """
    start = time.time()
    
    # First, get good initial solutions
    rng = random.Random(12345)
    
    # Multiple greedy restarts with different orderings
    initial_families = []
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        order = list(range(_N))
        rng.shuffle(order)
        fam = _greedy_build(order, deadline - 30)
        initial_families.append(fam)
    
    # Also try lexicographic
    fam_lex = _greedy_build(list(range(_N)), deadline - 30)
    initial_families.append(fam_lex)
    
    # Local improvement on best initial
    best_initial = max(initial_families, key=len)
    improved = _local_improve(list(best_initial), deadline - 20)
    if len(improved) > len(best_initial):
        best_initial = improved
    
    # Try to add more
    new_fam, _ = _try_add_any(list(best_initial), deadline - 20)
    if len(new_fam) > len(best_initial):
        best_initial = new_fam
    
    best_size = len(best_initial)
    best_solution = list(best_initial)
    
    # Genetic algorithm for broader exploration
    if time.time() < deadline - 10:
        ga_deadline = deadline - 5
        ga_family = _genetic_algorithm(initial_families, ga_deadline, random.Random(777))
        if len(ga_family) > best_size:
            best_size = len(ga_family)
            best_solution = ga_family
    
    # Simulated annealing for final polish
    if time.time() < deadline - 3:
        sa_family = _simulated_annealing(list(best_solution), deadline - 1, random.Random(999))
        if len(sa_family) > best_size:
            best_size = len(sa_family)
            best_solution = sa_family
    
    # Final: try to add any more
    if time.time() < deadline - 1:
        new_fam, _ = _try_add_any(list(best_solution), deadline - 0.5)
        if len(new_fam) > best_size:
            best_size = len(new_fam)
            best_solution = new_fam
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Use combined approach
    best_family = _branch_and_bound(deadline)
    
    # Convert back to sorted lists of 1-based integers
    result = []
    for idx in best_family:
        mask = _ALL_MASKS[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result

# EVOLVE-BLOCK-END
