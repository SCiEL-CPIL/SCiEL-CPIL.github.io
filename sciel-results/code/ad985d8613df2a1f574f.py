"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 256

# Precompute all 4-APs and 3-APs
# A 4-AP is (a, a+d, a+2d, a+3d) with a >= 1, a+3d <= 256
# A 3-AP is (a, a+d, a+2d) with a >= 1, a+2d <= 256, d >= 1

def _precompute():
    """Precompute all 4-APs and 3-APs, and build conflict structures."""
    four_aps = []  # list of tuples (a, a+d, a+2d, a+3d)
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    three_aps = []  # list of tuples (a, a+d, a+2d)
    for d in range(1, (N // 2) + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    # For each element, which 4-APs does it belong to?
    elem_to_four_aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(four_aps):
        for x in ap:
            elem_to_four_aps[x].append(idx)
    
    # For each element, which 3-APs does it belong to?
    elem_to_three_aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(three_aps):
        for x in ap:
            elem_to_three_aps[x].append(idx)
    
    # For each 4-AP, the set of elements
    four_ap_elems = [set(ap) for ap in four_aps]
    
    # For each 3-AP, the set of elements
    three_ap_elems = [set(ap) for ap in three_aps]
    
    return four_aps, three_aps, elem_to_four_aps, elem_to_three_aps, four_ap_elems, three_ap_elems

# Global precomputation
_FOUR_APS, _THREE_APS, _ELEM_TO_FOUR, _ELEM_TO_THREE, _FOUR_AP_ELEMS, _THREE_AP_ELEMS = _precompute()

def _count_three_aps(selected_set):
    """Count the number of distinct 3-APs fully contained in selected_set."""
    count = 0
    for ap in _THREE_AP_ELEMS:
        if ap.issubset(selected_set):
            count += 1
    return count

def _has_four_ap(selected_set):
    """Check if selected_set contains any 4-AP."""
    for ap in _FOUR_AP_ELEMS:
        if ap.issubset(selected_set):
            return True
    return False

def _greedy_construct():
    """Greedy construction: add elements that maximize 3-APs while avoiding 4-APs."""
    selected = set()
    # Try to add each element in order, checking 4-AP constraint
    for x in range(1, N + 1):
        candidate = selected | {x}
        if not _has_four_ap(candidate):
            selected.add(x)
    return selected

def _local_search(selected_set, time_limit=30.0):
    """Local search: try adding, removing, or swapping elements to improve 3-AP count."""
    selected = set(selected_set)
    best_score = _count_three_aps(selected)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try adding elements
        for x in range(1, N + 1):
            if x not in selected:
                candidate = selected | {x}
                if not _has_four_ap(candidate):
                    new_score = _count_three_aps(candidate)
                    if new_score > best_score:
                        selected = candidate
                        best_score = new_score
                        improved = True
        
        # Try removing elements that are in few 3-APs but block many 4-APs
        # Actually, removing reduces 3-APs, so only do this if it enables adding more
        # Try swaps: remove one, add one or two
        for x in list(selected):
            candidate = selected - {x}
            for y in range(1, N + 1):
                if y not in candidate:
                    new_candidate = candidate | {y}
                    if not _has_four_ap(new_candidate):
                        new_score = _count_three_aps(new_candidate)
                        if new_score > best_score:
                            selected = new_candidate
                            best_score = new_score
                            improved = True
                            break
        
        if not improved:
            break
    
    return selected

def _genetic_algorithm(time_limit=20.0):
    """Genetic algorithm to find good 4-AP-free sets."""
    # Start with a greedy solution
    initial = _greedy_construct()
    population = [set(initial)]
    
    # Add some random perturbations
    for _ in range(19):
        perturbed = set(initial)
        # Remove some random elements
        for _ in range(random.randint(5, 15)):
            if perturbed:
                perturbed.discard(random.choice(list(perturbed)))
        # Add some random elements
        for _ in range(random.randint(5, 15)):
            x = random.randint(1, N)
            candidate = perturbed | {x}
            if not _has_four_ap(candidate):
                perturbed.add(x)
        population.append(perturbed)
    
    # Evaluate all
    scores = [_count_three_aps(ind) for ind in population]
    best_idx = scores.index(max(scores))
    best_score = scores[best_idx]
    
    start_time = time.time()
    gen = 0
    
    while time.time() - start_time < time_limit:
        gen += 1
        # Sort by score
        population.sort(key=lambda x: _count_three_aps(x), reverse=True)
        scores = [_count_three_aps(ind) for ind in population]
        
        # Keep top half
        half = len(population) // 2
        parents = population[:half]
        
        # Generate children
        children = []
        while len(children) < half:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            # Crossover: take union, then remove elements that cause 4-APs
            child = set(p1)
            # Add some from p2
            for x in p2:
                if x not in child:
                    candidate = child | {x}
                    if not _has_four_ap(candidate):
                        child.add(x)
            
            # Mutation
            for _ in range(random.randint(1, 5)):
                action = random.choice(['add', 'remove'])
                if action == 'add':
                    x = random.randint(1, N)
                    candidate = child | {x}
                    if not _has_four_ap(candidate):
                        child.add(x)
                else:
                    if child:
                        child.discard(random.choice(list(child)))
            
            children.append(child)
        
        population = parents + children
    
    # Return best
    population.sort(key=lambda x: _count_three_aps(x), reverse=True)
    return population[0]

def construct():
    """Construct a subset of [1,256] with no 4-AP, maximizing 3-APs."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Phase 1: Greedy construction
    greedy_result = _greedy_construct()
    greedy_score = _count_three_aps(greedy_result)
    
    # Phase 2: Local search on greedy result
    local_result = _local_search(greedy_result, time_limit=20.0)
    local_score = _count_three_aps(local_result)
    
    best = local_result if local_score >= greedy_score else greedy_result
    best_score = max(local_score, greedy_score)
    
    # Phase 3: Genetic algorithm if time permits
    if time.time() < deadline - 10.0:
        ga_result = _genetic_algorithm(time_limit=25.0)
        ga_score = _count_three_aps(ga_result)
        if ga_score > best_score:
            best = ga_result
            best_score = ga_score
    
    # Phase 4: Final local search on best
    if time.time() < deadline - 5.0:
        final_result = _local_search(best, time_limit=15.0)
        final_score = _count_three_aps(final_result)
        if final_score > best_score:
            best = final_result
            best_score = final_score
    
    return sorted(best)

# EVOLVE-BLOCK-END
