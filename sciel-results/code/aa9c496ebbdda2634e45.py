"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflicts():
    """
    For each element x, precompute the set of pairs (y,z) such that {x,y,z} is a bad triple.
    conflicts[x] = set of frozensets {y,z} where {x,y,z} forms a bad triple.
    """
    conflicts = {x: set() for x in range(1, 129)}
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))
    return conflicts

def _can_add(x, selected_set, conflicts):
    """Check if x can be added to selected_set without creating a bad triple."""
    for pair in conflicts[x]:
        if pair.issubset(selected_set):
            return False
    return True

def _greedy_smart(conflicts, order=None):
    """Greedy with conflict-degree-based ordering."""
    if order is None:
        # Sort by conflict degree (fewer conflicts first)
        order = sorted(range(1, 129), key=lambda x: len(conflicts[x]))
    
    selected = []
    selected_set = set()
    for x in order:
        if _can_add(x, selected_set, conflicts):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, conflicts, time_limit=10):
    """Local search: try removing one element and adding another."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        for rem in list(selected_set):
            candidate = selected_set - {rem}
            for x in range(1, 129):
                if x in candidate:
                    continue
                if _can_add(x, candidate, conflicts):
                    new_set = candidate | {x}
                    if len(new_set) > best_size:
                        best = list(new_set)
                        best_size = len(best)
                        selected_set = new_set
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def _simulated_annealing(selected, conflicts, time_limit=15):
    """Simulated annealing to escape local optima."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    current = list(selected)
    current_set = set(selected)
    
    start = time.time()
    temperature = 5.0
    cooling = 0.95
    
    while time.time() - start < time_limit and temperature > 0.1:
        # Pick a random element to remove
        if not current:
            break
        rem = random.choice(current)
        candidate_set = current_set - {rem}
        
        # Try to add a random element
        available = [x for x in range(1, 129) if x not in candidate_set]
        if not available:
            temperature *= cooling
            continue
        add = random.choice(available)
        
        if not _can_add(add, candidate_set, conflicts):
            temperature *= cooling
            continue
        
        new_set = candidate_set | {add}
        new_size = len(new_set)
        
        # Accept if better, or with probability based on temperature
        if new_size >= best_size or random.random() < (1.0 / (temperature + 0.1)):
            current = list(new_set)
            current_set = new_set
            if new_size > best_size:
                best = list(new_set)
                best_size = new_size
        
        temperature *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    conflicts = _build_conflicts()
    
    best = []
    best_size = 0
    
    # Initial greedy with smart ordering
    sel = _greedy_smart(conflicts)
    best = sel
    best_size = len(sel)
    
    # Local search
    if time.time() < deadline - 10:
        improved = _local_search(best, conflicts, time_limit=15)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Multiple restarts with different orderings
    random.seed(42)
    for restart in range(80):
        if time.time() > deadline - 5:
            break
        
        # Different ordering strategies
        if restart % 3 == 0:
            order = list(range(1, 129))
            random.shuffle(order)
        elif restart % 3 == 1:
            # Sort by random priority
            order = sorted(range(1, 129), key=lambda x: random.random())
        else:
            # Sort by conflict degree with random tie-breaking
            order = sorted(range(1, 129), key=lambda x: (len(conflicts[x]), random.random()))
        
        sel = _greedy_smart(conflicts, order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        
        # Local search on promising solutions
        if len(sel) >= best_size - 1 and time.time() < deadline - 15:
            improved = _local_search(sel, conflicts, time_limit=5)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
        
        # Simulated annealing on best solution occasionally
        if restart % 10 == 5 and time.time() < deadline - 20:
            improved = _simulated_annealing(best, conflicts, time_limit=8)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Final local search
    if time.time() < deadline - 3 and len(best) > 0:
        improved = _local_search(best, conflicts, time_limit=20)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if not best:
        return [1]
    
    return best
# EVOLVE-BLOCK-END
