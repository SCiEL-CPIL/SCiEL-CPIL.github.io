"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

# Precompute prime factors for all numbers 1..511
def _factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

_PRIME_FACTORS = {}
for _i in range(1, 512):
    _PRIME_FACTORS[_i] = _factorize(_i)

def _is_coprime_to_all_fast(a, chosen_set):
    """Check if a is coprime to all elements in chosen_set using precomputed factors."""
    fa = _PRIME_FACTORS[a]
    for b in chosen_set:
        if fa & _PRIME_FACTORS[b]:
            return False
    return True

def _score(chosen):
    return sum(1.0 / (512 - a) for a in chosen)

def _greedy_from_high():
    chosen = []
    chosen_set = set()
    for a in range(511, 0, -1):
        if _is_coprime_to_all_fast(a, chosen_set):
            chosen.append(a)
            chosen_set.add(a)
    return chosen

def _local_search(chosen, deadline):
    """Improve by trying removals and swaps with re-greedy fill."""
    best = list(chosen)
    best_set = set(best)
    best_score = _score(best)
    n = len(best)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        # Try removing each element and re-greeding
        for i in range(n):
            if time.time() > deadline:
                break
            remaining = [x for j, x in enumerate(best) if j != i]
            remaining_set = set(remaining)
            new_list = list(remaining)
            for a in range(511, 0, -1):
                if a in remaining_set:
                    continue
                if _is_coprime_to_all_fast(a, remaining_set):
                    new_list.append(a)
                    remaining_set.add(a)
            new_score = _score(new_list)
            if new_score > best_score + 1e-12:
                best = new_list
                best_set = set(best)
                best_score = new_score
                n = len(best)
                improved = True
                break  # restart outer while
    return best

def _branch_and_bound(chosen_set, deadline, initial_bound=None):
    """
    Branch-and-bound to find maximum weight independent set.
    Vertices: 1..511, edges: non-coprime pairs.
    Weight of vertex a: 1/(512-a)
    Uses fractional knapsack upper bound.
    """
    # Precompute weights
    weights = {a: 1.0 / (512 - a) for a in range(1, 512)}
    
    # Precompute conflict sets: for each a, which other numbers conflict
    conflict = {a: set() for a in range(1, 512)}
    for a in range(1, 512):
        for b in range(a+1, 512):
            if _PRIME_FACTORS[a] & _PRIME_FACTORS[b]:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Start with the greedy solution as initial bound
    if initial_bound is None:
        initial_bound = _score(list(chosen_set))
    
    best_solution = list(chosen_set)
    best_score = initial_bound
    
    # Candidates not in chosen_set, sorted by weight descending
    candidates = [a for a in range(511, 0, -1) if a not in chosen_set]
    
    # For branch and bound, we'll use a recursive approach with pruning
    # State: current chosen set, list of remaining candidates (sorted by weight)
    
    def _bnb_recursive(current_set, candidates_list, current_score, depth=0):
        nonlocal best_solution, best_score
        
        if time.time() > deadline:
            return
        
        if not candidates_list:
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_set)
            return
        
        # Upper bound: fractional knapsack on remaining candidates
        # Sort candidates by weight (already sorted)
        # Take them in order, but skip those that conflict with current_set
        # For the bound, we assume we can take all non-conflicting ones fractionally
        
        bound = current_score
        remaining_weight = 0.0
        for a in candidates_list:
            if a in current_set:
                continue
            if _is_coprime_to_all_fast(a, current_set):
                remaining_weight += weights[a]
            # For a tighter bound, we could also account for conflicts among candidates,
            # but that's expensive. Use a simple bound: sum of all non-conflicting weights.
        
        bound = current_score + remaining_weight
        if bound <= best_score + 1e-12:
            return  # Prune
        
        # Branch: try adding the highest-weight candidate that's coprime to current_set
        # Or skip it
        
        # Find the first candidate that's coprime to current_set
        added = False
        for i, a in enumerate(candidates_list):
            if a in current_set:
                continue
            if _is_coprime_to_all_fast(a, current_set):
                # Branch 1: include a
                new_set = current_set | {a}
                new_candidates = [x for x in candidates_list if x != a and x not in new_set]
                _bnb_recursive(new_set, new_candidates, current_score + weights[a], depth+1)
                
                # Branch 2: exclude a
                new_candidates2 = [x for x in candidates_list if x != a]
                _bnb_recursive(current_set, new_candidates2, current_score, depth+1)
                
                added = True
                break
        
        if not added:
            # No more candidates can be added
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_set)
    
    _bnb_recursive(chosen_set, candidates, initial_bound)
    
    return best_solution

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill
    
    # Initial greedy from high
    chosen = _greedy_from_high()
    
    # Local search
    chosen = _local_search(chosen, deadline)
    
    # Try branch-and-bound to escape local optima
    if time.time() < deadline:
        bnb_result = _branch_and_bound(set(chosen), deadline, _score(chosen))
        if _score(bnb_result) > _score(chosen):
            chosen = bnb_result
            # Polish with local search
            chosen = _local_search(chosen, deadline)
    
    # Final local search polish
    chosen = _local_search(chosen, deadline)
    
    return chosen
# EVOLVE-BLOCK-END
