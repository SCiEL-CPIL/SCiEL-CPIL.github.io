"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

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

# Precompute coprime masks using bitsets for speed
# Each number 1..511 gets a bitmask of which other numbers it's coprime to
# Actually, let's use a different approach: precompute for each number, the set of prime factors
# and use set intersection for fast coprime checking

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

def _greedy_with_lookahead(depth=2):
    """Greedy that looks ahead by depth to avoid blocking high-value future picks."""
    chosen = []
    chosen_set = set()
    candidates = list(range(511, 0, -1))
    
    while candidates:
        best_candidate = None
        best_score = -1
        for i, a in enumerate(candidates):
            if not _is_coprime_to_all_fast(a, chosen_set):
                continue
            # Try adding a and see what we can still get
            new_set = chosen_set | {a}
            temp_chosen = list(chosen) + [a]
            temp_set = set(new_set)
            for b in candidates[i+1:]:
                if _is_coprime_to_all_fast(b, temp_set):
                    temp_chosen.append(b)
                    temp_set.add(b)
            sc = _score(temp_chosen)
            if sc > best_score:
                best_score = sc
                best_candidate = a
        
        if best_candidate is None:
            break
        
        chosen.append(best_candidate)
        chosen_set.add(best_candidate)
        candidates = [a for a in candidates if a != best_candidate]
    
    return chosen

def _re_greedy_fill(remaining_set):
    """Fill with greedy from 511 down to 1, adding coprime numbers."""
    new_list = list(remaining_set)
    for a in range(511, 0, -1):
        if a in remaining_set:
            continue
        if _is_coprime_to_all_fast(a, remaining_set):
            new_list.append(a)
            remaining_set.add(a)
    return new_list

def _local_search_1swap(chosen, deadline):
    """Local search with 1-swap moves: remove one element, add one different compatible element, then re-greedy."""
    best = list(chosen)
    best_set = set(best)
    best_score = _score(best)
    n = len(best)
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        if no_improve_count > 5:
            break
        
        for i in range(n):
            if time.time() > deadline:
                break
            removed = best[i]
            remaining = [x for j, x in enumerate(best) if j != i]
            remaining_set = set(remaining)
            
            # Find all compatible candidates not in remaining_set
            candidates = [a for a in range(511, 0, -1) if a not in remaining_set and a != removed and _is_coprime_to_all_fast(a, remaining_set)]
            
            if not candidates:
                continue
            
            # Try each candidate (but limit to top 50 by weight for speed)
            candidates.sort(key=lambda a: 1.0/(512-a), reverse=True)
            candidates = candidates[:50]
            
            for a in candidates:
                if time.time() > deadline:
                    break
                new_set = remaining_set | {a}
                new_list = _re_greedy_fill(new_set)
                new_score = _score(new_list)
                if new_score > best_score + 1e-12:
                    best = new_list
                    best_set = set(best)
                    best_score = new_score
                    n = len(best)
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
    
    return best

def _local_search_2swap_fast(chosen, deadline):
    """Faster 2-swap: remove two elements, try adding pairs from top candidates, then re-greedy."""
    best = list(chosen)
    best_set = set(best)
    best_score = _score(best)
    n = len(best)
    improved = True
    no_improve_count = 0
    
    # Precompute top candidates by weight
    all_candidates = list(range(511, 0, -1))
    all_candidates.sort(key=lambda a: 1.0/(512-a), reverse=True)
    top_candidates = all_candidates[:80]
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        if no_improve_count > 3:
            break
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                removed1, removed2 = best[i], best[j]
                remaining = [x for k, x in enumerate(best) if k != i and k != j]
                remaining_set = set(remaining)
                
                # Try pairs from top candidates
                best_swap = None
                best_swap_score = best_score
                
                # Limit to top 30 candidates for each position
                for a in top_candidates[:30]:
                    if time.time() > deadline:
                        break
                    if a in remaining_set or a == removed1 or a == removed2:
                        continue
                    if not _is_coprime_to_all_fast(a, remaining_set):
                        continue
                    set1 = remaining_set | {a}
                    for b in top_candidates[:30]:
                        if b == a or b in set1 or b == removed1 or b == removed2:
                            continue
                        if not _is_coprime_to_all_fast(b, set1):
                            continue
                        new_set = set1 | {b}
                        new_list = _re_greedy_fill(new_set)
                        sc = _score(new_list)
                        if sc > best_swap_score + 1e-12:
                            best_swap_score = sc
                            best_swap = new_list
                    if best_swap is not None:
                        break
                    if time.time() > deadline:
                        break
                
                if best_swap is not None:
                    best = best_swap
                    best_set = set(best)
                    best_score = best_swap_score
                    n = len(best)
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
    
    return best

def _simulated_annealing(chosen, deadline):
    """Simulated annealing to escape local optima with faster move generation."""
    current = list(chosen)
    current_set = set(current)
    current_score = _score(current)
    best = list(current)
    best_set = set(best)
    best_score = current_score
    
    T = 0.005
    T_min = 1e-7
    cooling = 0.998
    max_steps = 50000
    
    step = 0
    while time.time() < deadline and T > T_min and step < max_steps:
        step += 1
        
        if len(current) < 2:
            break
        
        # Remove a random element
        idx = random.randint(0, len(current) - 1)
        removed = current[idx]
        new_set = set(current)
        new_set.remove(removed)
        
        # Find compatible candidates
        candidates = [a for a in range(1, 512) if a not in new_set and _is_coprime_to_all_fast(a, new_set)]
        if not candidates:
            T *= cooling
            continue
        
        # Pick weighted random candidate
        weights = [1.0 / (512 - a) for a in candidates]
        total_w = sum(weights)
        r = random.random() * total_w
        cum = 0
        chosen_a = candidates[0]
        for a, w in zip(candidates, weights):
            cum += w
            if r <= cum:
                chosen_a = a
                break
        
        new_list = [x for x in current if x != removed] + [chosen_a]
        new_score = _score(new_list)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current = new_list
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best = list(current)
                best_set = set(best)
                best_score = current_score
        
        T *= cooling
    
    return best

def _polish_pass(chosen, deadline):
    """Final high-precision polish: try all single swaps with full re-greedy."""
    best = list(chosen)
    best_set = set(best)
    best_score = _score(best)
    n = len(best)
    
    for i in range(n):
        if time.time() > deadline:
            break
        removed = best[i]
        remaining = [x for j, x in enumerate(best) if j != i]
        remaining_set = set(remaining)
        
        # Try ALL compatible candidates
        candidates = [a for a in range(511, 0, -1) if a not in remaining_set and _is_coprime_to_all_fast(a, remaining_set)]
        
        for a in candidates:
            if time.time() > deadline:
                break
            new_set = remaining_set | {a}
            new_list = _re_greedy_fill(new_set)
            new_score = _score(new_list)
            if new_score > best_score + 1e-12:
                best = new_list
                best_set = set(best)
                best_score = new_score
                n = len(best)
    
    return best

def construct():
    start = time.time()
    deadline = start + 135.0  # Safety margin before 150s kill
    
    # Generate multiple initial solutions
    solutions = []
    
    # Standard greedy
    sol1 = _greedy_from_high()
    solutions.append(sol1)
    
    # Greedy with lookahead
    sol2 = _greedy_with_lookahead()
    solutions.append(sol2)
    
    # Multi-start with random perturbations
    rng = random.Random(42)
    for restart in range(30):
        if time.time() > deadline - 30:
            break
        order = list(range(511, 0, -1))
        # Perturb: swap random pairs
        for _ in range(15):
            i, j = rng.randint(0, 510), rng.randint(0, 510)
            order[i], order[j] = order[j], order[i]
        chosen = []
        chosen_set = set()
        for a in order:
            if _is_coprime_to_all_fast(a, chosen_set):
                chosen.append(a)
                chosen_set.add(a)
        solutions.append(chosen)
    
    # Local search on each solution
    best_overall = None
    best_overall_score = -1
    
    for sol in solutions:
        if time.time() > deadline - 20:
            break
        
        # Apply 1-swap local search
        improved = _local_search_1swap(sol, deadline - 15)
        sc = _score(improved)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = improved
    
    # Apply 2-swap on the best solution
    if time.time() < deadline - 10 and best_overall is not None:
        improved = _local_search_2swap_fast(best_overall, deadline - 8)
        sc = _score(improved)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = improved
    
    # Simulated annealing on the best solution
    if time.time() < deadline - 5 and best_overall is not None:
        sa_result = _simulated_annealing(best_overall, deadline - 3)
        sa_score = _score(sa_result)
        if sa_score > best_overall_score:
            best_overall = sa_result
            best_overall_score = sa_score
    
    # Final polish pass
    if time.time() < deadline - 1 and best_overall is not None:
        polished = _polish_pass(best_overall, deadline)
        pol_score = _score(polished)
        if pol_score > best_overall_score:
            best_overall = polished
            best_overall_score = pol_score
    
    return best_overall
# EVOLVE-BLOCK-END
