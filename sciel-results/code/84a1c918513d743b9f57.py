"""Optimized approach: Prime-factor-based conflict detection + First-Improvement local search + SA."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt, log
from fractions import Fraction

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    if n < 25:
        return True
    if n % 5 == 0:
        return False
    if n < 49:
        return True
    if n % 7 == 0:
        return False
    if n < 121:
        return True
    i = 11
    while i <= isqrt(n):
        if n % i == 0:
            return False
        i += 2
    return True

def _get_prime_factors(n):
    """Return list of unique prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    N = 512
    
    # Precompute values and prime factors for all numbers 1..511
    values = [Fraction(0)] * N
    prime_factors = [[] for _ in range(N)]
    is_prime_arr = [False] * N
    
    for a in range(1, N):
        values[a] = Fraction(1, N - a)
        prime_factors[a] = _get_prime_factors(a)
        is_prime_arr[a] = _is_prime(a)
        
    # Identify primes
    primes = [a for a in range(1, N) if is_prime_arr[a]]
    
    # Initialize with all primes
    selected = set(primes)
    
    # Build mapping: prime -> set of selected elements that have this prime as a factor
    prime_to_selected = {}
    for b in selected:
        for p in prime_factors[b]:
            if p not in prime_to_selected:
                prime_to_selected[p] = set()
            prime_to_selected[p].add(b)
    
    def get_conflicts(candidates):
        """Get all selected elements that conflict with any candidate."""
        conflict_set = set()
        for c in candidates:
            for p in prime_factors[c]:
                if p in prime_to_selected:
                    conflict_set.update(prime_to_selected[p])
        return conflict_set

    def apply_move(add_candidates, remove_elements):
        """Apply a move: remove elements, add candidates. Update prime_to_selected."""
        for b in remove_elements:
            for p in prime_factors[b]:
                if p in prime_to_selected:
                    prime_to_selected[p].discard(b)
                    if not prime_to_selected[p]:
                        del prime_to_selected[p]
        for c in add_candidates:
            for p in prime_factors[c]:
                if p not in prime_to_selected:
                    prime_to_selected[p] = set()
                prime_to_selected[p].add(c)
        selected.difference_update(remove_elements)
        selected.update(add_candidates)

    def get_score():
        return sum(values[a] for a in selected)
    
    current_score = get_score()
    
    # Phase 1: 1-for-k First-Improvement Search
    # We scan candidates and apply the first move that improves the score.
    # This allows for more iterations than Best-Improvement.
    improved = True
    while improved:
        if time.time() > deadline - 30:
            break
            
        improved = False
        # Shuffle candidates to avoid bias? Or just iterate 1..N?
        # Iterating 1..N is deterministic. Let's try 1..N first.
        for c in range(1, N):
            if c in selected:
                continue
            
            # Find conflicts using prime factor mapping
            conflict_set = set()
            for p in prime_factors[c]:
                if p in prime_to_selected:
                    conflict_set.update(prime_to_selected[p])
            
            conflict_cost = sum(values[b] for b in conflict_set)
            gain = values[c] - conflict_cost
            
            if gain > 0:
                apply_move([c], list(conflict_set))
                current_score += gain
                improved = True
                break # Restart scan
    
    # Phase 2: 2-for-k Search with Conflict-Aware Candidate Selection
    if time.time() < deadline - 20:
        # Candidates are composites not in selected
        # We prioritize candidates with low conflict count
        candidates = []
        for a in range(1, N):
            if a in selected or a == 1:
                continue
            # Quick conflict check
            conf_count = 0
            for p in prime_factors[a]:
                if p in prime_to_selected:
                    conf_count += 1
                    # Early exit if too many conflicts?
                    if conf_count > 2:
                        break
            if conf_count <= 2:
                candidates.append(a)
                
        # Sort by value descending
        candidates.sort(key=lambda x: values[x], reverse=True)
        top_composites = candidates[:100]
        
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
                
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            # Try pairs from top_composites
            # Optimization: Only check pairs that are coprime to each other
            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    
                    # Find conflicts
                    conflict_set = set()
                    for p in prime_factors[c1]:
                        if p in prime_to_selected:
                            conflict_set.update(prime_to_selected[p])
                    for p in prime_factors[c2]:
                        if p in prime_to_selected:
                            conflict_set.update(prime_to_selected[p])
                    
                    conflict_cost = sum(values[b] for b in conflict_set)
                    gain = values[c1] + values[c2] - conflict_cost
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_add = [c1, c2]
                        best_remove = list(conflict_set)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
    
    # Phase 3: Simulated Annealing (if time permits)
    if time.time() < deadline - 5:
        # SA Parameters
        T = 0.1
        T_min = 0.001
        cooling = 0.99
        
        # We need a fast way to check validity of a move
        # Move: Remove a random selected element, Add a random unselected element (if coprime)
        
        sa_start = time.time()
        sa_deadline = deadline - 2
        
        while T > T_min and time.time() < sa_deadline:
            # Pick a random element to remove
            if not selected:
                break
            a_out = random.choice(list(selected))
            
            # Pick a random element to add
            # To speed up, we can pick from a precomputed list of unselected
            # But building the list is O(N). N=512 is small.
            unselected = [a for a in range(1, N) if a not in selected]
            if not unselected:
                break
            a_in = random.choice(unselected)
            
            # Check if a_in is coprime to all selected except a_out
            ok = True
            for p in prime_factors[a_in]:
                if p in prime_to_selected:
                    # Check if the conflicting elements include a_out
                    # If all conflicts are a_out, then it's valid after removing a_out
                    conflicts = prime_to_selected[p]
                    if len(conflicts) == 1 and a_out in conflicts:
                        continue
                    else:
                        ok = False
                        break
            
            if not ok:
                continue
            
            # Calculate gain
            gain = values[a_in] - values[a_out]
            
            # Acceptance criterion
            if gain > 0 or random.random() < pow(2.718281828, gain / T):
                # Apply move
                apply_move([a_in], [a_out])
                current_score += gain
            
            T *= cooling
    
    # Final Verification
    result = list(selected)
    for a in result:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]}, {result[j]}")
    
    return result
# EVOLVE-BLOCK-END
