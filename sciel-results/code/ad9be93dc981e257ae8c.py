"""Improved program using iterated local search with multiple move types."""
import time
import math
import random

MAX_MOD = 128
DEADLINE = 85.0  # 90s limit with 5s safety margin

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _can_add(r, m, selected):
    """Check if (r, m) can be added to the existing set."""
    for (er, em) in selected:
        g = math.gcd(m, em)
        if (r - er) % g == 0:
            return False
    return True

def _greedy(ordering, time_limit):
    """Greedy construction: iterate through moduli in given order."""
    start_time = time.time()
    selected = []
    
    for m in ordering:
        if time.time() - start_time > time_limit:
            break
        # Try residues, preferring those that are "far" from existing
        # Use a simple heuristic: try all residues, pick first that works
        found = False
        for r in range(m):
            if _can_add(r, m, selected):
                selected.append((r, m))
                found = True
                break
        # If no residue works for this modulus, skip it
    
    return selected

def _local_search(selected, time_limit):
    """Local search with 1-for-2 swap moves."""
    start_time = time.time()
    current = list(selected)
    current_size = len(current)
    best = list(current)
    best_size = current_size
    
    # Precompute gcd table for speed
    # We'll cache gcd(m1, m2) results
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try each possible removal
        for i in range(len(current)):
            if time.time() - start_time > time_limit:
                break
            
            removed = current[i]
            remaining = current[:i] + current[i+1:]
            removed_mod = removed[1]
            
            # Find available moduli
            used_moduli = set(mm for (_, mm) in remaining)
            avail_moduli = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
            
            if len(avail_moduli) < 2:
                continue
            
            # Try adding two new moduli
            for j in range(len(avail_moduli)):
                if time.time() - start_time > time_limit:
                    break
                m1 = avail_moduli[j]
                
                # Find a valid residue for m1
                r1_found = None
                for r1 in range(m1):
                    if _can_add(r1, m1, remaining):
                        r1_found = r1
                        break
                
                if r1_found is None:
                    continue
                
                temp_remaining = remaining + [(r1_found, m1)]
                
                for k in range(j + 1, len(avail_moduli)):
                    m2 = avail_moduli[k]
                    
                    for r2 in range(m2):
                        if _can_add(r2, m2, temp_remaining):
                            new_current = remaining + [(r1_found, m1), (r2, m2)]
                            if len(new_current) > best_size:
                                current = new_current
                                best = list(current)
                                best_size = len(current)
                                improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try 1-for-1 moves (swap one element for another)
            swapped = False
            for i in range(len(current)):
                if time.time() - start_time > time_limit:
                    break
                
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                removed_mod = removed[1]
                
                used_moduli = set(mm for (_, mm) in remaining)
                used_moduli.add(removed_mod)  # Keep the same modulus available? No, we want different
                # Actually for 1-for-1, we replace (r, m) with (r', m') where m' != m
                # But we want to potentially keep the same modulus with different residue
                # Let's try both: same modulus different residue, or different modulus
                
                # First: try same modulus with different residue
                for r in range(removed[1]):
                    if r == removed[0]:
                        continue
                    if _can_add(r, removed[1], remaining):
                        new_current = remaining + [(r, removed[1])]
                        # Size is same, but might enable further improvements
                        # Don't accept same-size moves in 1-for-1
                        pass
                
                # Try different modulus
                avail = [m for m in range(2, MAX_MOD + 1) if m not in set(mm for (_, mm) in remaining)]
                for m_new in avail:
                    if m_new == removed_mod:
                        continue
                    for r_new in range(m_new):
                        if _can_add(r_new, m_new, remaining):
                            # Same size, skip for now (would need 1-for-2 to gain)
                            pass
            
            # If stuck, try a random perturbation
            if time.time() - start_time > time_limit:
                break
            
            # Remove a random element and re-greedy
            if len(current) > 2:
                idx = random.randint(0, len(current) - 1)
                perturbed = current[:idx] + current[idx+1:]
                # Re-add using greedy on remaining moduli
                used = set(mm for (_, mm) in perturbed)
                avail = [m for m in range(2, MAX_MOD + 1) if m not in used]
                random.shuffle(avail)
                for m in avail:
                    if time.time() - start_time > time_limit:
                        break
                    for r in range(m):
                        if _can_add(r, m, perturbed):
                            perturbed.append((r, m))
                            break
                if len(perturbed) >= best_size:
                    current = perturbed
                break
    
    return best

def construct():
    start_time = time.time()
    time_limit = DEADLINE - (time.time() - start_time)
    
    if time_limit <= 1:
        # Fallback
        return [[1, 2], [1, 3], [2, 4], [1, 5], [3, 6], [1, 7], [2, 8]]
    
    random.seed(42)
    
    best = []
    best_size = 0
    
    # Phase 1: Multiple greedy constructions with different orderings
    greedy_time = min(5.0, time_limit * 0.15)
    greedy_start = time.time()
    
    orderings = []
    
    # Natural order
    orderings.append(list(range(2, MAX_MOD + 1)))
    # Reverse
    orderings.append(list(range(MAX_MOD, 1, -1)))
    
    # Primes first
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [m for m in range(2, MAX_MOD + 1) if is_prime(m)]
    composites = [m for m in range(2, MAX_MOD + 1) if not is_prime(m)]
    orderings.append(primes + composites)
    orderings.append(composites + primes)
    
    # Sort by number of divisors (more divisors = more constraints = harder)
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    by_divisors_desc = sorted(range(2, MAX_MOD + 1), key=lambda x: -num_divisors(x))
    by_divisors_asc = sorted(range(2, MAX_MOD + 1), key=lambda x: num_divisors(x))
    orderings.append(by_divisors_desc)
    orderings.append(by_divisors_asc)
    
    # Random orderings
    for _ in range(30):
        shuffled = list(range(2, MAX_MOD + 1))
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    for ordering in orderings:
        if time.time() - greedy_start > greedy_time:
            break
        remaining = greedy_time - (time.time() - greedy_start)
        if remaining <= 0:
            break
        result = _greedy(ordering, remaining)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Phase 2: Local search
    ls_time = min(40.0, time_limit * 0.6)
    ls_start = time.time()
    
    if best_size > 0 and ls_time > 1:
        improved = _local_search(best, ls_time)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Phase 3: More restarts with perturbation
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed
    
    if remaining_time > 2 and best_size > 0:
        # Try additional local search passes
        while time.time() - start_time < DEADLINE - 1.0:
            # Perturb current best
            perturbed = list(best)
            if len(perturbed) > 3:
                # Remove 1-3 random elements
                num_remove = min(3, len(perturbed) - 2)
                indices = random.sample(range(len(perturbed)), num_remove)
                indices.sort(reverse=True)
                for idx in indices:
                    perturbed.pop(idx)
            
            # Re-greedy on remaining moduli
            used = set(mm for (_, mm) in perturbed)
            avail = [m for m in range(2, MAX_MOD + 1) if m not in used]
            random.shuffle(avail)
            for m in avail:
                if time.time() - start_time > DEADLINE - 1.0:
                    break
                for r in range(m):
                    if _can_add(r, m, perturbed):
                        perturbed.append((r, m))
                        break
            
            # Local search on perturbed
            ls_remaining = min(5.0, DEADLINE - 1.0 - (time.time() - start_time))
            if ls_remaining > 0.5 and len(perturbed) >= 2:
                improved = _local_search(perturbed, ls_remaining)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(best)
            
            if time.time() - start_time > DEADLINE - 2.0:
                break
    
    # Convert to required format
    result = [[r, m] for r, m in best]
    
    if not result:
        return [[1, 2], [1, 3], [2, 4], [1, 5], [3, 6], [1, 7], [2, 8]]
    
    return result
