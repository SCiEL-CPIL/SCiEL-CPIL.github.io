import time
from fractions import Fraction
import math
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _compute_L_fast(classes, max_check=200000):
    """Compute largest L such that 1..L all covered. Uses coverage count array."""
    if not classes:
        return 0
    
    # Count how many classes cover each position
    # For efficiency, we only need to check up to some reasonable bound
    # But L is capped at 10^5 per task spec
    max_check = min(max_check, 100000)
    
    # Use an array to count coverage
    # covered[n] = number of classes that cover n
    # We need covered[n] >= 1 for all n in 1..L
    
    # Efficient approach: for each class, mark its coverage
    # But we need to find the first gap
    
    # Alternative: for each n from 1 upward, check if any class covers it
    # This is O(L * num_classes) which could be slow
    
    # Better: build a boolean array
    covered = bytearray(max_check + 1)  # 0 or 1
    
    for p, r in classes:
        # Mark all n in 1..max_check where n ≡ r (mod p)
        # First such n >= 1:
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p
        for n in range(first, max_check + 1, p):
            covered[n] = 1
    
    # Find first gap
    L = 0
    for n in range(1, max_check + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _compute_L_binary(classes, lo=0, hi=100000):
    """Binary search for L using a coverage check."""
    # We need to check if 1..mid are all covered
    # For a given mid, check each n in 1..mid
    
    def all_covered(mid):
        for n in range(1, mid + 1):
            found = False
            for p, r in classes:
                if n % p == r:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # This is too slow for large mid. Use the fast method instead.
    return _compute_L_fast(classes, min(hi, 100000))

def construct():
    deadline = time.time() + 145
    random.seed(42)
    
    primes = _primes_upto(1000)
    prime_set = set(primes)
    
    # Precompute 1/p as Fraction for all primes
    inv_p = {p: Fraction(1, p) for p in primes}
    
    # Strategy: 
    # 1. Start with a good initial solution
    # 2. Use simulated annealing / local search to improve
    # 3. The state is: which primes are used and their residues
    # 4. Moves: add a prime, remove a prime, change a residue
    
    # Initial solution: use primes from 13 upward with residue = index+1
    seed_primes = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    seed_budget = sum(inv_p[p] for p, r in seed_classes)
    seed_L = _compute_L_fast(seed_classes, 100000)
    
    best_L = seed_L
    best_classes = seed_classes[:]
    best_budget = seed_budget
    
    # Simulated annealing parameters
    T_initial = 100.0
    T_min = 0.1
    cooling = 0.999
    
    current_classes = seed_classes[:]
    current_budget = seed_budget
    current_L = seed_L
    
    T = T_initial
    
    # Track which primes are in use
    in_use = set(p for p, r in current_classes)
    
    # For efficient coverage checking, maintain a coverage count array
    # covered_count[n] = number of classes covering n
    max_check = 100000
    covered_count = [0] * (max_check + 1)
    
    def rebuild_coverage(classes):
        nonlocal covered_count
        covered_count = [0] * (max_check + 1)
        for p, r in classes:
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p
            for n in range(first, max_check + 1, p):
                covered_count[n] += 1
    
    def get_L_from_coverage():
        L = 0
        for n in range(1, max_check + 1):
            if covered_count[n] > 0:
                L = n
            else:
                break
        return L
    
    rebuild_coverage(current_classes)
    current_L = get_L_from_coverage()
    
    if current_L > best_L:
        best_L = current_L
        best_classes = current_classes[:]
        best_budget = current_budget
    
    # Simulated annealing loop
    step = 0
    while time.time() < deadline - 5 and T > T_min:
        step += 1
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Change residue of an existing prime
            if not current_classes:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            p, old_r = current_classes[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Apply move
            current_classes[idx] = [p, new_r]
            
            # Update coverage
            # Remove old coverage
            if old_r == 0:
                first_old = p
            else:
                first_old = old_r
            if first_old < 1:
                first_old += p
            for n in range(first_old, max_check + 1, p):
                covered_count[n] -= 1
            
            # Add new coverage
            if new_r == 0:
                first_new = p
            else:
                first_new = new_r
            if first_new < 1:
                first_new += p
            for n in range(first_new, max_check + 1, p):
                covered_count[n] += 1
            
            new_L = get_L_from_coverage()
            delta_L = new_L - current_L
            
            # Accept if better, or with probability exp(delta_L / T) if worse
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = current_classes[:]
                    best_budget = current_budget
            else:
                # Revert
                current_classes[idx] = [p, old_r]
                # Revert coverage
                if new_r == 0:
                    first_new = p
                else:
                    first_new = new_r
                if first_new < 1:
                    first_new += p
                for n in range(first_new, max_check + 1, p):
                    covered_count[n] -= 1
                if old_r == 0:
                    first_old = p
                else:
                    first_old = old_r
                if first_old < 1:
                    first_old += p
                for n in range(first_old, max_check + 1, p):
                    covered_count[n] += 1
                    
        elif move_type < 0.7:
            # Add a new prime (if budget allows)
            available_primes = [p for p in primes if p not in in_use]
            if not available_primes:
                continue
            
            # Choose a prime that's affordable
            affordable = [p for p in available_primes if current_budget + inv_p[p] <= 1]
            if not affordable:
                continue
            
            p = random.choice(affordable)
            
            # Choose residue: try to cover the first gap
            # Find first uncovered position
            gap = None
            for n in range(1, min(current_L + 500, max_check + 1)):
                if covered_count[n] == 0:
                    gap = n
                    break
            
            if gap is not None:
                # Choose residue to cover the gap
                r = gap % p
            else:
                r = random.randint(0, p - 1)
            
            # Apply move
            current_classes.append([p, r])
            in_use.add(p)
            current_budget += inv_p[p]
            
            # Add coverage
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p
            for n in range(first, max_check + 1, p):
                covered_count[n] += 1
            
            new_L = get_L_from_coverage()
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = current_classes[:]
                    best_budget = current_budget
            else:
                # Revert
                current_classes.pop()
                in_use.discard(p)
                current_budget -= inv_p[p]
                if r == 0:
                    first = p
                else:
                    first = r
                if first < 1:
                    first += p
                for n in range(first, max_check + 1, p):
                    covered_count[n] -= 1
                    
        else:
            # Remove a prime (to free up budget or simplify)
            if len(current_classes) <= 10:
                continue
            
            idx = random.randint(0, len(current_classes) - 1)
            p, r = current_classes[idx]
            
            # Only remove if it doesn't create a gap in 1..current_L
            # Check if any position in 1..current_L is only covered by this prime
            safe = True
            for n in range(1, min(current_L + 1, max_check + 1)):
                if covered_count[n] == 1:
                    # Check if this prime covers n
                    if n % p == r:
                        safe = False
                        break
            
            if not safe:
                continue
            
            # Apply removal
            current_classes.pop(idx)
            in_use.discard(p)
            current_budget -= inv_p[p]
            
            # Remove coverage
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p
            for n in range(first, max_check + 1, p):
                covered_count[n] -= 1
            
            new_L = get_L_from_coverage()
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = current_classes[:]
                    best_budget = current_budget
            else:
                # Revert
                current_classes.insert(idx, [p, r])
                in_use.add(p)
                current_budget += inv_p[p]
                if r == 0:
                    first = p
                else:
                    first = r
                if first < 1:
                    first += p
                for n in range(first, max_check + 1, p):
                    covered_count[n] += 1
        
        T *= cooling
        
        # Periodic restart: if stuck for a while, try a different strategy
        if step % 500 == 0 and time.time() < deadline - 20:
            # Try a greedy construction from scratch
            greedy_classes = _greedy_construct(primes, inv_p, max_check, deadline)
            if greedy_classes:
                gL = _compute_L_fast(greedy_classes, max_check)
                if gL > best_L:
                    best_L = gL
                    best_classes = greedy_classes[:]
                    best_budget = sum(inv_p[p] for p, r in greedy_classes)
                    # Reset current state to this
                    current_classes = greedy_classes[:]
                    current_budget = best_budget
                    current_L = gL
                    in_use = set(p for p, r in current_classes)
                    rebuild_coverage(current_classes)
    
    # Final local search on best solution
    if time.time() < deadline - 3:
        final_classes = best_classes[:]
        final_budget = best_budget
        final_L = best_L
        in_use_final = set(p for p, r in final_classes)
        
        rebuild_coverage(final_classes)
        final_L = get_L_from_coverage()
        
        # Try to improve each residue
        for _iter in range(3):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(len(final_classes)):
                if time.time() > deadline - 2:
                    break
                p, old_r = final_classes[i]
                best_r = old_r
                best_L_i = final_L
                
                # Try a subset of residues (not all, for speed)
                candidates = [old_r]
                # Add some random candidates
                for _ in range(min(p, 20)):
                    candidates.append(random.randint(0, p - 1))
                
                for r2 in candidates:
                    if r2 == old_r:
                        continue
                    
                    # Temporarily change
                    # Remove old coverage
                    if old_r == 0:
                        first_old = p
                    else:
                        first_old = old_r
                    if first_old < 1:
                        first_old += p
                    for n in range(first_old, max_check + 1, p):
                        covered_count[n] -= 1
                    
                    # Add new coverage
                    if r2 == 0:
                        first_new = p
                    else:
                        first_new = r2
                    if first_new < 1:
                        first_new += p
                    for n in range(first_new, max_check + 1, p):
                        covered_count[n] += 1
                    
                    new_L = get_L_from_coverage()
                    
                    # Revert
                    if r2 == 0:
                        first_new = p
                    else:
                        first_new = r2
                    if first_new < 1:
                        first_new += p
                    for n in range(first_new, max_check + 1, p):
                        covered_count[n] -= 1
                    if old_r == 0:
                        first_old = p
                    else:
                        first_old = old_r
                    if first_old < 1:
                        first_old += p
                    for n in range(first_old, max_check + 1, p):
                        covered_count[n] += 1
                    
                    if new_L > best_L_i:
                        best_L_i = new_L
                        best_r = r2
                
                if best_r != old_r:
                    # Apply the best
                    p, old_r = final_classes[i]
                    if old_r == 0:
                        first_old = p
                    else:
                        first_old = old_r
                    if first_old < 1:
                        first_old += p
                    for n in range(first_old, max_check + 1, p):
                        covered_count[n] -= 1
                    if best_r == 0:
                        first_new = p
                    else:
                        first_new = best_r
                    if first_new < 1:
                        first_new += p
                    for n in range(first_new, max_check + 1, p):
                        covered_count[n] += 1
                    final_classes[i] = [p, best_r]
                    final_L = best_L_i
                    improved = True
            
            if not improved:
                break
        
        if final_L > best_L:
            best_L = final_L
            best_classes = final_classes[:]
            best_budget = final_budget
    
    return best_classes

def _greedy_construct(primes, inv_p, max_check, deadline):
    """Greedy: start empty, add primes to cover gaps."""
    classes = []
    budget = Fraction(0)
    covered = bytearray(max_check + 1)
    
    for p in primes:
        if time.time() > deadline - 10:
            break
        cost = inv_p[p]
        if budget + cost > 1:
            continue
        
        # Find the first uncovered position
        gap = None
        for n in range(1, max_check + 1):
            if not covered[n]:
                gap = n
                break
        
        if gap is None:
            # All covered up to max_check
            break
        
        # Choose residue to cover the gap
        r = gap % p
        
        # Check if this helps: does it cover the gap and maybe more?
        # Just add it
        classes.append([p, r])
        budget += cost
        
        # Update coverage
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p
        for n in range(first, max_check + 1, p):
            covered[n] = 1
        
        # Check if we've covered 1..L for some L
        # (We'll compute at the end)
    
    return classes if classes else None

# EVOLVE-BLOCK-END
