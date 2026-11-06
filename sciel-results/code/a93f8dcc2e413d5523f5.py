import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    
    # Helper to check coverage
    def evaluate(classes, max_check=5000):
        # classes: list of [p, r]
        # Returns (L, covered_bool, cover_count)
        covered = [False] * (max_check + 1)
        cover_count = [0] * (max_check + 1)
        for p, r in classes:
            for k in range(r if r > 0 else p, max_check + 1, p):
                covered[k] = True
                cover_count[k] += 1
        
        L = 0
        for i in range(1, max_check + 1):
            if covered[i]:
                L = i
            else:
                break
        return L, covered, cover_count

    def sum_recip(classes):
        s = Fraction(0)
        for p, r in classes:
            s += Fraction(1, p)
        return s

    # Local search on residues
    def optimize_residues(initial_classes, max_check=5000):
        classes = [c[:] for c in initial_classes]
        L, covered, cover_count = evaluate(classes, max_check)
        
        # Precompute primes in classes
        p_list = [c[0] for c in classes]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_target = L + 1
            if n_target > max_check:
                break
            
            # Try to change residue of an existing prime to cover n_target
            best_move = None
            best_gain = 0
            
            for i, p in enumerate(p_list):
                r_old = classes[i][1]
                r_new = n_target % p
                if r_new == r_old:
                    continue
                
                # Check if changing r_i to r_new keeps 1..L covered
                # A number k is at risk if cover_count[k] == 1 and k % p == r_old
                # We need to check if any such k exists
                safe = True
                # Iterate through multiples of p up to L
                # k = r_old, r_old + p, ... <= L
                first = r_old if r_old > 0 else p
                for k in range(first, L + 1, p):
                    if cover_count[k] == 1:
                        # k is only covered by p with r_old
                        # If we change r, k is uncovered
                        # Unless k is also covered by new r? No, r_new != r_old
                        # So k becomes uncovered.
                        safe = False
                        break
                
                if safe:
                    # Move is valid
                    # Gain is at least 1 (covers n_target)
                    # Could be more if it covers other numbers > L
                    # But we only care about extending L
                    # So gain is 1
                    # We can just take the first valid move
                    best_move = (i, r_new)
                    break # Take first valid move to save time? Or search for best?
                    # Taking first valid move is a greedy strategy.
                    # It might get stuck in local optima.
                    # But with time limit, it's a good heuristic.
            
            if best_move:
                i, r_new = best_move
                classes[i][1] = r_new
                # Update coverage
                L, covered, cover_count = evaluate(classes, max_check)
                improved = True
            else:
                # No single residue change works
                # Try adding a new prime?
                # Check budget
                current_sum = sum_recip(classes)
                remaining = Fraction(1) - current_sum
                if remaining > 0:
                    # Find a prime not in classes with 1/p <= remaining
                    used_primes = set(p_list)
                    # Try smallest primes first
                    for p in primes:
                        if p in used_primes:
                            continue
                        if Fraction(1, p) <= remaining:
                            # Add this prime
                            r_new = n_target % p
                            classes.append([p, r_new])
                            p_list.append(p)
                            L, covered, cover_count = evaluate(classes, max_check)
                            improved = True
                            break
                
                if not improved:
                    # Try removing a prime to free up budget?
                    # Or random restart?
                    # For now, stop if no move found
                    break
        
        return classes, L

    # Candidate sets of primes
    candidates = []
    
    # 1. Seed-like: primes from 13 upwards
    seed_primes = [p for p in primes if p >= 13]
    # Filter to fit budget
    curr_sum = Fraction(0)
    selected = []
    for p in seed_primes:
        if curr_sum + Fraction(1, p) <= 1:
            selected.append(p)
            curr_sum += Fraction(1, p)
        else:
            # Try to skip this prime and see if we can add more later?
            # Greedy is simple
            pass
    # Assign residues to cover 1, 2, ...
    init_classes = [[p, (i+1) % p] for i, p in enumerate(selected)]
    candidates.append(init_classes)
    
    # 2. Small primes set
    small_primes = [p for p in primes if p <= 30]
    # Greedy select to fit budget
    curr_sum = Fraction(0)
    selected = []
    # Sort by 1/p descending (i.e., p ascending)
    for p in sorted(small_primes):
        if curr_sum + Fraction(1, p) <= 1:
            selected.append(p)
            curr_sum += Fraction(1, p)
    # Assign residues randomly or greedily
    init_classes = [[p, random.randint(0, p-1)] for p in selected]
    candidates.append(init_classes)
    
    # 3. Mixed set
    mixed_primes = [p for p in primes if p <= 100]
    curr_sum = Fraction(0)
    selected = []
    for p in sorted(mixed_primes):
        if curr_sum + Fraction(1, p) <= 1:
            selected.append(p)
            curr_sum += Fraction(1, p)
    init_classes = [[p, random.randint(0, p-1)] for p in selected]
    candidates.append(init_classes)
    
    best_classes = None
    best_L = 0
    
    for init_classes in candidates:
        if time.time() > deadline:
            break
        classes, L = optimize_residues(init_classes)
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Final check and return
    if best_classes is None:
        return []
        
    # Ensure validity
    if sum_recip(best_classes) > 1:
        # This shouldn't happen, but just in case
        return []
        
    return best_classes
# EVOLVE-BLOCK-END
