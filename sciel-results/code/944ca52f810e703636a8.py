"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_primes(limit):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    primes = _get_primes(1000)
    N = 10000  # Initial coverage horizon to search in
    
    # Precompute mask for each prime and residue? Too big.
    # Instead, we represent coverage as a boolean array or bitset.
    # For speed in local search, we'll use a list of booleans or integers.
    # Let's use a list of ints 0/1 for coverage of 1..N.
    
    # Initial solution: Use a greedy construction or the seed's heuristic.
    # Seed heuristic: primes from 13 up, residue (i+1) % p.
    # Let's start with a greedy "maximize coverage" construction.
    
    # State: list of (p, r)
    # We want to maximize L such that 1..L covered.
    
    # Greedy initialization:
    # Try to cover 1..N.
    # Start with empty set.
    # At each step, add (p, r) that maximizes new coverage in 1..N.
    # This is expensive. Let's use a simpler init: the seed's solution.
    
    # Seed solution:
    seed_primes = [p for p in primes if p >= 13]
    current_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    
    # Calculate initial coverage
    def compute_coverage(classes, limit):
        covered = [False] * (limit + 1)
        for p, r in classes:
            for k in range(r, limit + 1, p):
                if k >= 1:
                    covered[k] = True
        # Find largest L such that 1..L all covered
        L = 0
        for i in range(1, limit + 1):
            if covered[i]:
                L = i
            else:
                break
        return L, covered
    
    current_L, current_covered = compute_coverage(current_classes, N)
    best_L = current_L
    best_classes = [c[:] for c in current_classes]
    
    # Local Search:
    # Move types:
    # 1. Change residue of an existing prime.
    # 2. Swap a prime with another prime (and optimize its residue).
    # 3. Add a new prime (if budget allows).
    # 4. Remove a prime (if it doesn't reduce L).
    
    # Budget check
    def get_budget(classes):
        s = Fraction(0, 1)
        for p, r in classes:
            s += Fraction(1, p)
        return s
    
    current_budget = get_budget(current_classes)
    
    # To speed up, we only recompute coverage if the move potentially improves L.
    # We can check if the move covers the first uncovered number.
    
    # Let's implement a simulated annealing or hill climbing loop.
    # Given time constraints, a simple hill climbing with random restarts or diverse moves is good.
    
    # Optimization: Use a set of covered numbers? No, list is faster for sequential check.
    # But updating the list for a move is O(N/p).
    
    # Let's try to improve L by targeting the first gap.
    
    # Heuristic: If L is the current coverage, the first gap is L+1.
    # We want to cover L+1.
    # Find a prime p in current set (or new) such that we can change/add residue to cover L+1
    # without breaking coverage of 1..L.
    
    # This is a constraint satisfaction problem.
    
    # Simpler approach: Random local search with exact evaluation.
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.choice(['residue', 'swap', 'add', 'remove'])
        
        if move_type == 'residue':
            if not current_classes:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            p, r = current_classes[idx]
            new_r = random.randint(0, p - 1)
            if new_r == r:
                continue
            
            # Test move
            trial_classes = [c[:] for c in current_classes]
            trial_classes[idx][1] = new_r
            trial_L, _ = compute_coverage(trial_classes, N)
            
            if trial_L > current_L:
                current_classes = trial_classes
                current_L = trial_L
                current_budget = get_budget(current_classes)
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [c[:] for c in current_classes]
        
        elif move_type == 'swap':
            if not current_classes:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            p_old, r_old = current_classes[idx]
            
            # Choose new prime
            p_new = random.choice(primes)
            if p_new == p_old:
                continue
            
            # Check budget
            new_budget = current_budget - Fraction(1, p_old) + Fraction(1, p_new)
            if new_budget > Fraction(1, 1):
                continue
            
            # Optimize residue for p_new to maximize coverage
            # Try all residues? Or random?
            # Let's try a few random residues or the one that covers the first gap.
            # First gap is current_L + 1
            target = current_L + 1
            # We want p_new | (target - r) => r = target % p_new
            best_r = -1
            best_trial_L = -1
            
            # Try specific residue
            r_candidate = target % p_new
            trial_classes = [c[:] for c in current_classes]
            trial_classes[idx] = [p_new, r_candidate]
            trial_L, _ = compute_coverage(trial_classes, N)
            
            if trial_L > best_trial_L:
                best_trial_L = trial_L
                best_r = r_candidate
            
            # Try a few random ones
            for _ in range(5):
                r_rand = random.randint(0, p_new - 1)
                trial_classes[idx][1] = r_rand
                trial_L, _ = compute_coverage(trial_classes, N)
                if trial_L > best_trial_L:
                    best_trial_L = trial_L
                    best_r = r_rand
            
            if best_trial_L > current_L:
                current_classes[idx] = [p_new, best_r]
                current_L = best_trial_L
                current_budget = new_budget
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [c[:] for c in current_classes]
        
        elif move_type == 'add':
            # Add a new prime
            # Find a prime that fits in budget
            available = [p for p in primes if all(p != cp for cp, _ in current_classes)]
            if not available:
                continue
            p_new = random.choice(available)
            new_budget = current_budget + Fraction(1, p_new)
            if new_budget > Fraction(1, 1):
                continue
            
            target = current_L + 1
            r_candidate = target % p_new
            
            trial_classes = [c[:] for c in current_classes]
            trial_classes.append([p_new, r_candidate])
            trial_L, _ = compute_coverage(trial_classes, N)
            
            if trial_L > current_L:
                current_classes = trial_classes
                current_L = trial_L
                current_budget = new_budget
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [c[:] for c in current_classes]
        
        elif move_type == 'remove':
            if len(current_classes) <= 1:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            p, r = current_classes[idx]
            
            trial_classes = [c[:] for c in current_classes]
            trial_classes.pop(idx)
            trial_L, _ = compute_coverage(trial_classes, N)
            
            # Accept if L doesn't drop (or drops slightly but we save budget? No, we want max L)
            # If L is same, we might want to keep it if it allows future additions?
            # For now, only accept if L >= current_L.
            if trial_L >= current_L:
                current_classes = trial_classes
                current_L = trial_L
                current_budget = get_budget(current_classes)
                # Update best if improved (unlikely with remove)
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [c[:] for c in current_classes]

    # Final verification and cleanup
    # Ensure budget is strictly <= 1
    # The local search enforces this.
    
    return best_classes

# EVOLVE-BLOCK-END
