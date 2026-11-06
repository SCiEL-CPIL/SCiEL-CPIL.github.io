"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    primes = _sieve(1000)
    max_check = 100000

    def compute_L(covered):
        """Compute the largest L such that 1..L are all covered."""
        sub = covered[1:]
        false_indices = np.where(~sub)[0]
        if len(false_indices) == 0:
            return max_check
        return false_indices[0]

    def apply_residue(covered, p, r):
        """Apply a residue class to the covered array (in-place)."""
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        covered[start::p] = True

    def remove_residue(covered, p, r):
        """Remove a residue class from the covered array (in-place)."""
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        covered[start::p] = False

    def greedy_residues_numpy(prime_list, max_check=100000):
        """Assign residues greedily using numpy for speed."""
        if not prime_list:
            return []

        covered = np.zeros(max_check + 1, dtype=bool)
        classes = []

        for p in prime_list:
            if time.time() > deadline:
                break

            # Find the first uncovered number
            sub = covered[1:]
            false_indices = np.where(~sub)[0]
            if len(false_indices) == 0:
                # All covered up to max_check, just pick any residue
                classes.append([p, 0])
                continue
            
            first_uncovered = int(false_indices[0])
            r_target = first_uncovered % p
            best_r = r_target
            best_L = first_uncovered - 1

            # For small primes, try all residues; for large, try target + neighbors
            if p <= 50:
                residues_to_try = list(range(p))
            else:
                residues_to_try = [r_target]
                for d in range(-10, 11):
                    r_alt = (r_target + d) % p
                    if r_alt != r_target:
                        residues_to_try.append(r_alt)
                # Add some random samples for diversity
                for _ in range(min(15, p // 10)):
                    rand_r = random.randint(0, p - 1)
                    if rand_r not in residues_to_try:
                        residues_to_try.append(rand_r)

            for r in residues_to_try:
                if r == 0:
                    start = p
                else:
                    start = r
                if start < 1:
                    start += p
                mask = np.zeros(max_check + 1, dtype=bool)
                mask[start::p] = True
                new_covered = covered | mask
                new_L = compute_L(new_covered)
                if new_L > best_L:
                    best_L = new_L
                    best_r = r

            classes.append([p, int(best_r)])

            # Update covered array
            r = best_r
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            covered[start::p] = True

        return classes

    def optimize_residues(classes, max_check=100000, time_limit=None):
        """
        Perform Cyclic Coordinate Descent to optimize residues.
        Repeatedly tries all residues for each prime to maximize L.
        """
        if not classes:
            return classes, 0
        
        if time_limit is None:
            time_limit = deadline - 5
            
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            apply_residue(covered, p, r)
        
        current_L = compute_L(covered)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            for idx, (p, r) in enumerate(classes):
                if time.time() > time_limit:
                    return classes, current_L
                
                best_r = r
                best_L = current_L
                
                # Try all residues for this prime
                for new_r in range(p):
                    if new_r == r:
                        continue
                    
                    # Remove current residue
                    remove_residue(covered, p, r)
                    # Add new residue
                    apply_residue(covered, p, new_r)
                    
                    new_L = compute_L(covered)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = new_r
                        # Keep the new residue applied for now to check next ones
                        # But we need to revert if this isn't the best at the end of the loop?
                        # Actually, Cyclic Coordinate Descent usually keeps the best change immediately.
                        # Let's keep it.
                        r = new_r 
                        classes[idx][1] = new_r
                        improved = True
                    else:
                        # Revert to the best found so far (which might be original r or a previous new_r)
                        # Wait, the logic above updates r and classes immediately if better.
                        # If not better, we need to revert to the state BEFORE this trial.
                        # But the state BEFORE this trial was the one with 'r' (which might have been updated in a previous iteration of this inner loop).
                        # Let's re-structure:
                        # Save state, try, if better update, else revert.
                        
                        # Correction: The code above is slightly flawed in revert logic if we update r inside.
                        # Let's do it cleanly:
                        pass

                # The inner loop above was messy. Let's rewrite the inner logic clearly.
                # We will restart the optimization function with a cleaner inner loop implementation below.
                # For now, break to avoid using the buggy version.
                # Actually, I will just implement the clean version in the final code.
                
        return classes, current_L

    # Refined Optimize Function
    def optimize_residues_v2(classes, max_check=100000, time_limit=None):
        if not classes:
            return classes, 0
        
        if time_limit is None:
            time_limit = deadline - 5
            
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            apply_residue(covered, p, r)
        
        current_L = compute_L(covered)
        
        # Cyclic Coordinate Descent with clean logic
        while time.time() < time_limit:
            any_improvement = False
            for idx in range(len(classes)):
                if time.time() > time_limit:
                    break
                
                p, r = classes[idx]
                best_r = r
                best_L = current_L
                
                # Determine candidate residues
                if p <= 40:
                    candidates = list(range(p))
                else:
                    candidates = [r]
                    for d in range(-15, 16):
                        alt = (r + d) % p
                        if alt != r:
                            candidates.append(alt)
                    # Add random samples for diversity
                    for _ in range(min(30, p // 3)):
                        rand_r = random.randint(0, p - 1)
                        if rand_r not in candidates:
                            candidates.append(rand_r)
                
                # Save current state
                saved_r = r
                
                for new_r in candidates:
                    if new_r == r:
                        continue
                    
                    # Remove current, try new
                    remove_residue(covered, p, r)
                    apply_residue(covered, p, new_r)
                    new_L = compute_L(covered)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = new_r
                        # Keep this applied for now
                        r = new_r
                    else:
                        # Revert to previous best state
                        remove_residue(covered, p, new_r)
                        apply_residue(covered, p, r)
                
                # After trying all candidates, ensure best_r is applied
                if best_r != saved_r:
                    # Remove whatever is currently applied, apply best_r
                    remove_residue(covered, p, r)
                    apply_residue(covered, p, best_r)
                    current_L = best_L
                    classes[idx][1] = best_r
                    any_improvement = True
                else:
                    # Ensure original is properly applied
                    remove_residue(covered, p, r)
                    apply_residue(covered, p, saved_r)
            
            if not any_improvement:
                break
                
        return classes, current_L

    def simulated_annealing(classes, primes_available, max_check=100000, time_limit=None):
        """Simulated annealing to escape local optima."""
        if not classes:
            return classes, 0
            
        if time_limit is None:
            time_limit = deadline - 5
        
        import math
        
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            apply_residue(covered, p, r)
        
        current_L = compute_L(covered)
        best_L = current_L
        best_classes = [list(c) for c in classes]
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        total_iterations = 5000
        current_time = time.time()
        elapsed = current_time - (time_limit - (deadline - 5)) if time_limit < deadline else 0
        
        for it in range(total_iterations):
            if time.time() > time_limit:
                break
            
            # Temperature annealing
            progress = it / max(1, total_iterations)
            T = T_start * (T_end / T_start) ** progress
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.6:
                # Change residue of an existing prime
                if not classes:
                    continue
                idx = random.randint(0, len(classes) - 1)
                p, r = classes[idx]
                new_r = random.randint(0, p - 1)
                if new_r == r:
                    continue
                
                # Apply move
                remove_residue(covered, p, r)
                apply_residue(covered, p, new_r)
                new_L = compute_L(covered)
                
                # Accept if better, or with probability exp(-(new_L - current_L)/T) if worse
                if new_L >= current_L:
                    classes[idx][1] = new_r
                    current_L = new_L
                else:
                    delta = current_L - new_L
                    if random.random() < math.exp(-delta / max(T, 0.01)):
                        classes[idx][1] = new_r
                        current_L = new_L
                    else:
                        # Revert
                        remove_residue(covered, p, new_r)
                        apply_residue(covered, p, r)
                        
            elif move_type < 0.85:
                # Add a new prime from available
                if not primes_available:
                    continue
                # Check budget
                current_budget = sum(Fraction(1, p) for p, r in classes)
                if current_budget >= Fraction(1):
                    continue
                
                p = random.choice(primes_available)
                r = random.randint(0, p - 1)
                
                # Check if this prime is already in classes
                existing_primes = [cp for cp, cr in classes]
                if p in existing_primes:
                    continue
                
                cost = Fraction(1, p)
                if current_budget + cost > Fraction(1):
                    continue
                
                apply_residue(covered, p, r)
                new_L = compute_L(covered)
                classes.append([p, r])
                current_L = new_L
                
            else:
                # Remove a prime
                if len(classes) <= 2:
                    continue
                idx = random.randint(0, len(classes) - 1)
                p, r = classes[idx]
                
                remove_residue(covered, p, r)
                new_L = compute_L(covered)
                classes.pop(idx)
                current_L = new_L
            
            # Track best
            if current_L > best_L:
                best_L = current_L
                best_classes = [list(c) for c in classes]
        
        return best_classes, best_L

    best_classes = []
    best_L = 0

    def evaluate_strategy(selected_primes, optimize_time=10):
        nonlocal best_classes, best_L
        if time.time() > deadline - 5:
            return
        if not selected_primes:
            return
        
        # 1. Greedy Init
        classes = greedy_residues_numpy(selected_primes, max_check)
        
        # 2. Optimize
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        opt_time = min(remaining - 5, optimize_time)
        if opt_time < 1:
            return
            
        classes, L = optimize_residues_v2(classes, max_check, time_limit=time.time() + opt_time)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]

    # Generate Candidate Strategies
    
    # Strategy 1: Smallest primes (Dense)
    primes_asc = sorted(primes)
    budget = Fraction(1)
    selected = []
    for p in primes_asc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    evaluate_strategy(selected, optimize_time=25)

    # Strategy 2: Largest primes (Many)
    primes_desc = sorted(primes, reverse=True)
    budget = Fraction(1)
    selected = []
    for p in primes_desc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    evaluate_strategy(selected, optimize_time=25)

    # Strategy 2b: Medium primes (balanced)
    if time.time() < deadline - 40:
        primes_mid = sorted(primes, key=lambda x: abs(x - 100))  # Around 100
        budget = Fraction(1)
        selected = []
        for p in primes_mid:
            cost = Fraction(1, p)
            if budget >= cost:
                budget -= cost
                selected.append(p)
            else:
                break
        evaluate_strategy(selected, optimize_time=20)

    # Strategy 3: Mixed (Small core, large fill)
    if time.time() < deadline - 40:
        for num_small in [2, 3, 4, 5]:
            if time.time() > deadline - 40:
                break
            small_primes = primes_asc[:num_small]
            budget = Fraction(1)
            for p in small_primes:
                budget -= Fraction(1, p)
            if budget < 0:
                continue

            selected = list(small_primes)
            for p in primes_desc:
                if p in small_primes:
                    continue
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)
            
            evaluate_strategy(selected, optimize_time=15)

    # Strategy 4: Randomized
    if time.time() < deadline - 25:
        random.seed(42)
        for restart in range(8):
            if time.time() > deadline - 25:
                break
            
            random_primes = primes.copy()
            random.shuffle(random_primes)
            budget = Fraction(1)
            selected = []
            for p in random_primes:
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)
                else:
                    continue
            
            if selected:
                evaluate_strategy(selected, optimize_time=12)

    # Strategy 5: Hybrid with weighted selection
    if time.time() < deadline - 35:
        random.seed(123)
        for restart in range(4):
            if time.time() > deadline - 35:
                break
            
            selected = []
            budget = Fraction(1)
            used = set()
            
            for p in primes:
                if p in used:
                    continue
                if random.random() < 0.25:
                    cost = Fraction(1, p)
                    if budget >= cost:
                        budget -= cost
                        selected.append(p)
                        used.add(p)
            
            if selected:
                evaluate_strategy(selected, optimize_time=12)

    # Strategy 6: Simulated Annealing on best solution
    if best_classes and time.time() < deadline - 20:
        available_primes = [p for p in primes if p not in [cp for cp, cr in best_classes]]
        sa_classes, sa_L = simulated_annealing(
            [list(c) for c in best_classes], 
            available_primes, 
            max_check, 
            time_limit=deadline - 5
        )
        if sa_L > best_L:
            best_L = sa_L
            best_classes = sa_classes

    return best_classes if best_classes else [[2, 1]]

# EVOLVE-BLOCK-END
