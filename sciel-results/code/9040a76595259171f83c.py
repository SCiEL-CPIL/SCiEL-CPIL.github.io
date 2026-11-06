import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

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

def _compute_coverage(primes, residues, max_check=100000):
    """
    Computes the largest L such that all integers 1..L are covered.
    Uses a boolean array for speed.
    """
    if not primes:
        return 0
    
    # We only care up to max_check
    covered = np.zeros(max_check + 1, dtype=bool)
    
    # Optimization: If a prime is larger than max_check, it covers at most one number in range?
    # Actually, if p > max_check, it covers at most 1 number (r) if r <= max_check and r > 0.
    # But residues are 0 <= r < p. If r=0, it covers multiples of p.
    # If p > max_check, multiples are 0, p, 2p... only 0 is in 0..max_check (if we consider 0, but we start at 1).
    # So if p > max_check, it covers r if 1 <= r <= max_check.
    
    for p, r in zip(primes, residues):
        if p > max_check:
            if 1 <= r <= max_check:
                covered[r] = True
        else:
            # Mark all n in 1..max_check such that n % p == r
            # Start from the first positive integer congruent to r mod p
            start = r
            if start == 0:
                start = p
            if start > max_check:
                # Maybe r=0 and p <= max_check, then p, 2p...
                # If r > 0 and r > max_check, no coverage.
                # If r == 0, start is p.
                if r == 0 and p <= max_check:
                    start = p
                else:
                    continue
            
            # Generate indices: start, start+p, start+2p...
            # Using numpy for speed
            indices = np.arange(start, max_check + 1, p)
            if len(indices) > 0:
                covered[indices] = True
        
    # Find the first False in covered[1:]
    # If all True, return max_check
    # np.argmax returns the index of the first True. We want the first False.
    # ~covered[1:] gives True where not covered.
    # If all covered, no True in ~covered[1:], argmax returns 0? No, argmax on all False returns 0.
    # So we need to check if any are uncovered.
    
    uncovered = np.where(~covered[1:])[0]
    if len(uncovered) == 0:
        return max_check
    else:
        # The first uncovered index in 1..max_check
        # covered[1:] starts at index 1 of covered.
        # uncovered[0] is the index in covered[1:], so the actual number is uncovered[0] + 1
        return uncovered[0] + 1

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    primes_all = _get_primes(1000)
    
    # Initial heuristic: Pick primes that give good coverage with budget < 1
    # Start with a greedy set? Or the seed's set?
    # Seed uses primes from 13 upwards. Sum 1/p for primes 13..1000 is approx 0.92.
    # Let's start with a set of small primes to have more "flexibility" in swaps, 
    # or stick to the seed's set and improve it.
    
    # Better initial state: 
    # The seed uses primes 13, 17, 19, ... 
    # Let's try to initialize with a set of primes that sums to close to 1 but allows swaps.
    # Small primes have high 1/p cost. Large primes have low 1/p cost.
    # To cover 1..L, we need residues.
    
    # Let's start with the seed's configuration as a baseline, then improve.
    # Seed: primes 13..1000 (approx 168 primes).
    # Residues: (i+1) % p.
    
    initial_primes = [p for p in primes_all if p >= 13]
    initial_residues = [(i + 1) % p for i, p in enumerate(initial_primes)]
    
    # Check budget
    budget = sum(Fraction(1, p) for p in initial_primes)
    
    if budget > 1:
        # Remove largest primes until budget <= 1
        # Actually, removing large primes reduces budget the least? No, 1/p is small for large p.
        # To reduce budget, remove small primes (large 1/p).
        # But we want to keep coverage.
        # For now, just take the seed's set which is known to be < 1.
        pass
        
    current_primes = list(initial_primes)
    current_residues = list(initial_residues)
    
    # Sort by prime for consistent indexing? Not strictly necessary if we keep pairs.
    # Let's maintain a list of (p, r) pairs.
    current_classes = [[p, r] for p, r in zip(current_primes, current_residues)]
    
    current_coverage = _compute_coverage(current_primes, current_residues, 100000)
    best_classes = [c[:] for c in current_classes]
    best_coverage = current_coverage
    
    # Simulated Annealing
    # Temperature
    T = 10.0
    cooling = 0.99
    
    # We need to be able to swap primes.
    # Available primes: primes_all not in current_primes.
    # Current primes set for fast lookup
    current_prime_set = set(current_primes)
    
    num_steps = 50000
    step = 0
    
    while time.time() < deadline and step < num_steps:
        step += 1
        
        # Choose a move
        # 1. Change residue of a random prime
        # 2. Swap a prime out and another in (to change budget/coverage potential)
        
        move_type = random.random()
        
        new_classes = [c[:] for c in current_classes]
        new_primes = list(current_primes)
        new_residues = list(current_residues)
        
        if move_type < 0.7 and new_primes:
            # Change residue
            idx = random.randint(0, len(new_primes) - 1)
            p = new_primes[idx]
            old_r = new_residues[idx]
            new_r = random.randint(0, p - 1)
            if new_r != old_r:
                new_residues[idx] = new_r
            else:
                continue # No change
        else:
            # Swap prime
            if not new_primes:
                continue
            
            # Remove a random prime
            idx_out = random.randint(0, len(new_primes) - 1)
            p_out = new_primes[idx_out]
            
            # Find a candidate prime to add
            # Prefer primes that are not currently in the set
            available = [p for p in primes_all if p not in current_prime_set]
            if not available:
                continue
            
            # Bias towards smaller primes? Or random?
            # Smaller primes cover more numbers per unit of "residue space" but cost more budget.
            # Let's pick randomly from available, maybe weighted by 1/p?
            p_in = random.choice(available)
            
            # Check budget constraint roughly
            # We must ensure sum(1/p) <= 1
            # Current budget + 1/p_in - 1/p_out <= 1
            
            # To avoid expensive Fraction sum every time, we can keep a running sum or use floats for quick check
            # But exactness is required for final validity. For SA, float is okay for guidance.
            
            # Let's use float for speed in the loop
            # Precompute 1/p for all primes
            # 1/p is small.
            
            # If we just swap, we need to assign a residue to the new prime.
            # Initial residue for new prime: random or 0?
            # Let's try 0 first, or random.
            
            # Remove old
            new_primes.pop(idx_out)
            new_residues.pop(idx_out)
            
            # Add new
            new_primes.append(p_in)
            new_residues.append(random.randint(0, p_in - 1))
            
            # Update sets
            current_prime_set.discard(p_out)
            current_prime_set.add(p_in)
            
            # Note: current_prime_set is modified in place. We need to revert if we reject.
            # This is getting messy with in-place modification.
            # Let's simplify: Just do residue changes for most of the time.
            # Prime swapping is complex to manage state.
            
            # Let's revert and only do residue changes for simplicity and robustness,
            # unless we implement a proper candidate list.
            
            # Actually, the seed already has a good set of primes.
            # Maybe optimizing residues for THIS set is sufficient to get a big jump?
            # The seed score is 1.0. Inspiration 1 got 1.67 with "randomized local search... residue assignments".
            # So maybe just optimizing residues is enough!
            
            # Let's revert the swap logic to avoid bugs.
            current_prime_set.discard(p_in)
            current_prime_set.add(p_out)
            new_primes.pop()
            new_residues.pop()
            new_primes.insert(idx_out, p_out)
            new_residues.insert(idx_out, current_residues[idx_out]) # Wait, old_r was popped?
            # This is getting error prone.
            
            # Strategy Shift: Stick to Residue Optimization only.
            # The set of primes is fixed to the seed's set (13..1000).
            # This is simpler and likely sufficient to beat 1.0 significantly.
            continue
        
        # Evaluate new state
        new_coverage = _compute_coverage(new_primes, new_residues, 100000)
        
        # Acceptance criterion
        if new_coverage > current_coverage:
            current_coverage = new_coverage
            current_classes = new_classes
            current_primes = new_primes
            current_residues = new_residues
            
            if current_coverage > best_coverage:
                best_coverage = current_coverage
                best_classes = [c[:] for c in current_classes]
                
            # Update T? No, keep T for now or cool down.
            T *= cooling
        else:
            # Accept with probability exp((delta) / T)
            # delta is negative
            delta = new_coverage - current_coverage
            if delta < 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_coverage = new_coverage
                    current_classes = new_classes
                    current_primes = new_primes
                    current_residues = new_residues
                    T *= cooling
                else:
                    T *= cooling
            else:
                # Should not happen if new_coverage <= current_coverage
                T *= cooling
        
        # Safety: Ensure we don't spend too much time on one evaluation
        if time.time() > deadline - 5.0:
            break
            
    # Final Verification
    # Check budget of best_classes
    best_primes = [c[0] for c in best_classes]
    best_residues = [c[1] for c in best_classes]
    
    budget = sum(Fraction(1, p) for p in best_primes)
    if budget > 1:
        # Fallback to seed if budget exceeded (shouldn't happen if we didn't swap primes)
        return [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]
    
    # Verify coverage one last time?
    # The score is determined by the evaluator.
    return best_classes

# EVOLVE-BLOCK-END
