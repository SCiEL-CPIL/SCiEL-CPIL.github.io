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
    Returns (L, covered_array) to allow reuse of the array in the main loop.
    """
    if not primes:
        return 0, np.zeros(max_check + 1, dtype=bool)
    
    covered = np.zeros(max_check + 1, dtype=bool)
    
    for p, r in zip(primes, residues):
        if p > max_check:
            if 1 <= r <= max_check:
                covered[r] = True
        else:
            start = r
            if start == 0:
                start = p
            if start > max_check:
                if r == 0 and p <= max_check:
                    start = p
                else:
                    continue
            
            indices = np.arange(start, max_check + 1, p)
            if len(indices) > 0:
                covered[indices] = True
        
    uncovered = np.where(~covered[1:])[0]
    if len(uncovered) == 0:
        return max_check, covered
    else:
        return uncovered[0] + 1, covered

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    primes_all = _get_primes(1000)
    # Precompute 1/p as floats for quick budget checks, keep Fractions for final verification
    inv_primes_float = {p: 1.0 / p for p in primes_all}
    
    # Strategy: Greedy Gap-Closing
    # 1. Initialize with a set of small primes to get a good start.
    # Small primes cover more numbers per residue assignment.
    # We need sum(1/p) <= 1.
    # Sum of 1/p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    # Let's start with primes up to 50.
    
    initial_primes = [p for p in primes_all if p <= 50]
    # Assign residues randomly initially, or 0? 
    # Random is better for exploration.
    initial_residues = [random.randint(0, p - 1) for p in initial_primes]
    
    current_primes = list(initial_primes)
    current_residues = list(initial_residues)
    
    # State variables
    current_L, covered_arr = _compute_coverage(current_primes, current_residues, 100000)
    best_L = current_L
    best_primes = list(current_primes)
    best_residues = list(current_residues)
    
    current_budget_float = sum(inv_primes_float[p] for p in current_primes)
    
    # Available primes for addition (start with primes > 50)
    available_primes = [p for p in primes_all if p > 50]
    
    step = 0
    max_steps = 100000 # Safety cap
    
    while time.time() < deadline and step < max_steps:
        step += 1
        
        # 1. Identify the gap
        if current_L >= 100000:
            break
            
        gap = current_L
        # gap is the first uncovered number. We want to cover it.
        
        best_move_score = -1e9
        best_move_type = None
        best_move_params = None
        
        # Candidate Moves:
        # A. Change residue of an existing prime to cover the gap.
        #    Score = (Number of NEW numbers covered by this change in range [1, current_L + window]) - (Cost penalty)
        #    We want to cover the gap. If we change residue of p to gap % p, it covers gap.
        #    Does it cover other currently uncovered numbers?
        #    To make this fast, we only check if the move is "viable" (budget allows, prime distinct).
        #    We can estimate the benefit by checking a small window around the gap.
        
        window_size = 200 # Look ahead
        
        for i, p in enumerate(current_primes):
            r_new = gap % p
            if r_new == current_residues[i]:
                continue # No change
            
            # Check if this residue change is "good".
            # It covers gap. 
            # How many other numbers in [1, current_L + window_size] does it cover that were NOT covered before?
            # This is expensive to compute exactly for every candidate.
            # Approximation: 
            # If we change residue, we lose coverage of the old residue's numbers and gain coverage of new residue's numbers.
            # Net gain = (New numbers covered) - (Old numbers lost).
            # Since we are targeting the gap, and the gap is uncovered, we gain at least 1 (the gap).
            # We lose coverage of numbers that were ONLY covered by this prime with the old residue.
            
            # Simple heuristic: Prefer smaller primes (higher frequency).
            # Penalty for changing residue of a small prime is high if it was covering many unique numbers.
            
            # Let's just try the move and measure the new L? 
            # That is O(N) per candidate. N ~ 100 primes. 100 * O(100000) is too slow per step.
            
            # Better: Local evaluation.
            # Calculate the set of numbers covered by (p, old_r) in [1, current_L + window].
            # Calculate the set of numbers covered by (p, new_r) in [1, current_L + window].
            # Intersection with currently UNCOVERED numbers in that range.
            
            # This is still heavy. Let's use a simpler heuristic:
            # Score = 1/p (cost) - 100 * (1/p) (benefit proxy: smaller p is better) 
            # Actually, we want to minimize cost for covering the gap.
            # If we can cover the gap with an existing prime, we don't spend budget.
            # So "Change Residue" is always preferred over "Add Prime" if it doesn't destroy coverage elsewhere.
            
            # Let's evaluate "Change Residue" moves by actually simulating the coverage change in a small window.
            # We maintain `covered_arr`.
            # If we change p, r_old to p, r_new:
            # 1. Unmark all n in [1, current_L+window] such that n%p==r_old AND no other prime covers n.
            # 2. Mark all n in [1, current_L+window] such that n%p==r_new.
            # 3. Check if gap is covered.
            # 4. Check if any previously covered number in [1, current_L] became uncovered.
            
            # This is complex to do without a full re-scan.
            
            # Alternative Strategy: 
            # Just pick the smallest prime in the set that can cover the gap (i.e., gap % p != current_residue) 
            # and change its residue. 
            # This is a greedy "first fit" approach.
            
            # Let's refine: 
            # 1. Try to cover the gap using an EXISTING prime.
            #    Pick the prime p in current_primes that minimizes "disruption".
            #    Disruption is hard to measure. 
            #    Heuristic: Pick the largest prime in the set? 
            #    No, large primes cover sparse numbers. Changing them is less disruptive to the dense core coverage.
            #    Small primes cover dense numbers. Changing them might break coverage of nearby numbers.
            #    So, prefer changing the LARGEST prime in the current set to cover the gap.
            
            candidate_p = max(current_primes)
            r_new = gap % candidate_p
            
            if r_new != current_residues[current_primes.index(candidate_p)]:
                # Simulate this move
                idx = current_primes.index(candidate_p)
                old_r = current_residues[idx]
                current_residues[idx] = r_new
                
                new_L, new_covered = _compute_coverage(current_primes, current_residues, 100000)
                
                # Revert
                current_residues[idx] = old_r
                
                if new_L > current_L:
                    # Accept
                    current_residues[idx] = r_new
                    current_L = new_L
                    covered_arr = new_covered
                    if current_L > best_L:
                        best_L = current_L
                        best_primes = list(current_primes)
                        best_residues = list(current_residues)
                    continue # Next iteration, gap has moved
                
        # If we couldn't improve by changing an existing prime (or it didn't help enough),
        # we need to ADD a prime.
        
        # Select a new prime to add.
        # We want a prime p such that p <= gap (so it can cover gap efficiently? No, p can be anything).
        # Actually, if p > gap, it covers only one number in [1, gap] (the residue itself if r=gap).
        # If p < gap, it covers multiple numbers.
        # Smaller primes are better for coverage density.
        # But we are limited by budget.
        
        # Try adding the smallest available prime.
        if available_primes:
            # Sort available primes by size? 
            # Let's try the smallest available prime.
            p_new = min(available_primes)
            
            # Check budget
            if current_budget_float + inv_primes_float[p_new] <= 1.0:
                r_new = gap % p_new
                
                # Try adding it
                test_primes = current_primes + [p_new]
                test_residues = current_residues + [r_new]
                
                new_L, new_covered = _compute_coverage(test_primes, test_residues, 100000)
                
                if new_L > current_L:
                    # Accept
                    current_primes = test_primes
                    current_residues = test_residues
                    current_L = new_L
                    covered_arr = new_covered
                    current_budget_float += inv_primes_float[p_new]
                    available_primes.remove(p_new)
                    
                    if current_L > best_L:
                        best_L = current_L
                        best_primes = list(current_primes)
                        best_residues = list(current_residues)
                    continue
        
        # If we can't add a prime (budget full) and can't improve by changing residue,
        # we might need to REMOVE a prime to make room for a better one?
        # Or just accept that we are stuck.
        
        # To avoid infinite loops, if no move improved L, we should probably break or randomize.
        # Let's add a small randomization: if stuck for 100 steps, try a random residue change.
        
        # For simplicity in this structural change, if no greedy move works, we break.
        # The greedy approach is very deterministic. 
        # To make it more robust, let's allow "random" moves if greedy fails.
        
        # Check if we made progress in the last 100 steps? 
        # We don't track that easily here. 
        # Let's just break if we can't find ANY move that increases L.
        
        # Actually, the "Change Residue" logic above only tries the LARGEST prime.
        # Let's try ALL primes if the largest one fails.
        
        found_improvement = False
        for i, p in enumerate(current_primes):
            r_new = gap % p
            if r_new == current_residues[i]:
                continue
                
            old_r = current_residues[i]
            current_residues[i] = r_new
            new_L, new_covered = _compute_coverage(current_primes, current_residues, 100000)
            current_residues[i] = old_r
            
            if new_L > current_L:
                current_residues[i] = r_new
                current_L = new_L
                covered_arr = new_covered
                found_improvement = True
                if current_L > best_L:
                    best_L = current_L
                    best_primes = list(current_primes)
                    best_residues = list(current_residues)
                break # Move made, restart loop with new gap
        
        if not found_improvement:
            # Try adding a prime again, maybe a slightly larger one?
            # Or break.
            # Let's try adding a random available prime if budget allows.
            if available_primes and current_budget_float < 0.99:
                p_new = random.choice(available_primes)
                if current_budget_float + inv_primes_float[p_new] <= 1.0:
                    r_new = gap % p_new
                    test_primes = current_primes + [p_new]
                    test_residues = current_residues + [r_new]
                    new_L, new_covered = _compute_coverage(test_primes, test_residues, 100000)
                    if new_L > current_L:
                        current_primes = test_primes
                        current_residues = test_residues
                        current_L = new_L
                        covered_arr = new_covered
                        current_budget_float += inv_primes_float[p_new]
                        available_primes.remove(p_new)
                        if current_L > best_L:
                            best_L = current_L
                            best_primes = list(current_primes)
                            best_residues = list(current_residues)
                        continue
            
            # If still no improvement, break
            break

    # Final Verification
    best_primes_final = best_primes
    best_residues_final = best_residues
    
    # Exact budget check
    budget = sum(Fraction(1, p) for p in best_primes_final)
    if budget > 1:
        # Fallback to a safe set if we somehow exceeded budget (shouldn't happen with float check, but Fractions are exact)
        # Use the seed's set as fallback
        initial_primes_fallback = [p for p in primes_all if p >= 13]
        initial_residues_fallback = [(i + 1) % p for i, p in enumerate(initial_primes_fallback)]
        return [[p, r] for p, r in zip(initial_primes_fallback, initial_residues_fallback)]
    
    return [[p, r] for p, r in zip(best_primes_final, best_residues_final)]

# EVOLVE-BLOCK-END
