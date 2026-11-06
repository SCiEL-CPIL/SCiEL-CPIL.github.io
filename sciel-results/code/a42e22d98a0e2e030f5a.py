import time
from fractions import Fraction
from functools import lru_cache

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Generate all primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(1000)
    prime_set = set(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Precompute 1/p as Fraction
    recip = {p: Fraction(1, p) for p in primes}
    
    def covered_prefix(classes, max_check=100000):
        """Find largest L such that every integer 1..L is in some class."""
        if not classes:
            return 0
        # Build a coverage array up to max_check
        covered = [False] * (max_check + 1)
        for p, r in classes:
            # Mark r, r+p, r+2p, ... up to max_check
            # r can be 0, meaning multiples of p
            start = r if r > 0 else p  # 1..max_check, so if r==0, first is p
            if start > max_check:
                continue
            for x in range(start, max_check + 1, p):
                covered[x] = True
        # Find the gap
        L = 0
        for i in range(1, max_check + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    def budget_ok(classes):
        """Check if sum of 1/p <= 1 using exact rationals."""
        total = Fraction(0, 1)
        for p, r in classes:
            total += recip[p]
        return total <= 1
    
    def current_budget(classes):
        total = Fraction(0, 1)
        for p, r in classes:
            total += recip[p]
        return total
    
    # Greedy construction: try to build a covering system
    # Start with a reasonable seed, then improve
    
    # Initial greedy: use the seed approach but optimize residues
    # The seed uses p_i (i-th prime from 13) covering integer i+1
    # Let's start with that and improve
    
    # Build initial solution: primes from 13 upward, each covering a specific integer
    # Prime list starting from 13:
    primes_from_13 = [p for p in primes if p >= 13]
    
    # For each prime, assign residue to cover a specific small integer
    # Greedy: for prime p at position i, cover integer (i+1)
    # But we need to be smarter about which integers to cover
    
    # Better greedy: iteratively add primes that cover the smallest uncovered integer
    # while staying within budget
    
    best_L = 0
    best_classes = []
    
    # Start with the seed solution
    seed_classes = []
    for i, p in enumerate(primes_from_13):
        r = (i + 1) % p
        seed_classes.append([p, r])
    
    if budget_ok(seed_classes):
        L = covered_prefix(seed_classes)
        if L > best_L:
            best_L = L
            best_classes = [c[:] for c in seed_classes]
    
    # Local search: try to improve by swapping primes or adjusting residues
    # A swap: remove one prime, add another (different prime), adjust residue
    
    # Also try: for each prime in the solution, try different residues to extend coverage
    # And try: remove a prime that doesn't help much, add a smaller prime (cheaper budget)
    
    # Key insight: we want to cover 1, 2, 3, ... consecutively
    # For each integer n, we need some prime p with n ≡ r (mod p)
    # i.e., r = n mod p
    
    # Strategy: for the current set of primes, for each prime, the best residue
    # is the one that covers the most uncovered small integers
    
    def optimize_residues(classes, primes_list, max_check=5000):
        """For a fixed set of primes, find residues to maximize covered prefix."""
        # This is a constraint satisfaction / optimization problem
        # Greedy: for each prime, pick the residue that covers the most currently-uncovered integers
        # in the prefix region
        
        n_primes = len(primes_list)
        residues = [0] * n_primes
        
        # We want to cover 1, 2, 3, ... as much as possible
        # Greedy assignment: process primes, for each pick residue maximizing new coverage
        # in the range 1..max_check
        
        # Actually, let's think about it differently.
        # For a given set of primes, the covered set is determined by residues.
        # We want to maximize the prefix.
        
        # Simple approach: for each prime, try all possible residues and pick the best
        # This is O(n_primes * p * max_check) which could be slow
        # But p <= 1000 and n_primes is at most ~168 (primes up to 1000)
        # Let's limit max_check for speed
        
        # Better: for each prime, the residue that covers integer k is k mod p
        # So if we want to cover a specific set of integers, we can determine residues
        
        # Greedy: target covering integers 1, 2, 3, ...
        # For each integer n starting from 1, if not covered, assign a prime to cover it
        # Pick the prime (from our set) that has the smallest p (cheapest) and assign r = n mod p
        
        # But we have fixed primes. Let's just try: for each prime, assign residue to cover
        # the smallest uncovered integer that is congruent to something mod p
        
        # Simpler: just try all residues for each prime and greedily pick best
        covered = [False] * (max_check + 1)
        
        # Process primes in order of increasing p (smaller p = cheaper, covers more densely)
        # Actually, let's process in the order given
        for idx, p in enumerate(primes_list):
            best_r = 0
            best_gain = -1
            # Try each residue
            for r in range(p):
                # Count how many new integers in 1..max_check get covered
                # that are not already covered
                gain = 0
                start = r if r > 0 else p
                if start > max_check:
                    continue
                for x in range(start, max_check + 1, p):
                    if not covered[x]:
                        gain += 1
                # Also weight by how early in the prefix
                # Actually, just maximize gain for now
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[idx] = best_r
            # Update covered
            start = best_r if best_r > 0 else p
            for x in range(start, max_check + 1, p):
                covered[x] = True
        
        return residues
    
    # Let's try a different approach: direct search for good residue assignments
    # Given a set of primes, find residues to maximize prefix coverage
    
    def find_best_residues(primes_list, max_check=20000):
        """Find residues for given primes to maximize covered prefix."""
        if not primes_list:
            return []
        
        # Greedy: for each integer from 1 upward, if not covered,
        # assign a prime to cover it (pick the prime with smallest p that can cover it)
        # A prime p can cover integer n with residue r = n % p
        
        residues = {}
        covered = [False] * (max_check + 1)
        
        for n in range(1, max_check + 1):
            if covered[n]:
                continue
            # Find a prime in primes_list that can cover n
            # All primes can cover any n (just use r = n % p)
            # Pick the one with smallest p (most efficient)
            best_p = None
            best_p_val = float('inf')
            for p in primes_list:
                if p < best_p_val:
                    best_p_val = p
                    best_p = p
            if best_p is None:
                break
            r = n % best_p
            residues[best_p] = r
            # Mark all multiples
            start = r if r > 0 else best_p
            for x in range(start, max_check + 1, best_p):
                covered[x] = True
        
        # Build result in order of primes_list
        result = []
        for p in primes_list:
            if p in residues:
                result.append([p, residues[p]])
            else:
                result.append([p, 0])
        return result
    
    # Now the main search:
    # 1. Start with seed
    # 2. Try to add/remove/swap primes to improve prefix
    
    # First, let's try to find a good set of primes
    # The budget is sum(1/p) <= 1
    # We want primes that together can cover 1..L for large L
    
    # Key: small primes are more efficient (cover more integers per unit budget)
    # But we need distinct primes
    
    # Let's try: use the smallest primes possible, but we need enough of them
    # to cover all residues mod their product... actually no, we just need prefix coverage
    
    # Greedy prime selection:
    # Start with empty set, budget = 0
    # Repeatedly add the prime that maximizes the covered prefix (with optimal residues)
    # subject to budget constraint
    
    # This is expensive. Let's do a simpler version:
    # Start with seed, then try swaps
    
    # Try to improve the seed solution
    current = [c[:] for c in seed_classes]
    current_L = best_L
    current_budget = current_budget(current)
    
    # Precompute for speed: which primes are in current
    current_primes = set(p for p, r in current)
    
    # Local search: try swapping one prime for another
    # For each prime in current, try removing it and adding a different prime
    # Check if budget is still ok, and if prefix improves
    
    # Also try: just changing residues of existing primes
    
    # Let's first try optimizing residues of the current set
    primes_in_current = [p for p, r in current]
    
    # Try to find better residues
    # For each prime, the residue determines which integers it covers
    # We want to maximize the prefix
    
    # Simple local search on residues:
    # For each prime, try changing its residue to see if prefix improves
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx, (p, r) in enumerate(current):
            # Try all residues for this prime
            best_r = r
            best_L_for_r = current_L
            # We only need to check residues that might help
            # The current residue covers certain integers
            # Try residues that cover the first uncovered integer
            # Find first uncovered integer
            # Actually, let's just try a few promising residues
            # The residue that covers integer n is n % p
            # Find the first uncovered integer in the prefix
            # For efficiency, let's check: what's the first gap?
            
            # Build coverage for current
            max_check = min(current_L + 500, 20000)
            covered = [False] * (max_check + 1)
            for p2, r2 in current:
                start = r2 if r2 > 0 else p2
                if start <= max_check:
                    for x in range(start, max_check + 1, p2):
                        covered[x] = True
            
            # Find first gap
            first_gap = None
            for i in range(1, max_check + 1):
                if not covered[i]:
                    first_gap = i
                    break
            
            if first_gap is None:
                # All covered up to max_check
                continue
            
            # Try residues that cover first_gap
            target = first_gap
            # For prime p, residue to cover target is target % p
            new_r = target % p
            if new_r != r:
                # Try this new residue
                old_r = current[idx][1]
                current[idx][1] = new_r
                # Check if this actually helps (recompute prefix)
                L_new = covered_prefix(current, max_check=min(max_check + 500, 50000))
                if L_new > current_L:
                    current_L = L_new
                    best_L = max(best_L, current_L)
                    best_classes = [c[:] for c in current]
                    improved = True
                else:
                    current[idx][1] = old_r
    
    # Now try swapping primes
    # For each prime in current, try removing it and adding a different prime
    # The new prime should be one that helps extend the prefix
    
    # Get list of all primes not in current
    all_primes = primes  # up to 1000
    
    # Try adding a prime (if budget allows)
    # Or swap: remove one, add another
    
    # First, try just adding a prime if there's budget
    remaining_budget = Fraction(1, 1) - current_budget
    
    # Sort candidate primes by how much they might help
    # A prime p helps if it can cover some uncovered small integer
    # Smaller p covers more densely
    
    # Let's try: for each prime not in current, if we can afford it,
    # add it with the best residue and see if prefix improves
    
    # This is O(n_primes * cost_of_prefix_check)
    # Let's limit the number of trials
    
    trial_count = 0
    max_trials = 5000
    
    # Try adding primes
    for p in sorted(all_primes, key=lambda x: x):  # smallest first
        if p in current_primes:
            continue
        if recip[p] > remaining_budget:
            continue
        # Try adding p with various residues
        # Best residue: cover the first uncovered integer
        max_check = min(current_L + 1000, 30000)
        covered = [False] * (max_check + 1)
        for p2, r2 in current:
            start = r2 if r2 > 0 else p2
            if start <= max_check:
                for x in range(start, max_check + 1, p2):
                    covered[x] = True
        
        first_gap = None
        for i in range(1, max_check + 1):
            if not covered[i]:
                first_gap = i
                break
        
        if first_gap is None:
            break
        
        r = first_gap % p
        new_classes = current + [[p, r]]
        L_new = covered_prefix(new_classes, max_check=min(max_check + 1000, 50000))
        if L_new > current_L:
            current = new_classes
            current_primes.add(p)
            current_budget += recip[p]
            current_L = L_new
            best_L = max(best_L, current_L)
            best_classes = [c[:] for c in current]
            remaining_budget = Fraction(1, 1) - current_budget
            trial_count += 1
            if trial_count >= max_trials or time.time() > deadline:
                break
        else:
            trial_count += 1
            if trial_count >= max_trials or time.time() > deadline:
                break
    
    # Now try swaps: remove a prime, add a different one
    # Focus on removing larger primes (more budget) and adding smaller ones (cheaper)
    
    swap_improved = True
    swap_iter = 0
    while swap_improved and time.time() < deadline and swap_iter < 50:
        swap_improved = False
        swap_iter += 1
        
        # Try removing each prime and see if we can add a better one
        for idx in range(len(current)):
            if time.time() > deadline:
                break
            p_remove, r_remove = current[idx]
            # Remove it
            new_current = current[:idx] + current[idx+1:]
            new_budget = current_budget - recip[p_remove]
            new_remaining = Fraction(1, 1) - new_budget
            
            # Try adding each candidate prime
            for p_add in sorted(all_primes, key=lambda x: x):
                if p_add == p_remove:
                    continue
                if p_add in set(p for p, r in new_current):
                    continue
                if recip[p_add] > new_remaining:
                    continue
                # Try with best residue
                max_check = min(current_L + 1000, 30000)
                covered = [False] * (max_check + 1)
                for p2, r2 in new_current:
                    start = r2 if r2 > 0 else p2
                    if start <= max_check:
                        for x in range(start, max_check + 1, p2):
                            covered[x] = True
                
                first_gap = None
                for i in range(1, max_check + 1):
                    if not covered[i]:
                        first_gap = i
                        break
                
                if first_gap is None:
                    break
                
                r_add = first_gap % p_add
                candidate = new_current + [[p_add, r_add]]
                L_new = covered_prefix(candidate, max_check=min(max_check + 1000, 50000))
                if L_new > current_L:
                    current = candidate
                    current_budget = new_budget + recip[p_add]
                    current_L = L_new
                    best_L = max(best_L, current_L)
                    best_classes = [c[:] for c in current]
                    current_primes = set(p for p, r in current)
                    swap_improved = True
                    break  # restart the outer loop
                elif L_new > best_L:
                    # Even if not better than current_L, might be better than best
                    pass
    
    # Final: try to optimize residues one more time with a wider search
    if time.time() < deadline:
        # For each prime, try the residue that covers the first gap
        for _ in range(10):  # a few iterations
            if time.time() > deadline:
                break
            max_check = min(current_L + 2000, 50000)
            covered = [False] * (max_check + 1)
            for p2, r2 in current:
                start = r2 if r2 > 0 else p2
                if start <= max_check:
                    for x in range(start, max_check + 1, p2):
                        covered[x] = True
            
            first_gap = None
            for i in range(1, max_check + 1):
                if not covered[i]:
                    first_gap = i
                    break
            
            if first_gap is None:
                break
            
            # Try to cover first_gap by adjusting one prime's residue
            found = False
            for idx, (p, r) in enumerate(current):
                new_r = first_gap % p
                if new_r == r:
                    continue
                old_r = current[idx][1]
                current[idx][1] = new_r
                L_new = covered_prefix(current, max_check=min(max_check + 2000, 80000))
                if L_new > current_L:
                    current_L = L_new
                    best_L = max(best_L, current_L)
                    best_classes = [c[:] for c in current]
                    found = True
                    break
                else:
                    current[idx][1] = old_r
            if not found:
                break
    
    # Return the best solution found
    if best_classes:
        return best_classes
    return seed_classes
