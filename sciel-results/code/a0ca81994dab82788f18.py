"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    
    # 1. Start with the full set
    points = set(range(N + 1))
    primes = _primes(N + 1)
    
    # 2. Greedy Removal Phase
    # We need to ensure that for every prime p <= len(points), some residue class is missed.
    # We iterate through primes. If a prime p <= len(points) has all residues covered, we remove the least populated class.
    # To be efficient, we process primes in increasing order.
    # However, removing points changes len(points), so the set of relevant primes shrinks.
    # We will use a loop that checks for violations.
    
    # Optimization: Precompute residues? No, set is dynamic.
    # We will repeatedly scan for the smallest violating prime.
    
    while time.time() < deadline:
        k = len(points)
        # Find the smallest prime p <= k that violates the condition
        violating_p = None
        min_class_size = None
        min_residue = None
        
        for p in primes:
            if p > k:
                break
            
            # Count elements in each residue class mod p
            # Optimization: Only count if we haven't already determined it's safe?
            # We need to know if ALL classes are covered.
            # If any class is empty, it's safe.
            # We can stop counting as soon as we find an empty class.
            
            # Initialize counts to 0
            # Using a dict or list of size p
            counts = [0] * p
            
            # Iterate through points? That's O(k * p). 
            # k ~ 800, p ~ 800. 800*800 = 640,000 ops. 
            # Number of primes ~ 140. 
            # Total ops ~ 90M. Might be too slow in Python if done naively in a loop.
            
            # Better: 
            # For a fixed p, we can check coverage by looking at points % p.
            # If the set of residues has size < p, it's safe.
            
            residues = set()
            for x in points:
                residues.add(x % p)
                if len(residues) == p:
                    break # All covered, violation found
            
            if len(residues) == p:
                violating_p = p
                # Now find the least populated class
                # We need counts again
                counts = [0] * p
                for x in points:
                    counts[x % p] += 1
                
                min_class_size = min(counts)
                min_residue = counts.index(min_class_size)
                break
        
        if violating_p is None:
            # No violations, we are admissible
            break
        
        # Remove all x in points such that x % violating_p == min_residue
        to_remove = {x for x in points if x % violating_p == min_residue}
        points -= to_remove
        
        # Check deadline
        if time.time() > deadline:
            break

    # 3. Expansion Phase
    # Try to add back elements that were removed, or elements never in the set (none, since we started with all).
    # Actually, we started with all, so "removed" elements are candidates.
    # But we don't track them explicitly in a list easily if we just used a set difference.
    # We can reconstruct candidates: range(N+1) - points
    
    # For each candidate x, check if it can be added.
    # Condition: For all p <= len(points) + 1, adding x does not make p fully covered 
    # (i.e., p must have had >1 empty classes, or x falls into an already non-empty class).
    
    # To make this efficient, we precompute the "empty classes" status for current points.
    # But len(points) changes as we add.
    
    # Simpler: Just try to add all candidates one by one, checking validity.
    # Sort candidates? Maybe by value?
    candidates = sorted(set(range(N + 1)) - points)
    
    # We can batch add? No, adding one might block another.
    # Greedy add: Iterate through candidates. If valid, add it.
    
    # To speed up validity check:
    # Maintain a list of "critical primes" for the current set size.
    
    # Let's just do a simple loop with a time check.
    # The number of candidates is N - len(points) ~ 4200.
    # Validity check for one x:
    #   k = len(points)
    #   for p in primes <= k+1:
    #       if x % p is the ONLY empty residue for p in current points:
    #           return False
    #   return True
    
    # We need to know the empty residues for each p.
    # Let's precompute for the current set `points` the empty residues for all p <= N.
    # empty_residues[p] = set of residues r such that no x in points has x%p == r
    
    # This precomputation is O(N * pi(N)) ~ 5000 * 669 ~ 3.3M ops. Fast.
    
    def compute_empty_residues(pts, primes_list):
        empties = {}
        for p in primes_list:
            if p > len(pts):
                # If p > len(pts), it's impossible to cover all p classes, 
                # so there is always at least one empty class. 
                # But for the "add" logic, we need to know if adding x fills the LAST empty class.
                # If p > len(pts), there are at least p - len(pts) empty classes.
                # If we add x, empty classes decrease by 1 (if x was in an empty class) or stay same.
                # It can never become 0 empty classes because p - len(pts) >= 1, 
                # and we are adding 1 element, so new size is len(pts)+1.
                # If len(pts)+1 < p, still impossible to cover all.
                # So for p > len(pts)+1, it's always safe to add w.r.t p.
                # For p == len(pts)+1, if we add x, size becomes p. 
                # Then we MUST have exactly 1 empty class remaining? No, we must have AT LEAST 1 empty class.
                # So if p == len(pts)+1, we need to ensure that after adding x, not all p classes are covered.
                # This means x must NOT fill the last empty class if there was only 1 empty class.
                # Actually, if current size is p-1, there are at least 1 empty class.
                # If there is exactly 1 empty class, and x falls in it, then new size p covers all classes -> Invalid.
                # If there are >= 2 empty classes, adding x leaves >= 1 empty -> Valid.
                
                # So we need to know the count of empty classes for p.
                pass
            
            residues_present = set()
            for x in pts:
                residues_present.add(x % p)
            all_res = set(range(p))
            empties[p] = all_res - residues_present
        return empties

    # The expansion phase might be slow if we recompute empties for every addition.
    # But we only add a few elements? Or many?
    # The gap between greedy result and optimal might be small.
    
    # Let's try a simpler expansion:
    # Just check candidates against the current set's empty residues.
    
    # Recompute empties for current points
    # We only care about primes p <= len(points) + 1
    
    k = len(points)
    relevant_primes = [p for p in primes if p <= k + 1]
    
    empties = compute_empty_residues(points, relevant_primes)
    
    # Now iterate through candidates
    # We will add them one by one.
    # To avoid O(N^2), we can just iterate and check.
    # Since N=5000, and we might add ~100-500 elements, 500 * 140 primes * 5000 checks? No.
    # Check for x:
    #   for p in relevant_primes:
    #       if x % p in empties[p]:
    #           if len(empties[p]) == 1:
    #               return False # x fills the last hole
    #   return True
    
    # This is O(pi(k)) per candidate. ~100 ops.
    # 4000 candidates * 100 ops = 400,000 ops. Very fast.
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        valid = True
        new_k = len(points) + 1
        # Primes up to new_k
        for p in relevant_primes:
            if p > new_k:
                break
            r = x % p
            if r in empties[p]:
                if len(empties[p]) == 1:
                    valid = False
                    break
        # Also need to consider primes p where p > k+1? 
        # If p > k+1, then len(points)+1 < p, so it's impossible to cover all p classes.
        # So they are always safe.
        
        if valid:
            points.add(x)
            # Update empties for primes affected by x
            # x % p changes empties[p]
            # For all p in relevant_primes (and potentially new primes if k increased):
            # Actually, k increased by 1. 
            # New relevant primes: if new_k is prime, add it.
            if new_k in _primes(new_k): # Check if new_k is prime? 
                 # We don't have a set of primes for O(1) lookup easily without precomp
                 pass
            # Simpler: Just recompute empties if we added a lot? 
            # Or update incrementally.
            # Incremental update:
            for p in relevant_primes:
                if p > new_k:
                    continue
                r = x % p
                if r in empties[p]:
                    empties[p].remove(r)
                    # If empties[p] becomes empty, we have a violation!
                    # But we checked valid=True, so this shouldn't happen 
                    # UNLESS our logic for "len(empties)==1" was wrong?
                    # If len(empties)==1 and r in empties, valid=False.
                    # So if valid=True, either r not in empties, or len(empties)>1.
                    # So empties[p] will not become empty.
            
            # If new_k is a prime, we need to add it to relevant_primes and compute its empties
            if new_k > k:
                # Check if new_k is prime
                # We can just check if new_k is in the primes list
                if new_k in primes:
                    relevant_primes.append(new_k)
                    # Compute empties for this new prime
                    residues_present = set()
                    for x_pt in points:
                        residues_present.add(x_pt % new_k)
                    all_res = set(range(new_k))
                    empties[new_k] = all_res - residues_present
    
    return list(points)
# EVOLVE-BLOCK-END
