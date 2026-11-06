"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

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
    deadline = time.time() + 290
    random.seed(42)
    
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    prime_set = set(primes)
    
    # We need to check admissibility for primes p <= k, where k is the size of the set.
    # Since k <= N+1, we need primes up to N+1.
    
    # Initial state: Random subset or Greedy start?
    # A good start is better. Let's use the greedy removal from the seed as a starting point 
    # but then use SA to improve it. 
    # Actually, a random start with high density might be better for SA to explore, 
    # but valid start is safer. Let's start with a valid set from a simple heuristic.
    
    # Simple Heuristic Start: 
    # Start with all numbers. Remove classes for small primes.
    # This is what the seed does. Let's just use a random valid-ish start or the seed's result.
    # To save time and ensure a good baseline, I'll implement a fast greedy removal first.
    
    points = set(range(N + 1))
    for p in primes:
        if p > len(points):
            break
        residues = [set() for _ in range(p)]
        for x in points:
            residues[x % p].add(x)
        
        if all(len(r) > 0 for r in residues):
            # Find least populated
            min_idx = min(range(p), key=lambda i: len(residues[i]))
            points -= residues[min_idx]
            
    # Now we have a valid set. Let's convert to a numpy boolean array for fast operations.
    # mask[i] = 1 if i is in the set, 0 otherwise.
    mask = np.zeros(N + 1, dtype=np.int8)
    for x in points:
        mask[x] = 1
        
    k = int(np.sum(mask))
    
    # Precompute residue classes for each relevant prime as bitmasks or arrays?
    # For fast validity check of a move (add/remove x), we need to know:
    # If we add x:
    #   For each prime p <= k+1:
    #     If p was "full" (all residues covered) before? No, it was valid, so at least one empty.
    #     If x fills the LAST empty residue for p, it becomes invalid.
    # If we remove x:
    #   For each prime p <= k:
    #     If x was the ONLY element in its residue class for p, that class becomes empty (good, or stays good).
    #     Removing x never makes a prime "full" if it wasn't already. It can only create more empty classes.
    #     So removing x is ALWAYS safe regarding the "full" constraint, EXCEPT that k decreases, 
    #     which might make larger primes irrelevant (which is also safe).
    #     Wait, if we remove x, k decreases. The constraint applies to p <= k_new.
    #     Since k_new < k_old, the set of primes we need to check is a subset of the previous ones.
    #     And for those primes, removing an element cannot fill an empty class.
    #     So REMOVAL IS ALWAYS VALID.
    
    # This means we can freely remove elements. The only risk is in ADDING elements.
    # So the search space is: Start with a valid set. Try to add elements. 
    # If adding fails, try removing some elements to "make room" (create more empty residues) 
    # and then add different elements.
    
    # Better strategy: Simulated Annealing on the mask.
    # Move types:
    # 1. Add a random x not in mask. Check validity.
    # 2. Remove a random x in mask. Always valid.
    # 3. Swap: Remove x1, Add x2.
    
    # To check validity of adding x to current mask (size k):
    # We need to know, for each prime p <= k+1, if x % p is the unique empty residue.
    # Let's precompute counts of residues for each prime.
    # counts[p][r] = number of elements in mask with x % p == r.
    # If counts[p][r] == 0, it's empty.
    # If we add x, counts[p][x%p] increases by 1.
    # It becomes invalid if for some p <= k+1, counts[p][x%p] was 0 AND (p - sum(counts[p])) == 1?
    # No. If counts[p][r] == 0, and we add x with x%p == r, counts becomes 1.
    # The prime p is invalid if ALL residues have count > 0.
    # So p is currently valid means at least one residue has count 0.
    # If there is exactly ONE residue with count 0, and we add x that falls into it, then all residues become > 0. Invalid.
    # If there are >= 2 residues with count 0, adding x to one of them leaves at least one empty. Valid.
    
    # So, for each prime p, we need to know:
    # num_empty[p] = number of residues r with counts[p][r] == 0.
    # If num_empty[p] == 1, let last_empty_residue[p] be that residue.
    # Adding x is invalid for p if num_empty[p] == 1 and x % p == last_empty_residue[p].
    
    # We maintain:
    # counts: dict of prime -> array of counts (size p)
    # num_empty: dict of prime -> int
    # last_empty: dict of prime -> int (only valid if num_empty==1)
    
    # Initialize counts for current mask
    relevant_primes = [p for p in primes if p <= N + 1]
    counts = {}
    num_empty = {}
    last_empty = {}
    
    for p in relevant_primes:
        c = np.zeros(p, dtype=np.int32)
        # Efficient counting using numpy? 
        # For small p, loop is fine. For large p, vectorized.
        # N=5000 is small.
        for x in range(N + 1):
            if mask[x]:
                c[x % p] += 1
        counts[p] = c
        empties = np.where(c == 0)[0]
        num_empty[p] = len(empties)
        if len(empties) == 1:
            last_empty[p] = int(empties[0])
        else:
            last_empty[p] = -1
            
    def is_valid_add(x, current_k):
        new_k = current_k + 1
        # Check primes p <= new_k
        # We only care about primes p where num_empty[p] == 1 and x % p == last_empty[p]
        # Also, if new_k is a prime, we need to check it.
        # But our `relevant_primes` includes all primes up to N+1.
        # However, we only need to check p <= new_k.
        
        for p in relevant_primes:
            if p > new_k:
                break
            if num_empty[p] == 1:
                if x % p == last_empty[p]:
                    return False
        return True

    def apply_add(x, current_k):
        new_k = current_k + 1
        mask[x] = 1
        for p in relevant_primes:
            if p > new_k:
                break
            r = x % p
            counts[p][r] += 1
            if counts[p][r] == 1:
                num_empty[p] -= 1
                if num_empty[p] == 1:
                    # Find the remaining empty
                    # This is expensive. Optimize?
                    # We can track empties as a set?
                    # For now, recompute if it becomes 1.
                    empties = np.where(counts[p] == 0)[0]
                    last_empty[p] = int(empties[0]) if len(empties)==1 else -1
                else:
                    last_empty[p] = -1
        return new_k

    def apply_remove(x, current_k):
        new_k = current_k - 1
        mask[x] = 0
        for p in relevant_primes:
            if p > new_k:
                break
            r = x % p
            counts[p][r] -= 1
            if counts[p][r] == 0:
                num_empty[p] += 1
                if num_empty[p] == 1:
                    empties = np.where(counts[p] == 0)[0]
                    last_empty[p] = int(empties[0]) if len(empties)==1 else -1
                else:
                    last_empty[p] = -1
        return new_k

    # Simulated Annealing
    best_mask = mask.copy()
    best_k = k
    current_mask = mask.copy()
    current_k = k
    
    # We need to be careful with state management. 
    # It's easier to just try moves and revert if invalid/bad.
    
    temp = 10.0
    cooling = 0.99
    
    # List of candidates to add: all x not in mask
    # List of candidates to remove: all x in mask
    
    # To speed up, we can precompute lists? 
    # Or just sample randomly.
    
    start_time = time.time()
    steps = 0
    
    while time.time() < deadline - 5:
        # Choose a move
        # 70% Add, 30% Remove (to explore larger sets)
        if random.random() < 0.7:
            # Try to add
            # Sample random x not in current_mask
            x = random.randint(0, N)
            if current_mask[x] == 0:
                if is_valid_add(x, current_k):
                    # Accept add
                    new_k = apply_add(x, current_k)
                    if new_k > best_k:
                        best_k = new_k
                        best_mask = current_mask.copy()
                else:
                    # Invalid add. Maybe try removing something to make it valid?
                    # For now, skip or penalty.
                    pass
        else:
            # Try to remove
            # Sample random x in current_mask
            # Efficient sampling from set?
            # We can keep a list of active indices? 
            # Or just sample and check.
            x = random.randint(0, N)
            if current_mask[x] == 1:
                # Always valid
                new_k = apply_remove(x, current_k)
                # Since we removed, k decreased. This is bad for score, 
                # but might enable future adds.
                # In SA, we accept worse states with probability exp(-delta/E).
                # Here, "energy" is -k. So delta = -1.
                # We accept with prob exp(1/temp).
                # But removing reduces k, so we only do this if we expect it to help.
                # Actually, simple strategy: only remove if it allows an immediate better add?
                # Or just do random walk.
                
                # Let's try to add immediately after removing?
                # No, let's just do random walk.
                
        # Cooling
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
        steps += 1

    # Restore best
    # Actually, best_mask is a copy. We just need to return list(best_mask.nonzero()[0])
    # But wait, current_mask might be better than best_mask if we didn't update? 
    # We update best_mask whenever we improve.
    
    # One final check: maybe we can greedily add more?
    # The SA might have ended in a state that is not locally optimal.
    # Let's do a final greedy expansion on best_mask.
    
    final_mask = best_mask.copy()
    final_k = best_k
    
    # Rebuild counts for final_mask? 
    # Or just use the is_valid_add logic on a fresh count?
    # Since N is small, we can just do a linear scan for final expansion.
    
    # Reconstruct counts for final_k
    counts_final = {}
    num_empty_final = {}
    last_empty_final = {}
    for p in relevant_primes:
        if p > final_k + 1:
            continue
        c = np.zeros(p, dtype=np.int32)
        for x in range(N + 1):
            if final_mask[x]:
                c[x % p] += 1
        counts_final[p] = c
        empties = np.where(c == 0)[0]
        num_empty_final[p] = len(empties)
        if len(empties) == 1:
            last_empty_final[p] = int(empties[0])
        else:
            last_empty_final[p] = -1
            
    # Greedy Add
    for x in range(N + 1):
        if time.time() > deadline - 1:
            break
        if final_mask[x] == 0:
            # Check validity
            valid = True
            new_k = final_k + 1
            for p in relevant_primes:
                if p > new_k:
                    break
                if num_empty_final[p] == 1:
                    if x % p == last_empty_final[p]:
                        valid = False
                        break
            if valid:
                final_mask[x] = 1
                final_k += 1
                # Update counts
                for p in relevant_primes:
                    if p > final_k:
                        break
                    r = x % p
                    counts_final[p][r] += 1
                    if counts_final[p][r] == 1:
                        num_empty_final[p] -= 1
                        if num_empty_final[p] == 1:
                            empties = np.where(counts_final[p] == 0)[0]
                            last_empty_final[p] = int(empties[0]) if len(empties)==1 else -1
                        else:
                            last_empty_final[p] = -1
                            
    return list(np.nonzero(final_mask)[0])
# EVOLVE-BLOCK-END
