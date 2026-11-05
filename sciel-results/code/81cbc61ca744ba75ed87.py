import time
import random
import math
from collections import defaultdict

N = 5000
START_TIME = time.time()
TIME_LIMIT = 325.0  # Keep margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

# Precompute primes up to N
PRIMES = _primes(N)
# Primes that could ever be relevant: p <= N
RELEVANT_PRIMES = [p for p in PRIMES if p <= N]

def _is_admissible(points, primes_list):
    """
    Check if the set of points is admissible.
    Admissible: for every prime p <= len(points), some residue class mod p is missed.
    """
    k = len(points)
    if k == 0:
        return True
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return False
    return True

def _find_violating_prime(points, primes_list):
    """Find a prime p <= len(points) where all residues are covered, or None if admissible."""
    k = len(points)
    if k == 0:
        return None
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return p
    return None

def _compute_residue_counts(points, primes_list, max_prime):
    """Compute residue counts for all primes up to max_prime."""
    counts = {}
    for p in primes_list:
        if p > max_prime:
            break
        cnt = [0] * p
        for v in points:
            cnt[v % p] += 1
        counts[p] = cnt
    return counts

def _can_add_candidate(x, current_set, residue_counts, primes_list, new_size):
    """
    Check if adding x to current_set keeps it admissible using precomputed counts.
    current_set is assumed admissible.
    """
    for p in primes_list:
        if p > new_size:
            break
        cnt = residue_counts[p]
        r = x % p
        # If count of residue r is 0, adding x makes it 1.
        # If count is > 0, it stays > 0.
        # The set becomes non-admissible iff for some p, all residues 0..p-1 have count >= 1.
        # Since current_set is admissible, for each p <= len(current_set), at least one residue has count 0.
        # Adding x can only increase one residue's count.
        # So it becomes non-admissible iff:
        #   1. p <= new_size
        #   2. Before adding, exactly one residue had count 0 (say r0)
        #   3. We are adding x such that x % p == r0
        #   4. After adding, all residues have count >= 1.
        
        # Optimization: precompute for each p, the list of residues with count 0.
        # But we don't have that here directly.
        # Let's just check: does adding x fill the last empty slot for any p?
        
        # Count how many residues have count 0 in current_set for this p.
        # Actually, we can just check if after adding, all are covered.
        # But that's O(p) per prime.
        
        # Faster check:
        # For this p, let zero_residues = [r for r in range(p) if cnt[r] == 0]
        # If len(zero_residues) > 1, adding x (which hits at most one residue) cannot fill all.
        # If len(zero_residues) == 1, let r0 = zero_residues[0]. If x % p == r0, then it fills it.
        #   Then we must check if all other residues have count >= 1. Since only r0 was 0, and now it's 1, yes.
        #   So it becomes non-admissible.
        # If len(zero_residues) == 0, current_set was already non-admissible (shouldn't happen).
        
        # So we need to know the zero residues.
        # Let's compute zero residues on the fly or precompute.
        # Precomputing zero residues for all p is cheap.
        
        # Actually, let's just do the simple check:
        # If cnt[r] == 0 and there are no other zero residues, then adding x makes it non-admissible.
        # How to check "no other zero residues"?
        # Sum of cnt is len(current_set).
        # Number of non-zero residues = sum(1 for c in cnt if c > 0).
        # If sum(1 for c in cnt if c > 0) == p - 1 and cnt[r] == 0, then adding x makes it p non-zero.
        
        # This is O(p) per prime.
        # Total cost per candidate: sum of p for p <= new_size.
        # This might be too slow if we check many candidates.
        
        # Alternative: Precompute for each p the list of zero residues.
        # Let's modify the approach: pass in zero_residues dict.
        
        # For now, let's just use the O(p) check but be smart.
        # Actually, the previous program did a full check. Let's try to optimize.
        
        # Let's precompute zero_residues outside this function.
        pass

def construct():
    global START_TIME
    start_time = START_TIME
    
    best_set = set()
    best_size = 0
    
    num_restarts = 500  # More restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Start with all numbers
        current_set = set(range(N))
        
        # Greedy Repair Loop
        max_iterations = 10000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            k = len(current_set)
            if k == 0:
                break
            
            # Find a violating prime
            violating_p = None
            for p in PRIMES:
                if p > k:
                    break
                # Check if all residues mod p are covered
                seen = bytearray(p)
                count = 0
                for v in current_set:
                    r = v % p
                    if not seen[r]:
                        seen[r] = 1
                        count += 1
                        if count == p:
                            violating_p = p
                            break
                if violating_p:
                    break
            
            if violating_p is None:
                # Admissible
                break
            
            # We need to remove elements to break the coverage for violating_p
            # Group elements by residue class mod violating_p
            classes = [[] for _ in range(violating_p)]
            for v in current_set:
                classes[v % violating_p].append(v)
            
            # Find the class with the minimum number of elements
            class_sizes = [(len(c), i) for i, c in enumerate(classes)]
            class_sizes.sort()
            
            # Randomly pick from the top few smallest classes for diversity
            if random.random() < 0.7:
                min_idx = class_sizes[0][1]
            else:
                top_k = min(3, len(class_sizes))
                min_idx = class_sizes[random.randint(0, top_k - 1)][1]
            
            # Remove all elements in this class
            to_remove = set(classes[min_idx])
            current_set -= to_remove
        
        # Now current_set is admissible (or empty)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        
        # Try to add back elements to improve size
        # This is a local search: try adding any element from [0, N) not in current_set
        # If it remains admissible, keep it.
        
        # Precompute residue counts and zero residues for efficient add-check
        k = len(current_set)
        residue_counts = {}
        zero_residues = {}
        for p in PRIMES:
            if p > k:
                break
            cnt = [0] * p
            for v in current_set:
                cnt[v % p] += 1
            residue_counts[p] = cnt
            zero_residues[p] = [r for r in range(p) if cnt[r] == 0]
        
        max_add_passes = 10
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            # Get candidates: elements not in current_set
            candidates = [x for x in range(N) if x not in current_set]
            random.shuffle(candidates)
            
            # Limit candidates to check to avoid timeout
            check_limit = 2000
            added_any = False
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 3:
                    break
                
                # Check if current_set U {x} is admissible
                new_size = len(current_set) + 1
                is_adm = True
                for p in PRIMES:
                    if p > new_size:
                        break
                    r = x % p
                    # If x falls into a zero residue, and that was the only zero residue, it becomes non-admissible
                    if r in zero_residues[p]:
                        if len(zero_residues[p]) == 1:
                            is_adm = False
                            break
                
                if is_adm:
                    current_set.add(x)
                    added_any = True
                    # Update residue counts and zero residues
                    new_k = len(current_set)
                    for p in PRIMES:
                        if p > new_k:
                            break
                        r = x % p
                        residue_counts[p][r] += 1
                        if r in zero_residues[p]:
                            zero_residues[p].remove(r)
                    # If new_k is a prime, we need to add it to our tracking
                    # But PRIMES is fixed. We just need to make sure we have counts for it.
                    # Actually, if new_k is prime, we need to compute its counts.
                    # This is a bit messy. Let's just recompute if new_k is prime.
                    if new_k in PRIMES:
                        cnt = [0] * new_k
                        for v in current_set:
                            cnt[v % new_k] += 1
                        residue_counts[new_k] = cnt
                        zero_residues[new_k] = [r for r in range(new_k) if cnt[r] == 0]
            
            # If we added none, we are likely at a local maximum
            if not added_any:
                break
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
    
    return list(best_set)

# Initial call to set start time and warm up
construct()
