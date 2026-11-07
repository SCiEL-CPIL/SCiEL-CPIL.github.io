import time
import random
import math

def _validate(sums, x):
    """
    Check if x can be added to a set with existing subset sums 'sums'.
    Returns True if valid, False otherwise.
    """
    # x itself must not be in sums
    if x in sums:
        return False
    
    new_sums = set()
    for s in sums:
        ns = s + x
        # If ns is already in sums, we have a duplicate sum
        if ns in sums:
            return False
        new_sums.add(ns)
    new_sums.add(x)
    
    # Check divisibility between new_sums and existing sums
    # Note: We don't need to check sums vs sums again
    for ns in new_sums:
        for s in sums:
            # They are distinct because ns was not in sums
            if ns % s == 0 or s % ns == 0:
                return False
                
    # Check divisibility within new_sums
    # Sort to make it easier? Or just nested loop.
    # Number of new sums is len(sums) + 1.
    # If len(sums) is large, this is slow.
    # But for small sets (initial phase), it's fine.
    new_sums_list = list(new_sums)
    for i in range(len(new_sums_list)):
        a = new_sums_list[i]
        for j in range(i + 1, len(new_sums_list)):
            b = new_sums_list[j]
            if b % a == 0: # Since a != b, only one direction needed if we assume order? No, a could be larger.
                return False
            if a % b == 0:
                return False

    return True

def _add_to_sums(sums, x):
    """Update sums with new element x."""
    new_sums = set()
    for s in sums:
        new_sums.add(s + x)
    new_sums.add(x)
    sums.update(new_sums)
    return sums

def construct():
    deadline = time.time() + 85
    best_set = []
    best_size = 0
    
    # Generate primes in 1..256
    primes = []
    for i in range(2, 257):
        is_prime = True
        for p in range(2, int(math.isqrt(i)) + 1):
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    # Filter to larger primes first, as they are "safer"
    # Smaller primes create smaller sums which are more likely to divide larger sums.
    # E.g. sum=10, sum=20. 10|20.
    # If min sum is 100, max sum 5000, 100|200, 100|300, 100|400, 100|500.
    # Still risky.
    # Let's try to start with the largest primes.
    
    initial_candidates = sorted(primes, reverse=True)
    
    # Greedy construction from largest primes
    current_set = []
    sums = set()
    
    for p in initial_candidates:
        if time.time() > deadline:
            break
        if _validate(sums, p):
            current_set.append(p)
            _add_to_sums(sums, p)
            
    best_set = current_set[:]
    best_size = len(current_set)
    
    # Local Search to improve
    # We can try to add non-primes or smaller primes if we have space
    # Or swap to find better configurations
    
    # If we haven't used all primes, try adding the remaining ones in random order
    used = set(current_set)
    remaining = [p for p in primes if p not in used]
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Start from best found so far
        current_set = best_set[:]
        sums = set()
        for x in current_set:
            _add_to_sums(sums, x)
            
        # Try to add remaining candidates
        candidates = [x for x in range(1, 257) if x not in current_set]
        random.shuffle(candidates)
        
        improved = False
        for x in candidates:
            if time.time() > deadline:
                break
            if _validate(sums, x):
                current_set.append(x)
                _add_to_sums(sums, x)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    sums = set()
                    for y in best_set:
                        _add_to_sums(sums, y)
                    # Rebuild sums for best_set to be safe? 
                    # Actually, sums is already updated.
                    # But if we break out to restart, we need to be careful.
                    # Let's just keep going.
        
        # If we improved, we might want to try to improve further from this new best
        # But the loop structure handles this by restarting from best_set in next iteration?
        # No, the loop restarts from best_set at the beginning of the block?
        # Yes: "current_set = best_set[:]"
        
    return list(best_set)
