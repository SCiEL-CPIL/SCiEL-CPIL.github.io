import time
import random

def get_primes_composites(max_val):
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    composites = [i for i in range(2, max_val + 1) if not is_prime[i]]
    return primes, composites

def count_prime_factors(n):
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def is_valid(seq):
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def is_valid_insertion(seq, pos, val):
    """Check if inserting val at pos maintains all distinct block products."""
    n = len(seq)
    if pos > 0 and seq[pos-1] >= val:
        return False
    if pos < n and seq[pos] <= val:
        return False
    
    # New sequence would be seq[:pos] + [val] + seq[pos:]
    # We need to check all blocks that include the new element
    # Blocks not including val are already valid
    
    # Precompute products of suffixes starting from pos
    # and prefixes ending at pos-1
    
    # Check blocks that start before pos and end after pos-1 (include val)
    # A block [i, j] in new seq includes val iff i <= pos <= j (in new indexing)
    # In old seq terms: i <= pos-1 and j >= pos (or i == pos and j >= pos, etc.)
    
    # Let's just build the new seq and check efficiently
    new_seq = seq[:pos] + [val] + seq[pos:]
    
    # Only check blocks that include the new element at position pos
    # Block [i, j] includes pos iff i <= pos <= j
    products = set()
    
    # First, collect all products that don't include the new element (should be unique already)
    # But we need to check against ALL products, so let's be careful
    
    # Actually, let's just check all blocks that include pos
    for i in range(pos + 1):  # i can be 0..pos in new_seq
        prod = 1
        for j in range(i, pos + 1):  # j goes from i to at least pos
            if j < pos:
                prod *= new_seq[j]
            else:
                prod *= val
                # Now extend j beyond pos
                for k in range(pos + 1, len(new_seq)):
                    prod *= new_seq[k]
                    if prod in products:
                        return False
                    products.add(prod)
                break
        # Also need blocks that start before pos and end exactly at pos-1 or before
        # Wait, this is getting complex. Let me just do a full check but optimized.
    
    # Simpler approach: full check but with early termination
    products = set()
    m = len(new_seq)
    for i in range(m):
        prod = 1
        for j in range(i, m):
            prod *= new_seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def local_search(seq, rng, deadline):
    """Try to improve seq by swapping or replacing elements."""
    best = list(seq)
    best_len = len(seq)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        n = len(best)
        if n < 2:
            break
        
        # Try swapping two elements
        i, j = rng.sample(range(n), 2)
        if i > j:
            i, j = j, i
        new_seq = best[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        # Check strictly increasing
        valid = True
        for k in range(len(new_seq) - 1):
            if new_seq[k] >= new_seq[k+1]:
                valid = False
                break
        
        if valid and is_valid(new_seq):
            best = new_seq
        
        # Try replacing an element with a nearby value
        if n < 127:
            idx = rng.randint(0, n - 1)
            old_val = best[idx]
            # Try values in range [max(2, old_val-5), min(128, old_val+5)]
            candidates = []
            for v in range(max(2, old_val - 5), min(128, old_val + 5) + 1):
                if v == old_val:
                    continue
                if v in best:
                    continue
                candidates.append(v)
            
            if candidates:
                new_val = rng.choice(candidates)
                new_seq = best[:]
                new_seq[idx] = new_val
                
                # Check strictly increasing
                valid = True
                for k in range(len(new_seq) - 1):
                    if new_seq[k] >= new_seq[k+1]:
                        valid = False
                        break
                
                if valid and is_valid(new_seq):
                    best = new_seq
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes, composites = get_primes_composites(128)
    base_seq = primes
    
    best_seq = list(base_seq)
    best_len = len(base_seq)
    
    rng = random.Random(42)
    
    # Phase 1: Greedy insertion with multiple restarts
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
            
        current_seq = list(base_seq)
        # Try different orderings of composites
        comps = list(composites)
        
        # Use different strategies for different restarts
        if restart % 3 == 0:
            rng.shuffle(comps)
        elif restart % 3 == 1:
            comps.sort(key=lambda x: count_prime_factors(x))
            rng.shuffle(comps)
        else:
            comps.sort(reverse=True)
            rng.shuffle(comps)
        
        for c in comps:
            if time.time() > deadline - 5.0:
                break
            # Try inserting c at every valid position
            valid_positions = []
            n = len(current_seq)
            for i in range(n + 1):
                if i > 0 and current_seq[i-1] >= c:
                    continue
                if i < n and current_seq[i] <= c:
                    continue
                if is_valid_insertion(current_seq, i, c):
                    valid_positions.append(i)
            
            if valid_positions:
                # Pick a position - bias towards middle for diversity
                pos = rng.choice(valid_positions)
                current_seq = current_seq[:pos] + [c] + current_seq[pos:]
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
    
    # Phase 2: Local search on the best sequence
    if time.time() < deadline - 2.0:
        improved = local_search(best_seq, rng, deadline - 2.0)
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(best_seq)
    
    return best_seq
