import time
import random

def is_valid(seq):
    # Check if all consecutive block products are distinct
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

def can_insert(seq, pos, val):
    """Check if inserting val at position pos in seq maintains validity.
    Only blocks that include the new element need to be checked against existing products.
    """
    n = len(seq)
    if pos > 0 and seq[pos-1] >= val:
        return False
    if pos < n and seq[pos] <= val:
        return False
    
    # Collect all existing products
    existing_prods = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            existing_prods.add(prod)
    
    # Check all blocks in new sequence that include val at position pos
    # new_seq = seq[0..pos-1], val, seq[pos..n-1]
    # Block [i, j] in new_seq includes val iff i <= pos <= j
    for i in range(0, pos + 1):
        prod = 1
        for k in range(i, pos):
            prod *= seq[k]
        prod *= val
        for j in range(pos, n):
            if prod in existing_prods:
                return False
            existing_prods.add(prod)
            prod *= seq[j]
    
    return True

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

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes, composites = get_primes_composites(128)
    base_seq = primes
    
    composites_sorted = sorted(composites, key=lambda x: count_prime_factors(x))
    
    best_seq = list(base_seq)
    best_len = len(base_seq)
    
    rng = random.Random(42)
    
    beam_width = 8
    beam = [list(base_seq)]
    
    for restart in range(300):
        if time.time() > deadline:
            break
            
        if len(beam) > 1:
            idx = rng.randrange(len(beam))
            current_seq = list(beam[idx])
        else:
            current_seq = list(base_seq)
        
        comps = list(composites)
        rng.shuffle(comps)
        
        for c in comps:
            if time.time() > deadline:
                break
            valid_positions = []
            n = len(current_seq)
            for i in range(n + 1):
                if i > 0 and current_seq[i-1] >= c:
                    continue
                if i < n and current_seq[i] <= c:
                    continue
                if can_insert(current_seq, i, c):
                    valid_positions.append(i)
            
            if valid_positions:
                pos = rng.choice(valid_positions)
                current_seq = current_seq[:pos] + [c] + current_seq[pos:]
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
        
        beam.append(list(current_seq))
        if len(beam) > beam_width:
            beam.sort(key=lambda s: -len(s))
            beam = beam[:beam_width]
            
    # Local search polish on best sequence
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            n = len(best_seq)
            for c in composites:
                if c in best_seq:
                    continue
                for i in range(n + 1):
                    if i > 0 and best_seq[i-1] >= c:
                        continue
                    if i < n and best_seq[i] <= c:
                        continue
                    if can_insert(best_seq, i, c):
                        best_seq = best_seq[:i] + [c] + best_seq[i:]
                        best_len += 1
                        improved = True
                        n = len(best_seq)
                        break
    
    return best_seq
