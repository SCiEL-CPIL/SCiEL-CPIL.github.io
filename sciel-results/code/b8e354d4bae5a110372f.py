import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    # Precompute prime exponent vectors for all numbers 2..128
    prime_list = primes
    num_primes = len(prime_list)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    
    # Exponent vector for each number, packed into an integer for fast hashing
    # We need to determine bits per prime.
    # Max value in product: 128^128 (if we used all). But max length is ~40.
    # 128^40 = (2^7)^40 = 2^280.
    # So max exponent for 2 is 280. 2^8 = 256, so 9 bits needed.
    # For larger primes, exponents are smaller.
    # Let's allocate 12 bits per prime to be safe. 30 primes * 12 bits = 360 bits.
    BITS_PER_PRIME = 12
    MASK = (1 << BITS_PER_PRIME) - 1
    
    packed_exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(prime_list):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        
        packed = 0
        for i, v in enumerate(vec):
            packed |= (v << (i * BITS_PER_PRIME))
        packed_exp_vecs[n] = packed

    def try_insertion(seq, c):
        """Try to insert c into seq maintaining strict increase and unique block products."""
        n = len(seq)
        
        # Find valid insertion positions
        valid_positions = []
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            valid_positions.append(i)
        
        if not valid_positions:
            return None, False
        
        # Sort positions by centrality
        mid = n / 2.0
        valid_positions.sort(key=lambda i: abs(i - mid))
        
        for i in valid_positions:
            new_seq = seq[:i] + [c] + seq[i:]
            m = len(new_seq)
            expected_count = m * (m + 1) // 2
            prods = set()
            valid = True
            for a in range(m):
                prod_val = 0
                for b in range(a, m):
                    prod_val += packed_exp_vecs[new_seq[b]]
                    if prod_val in prods:
                        valid = False
                        break
                    prods.add(prod_val)
                if not valid:
                    break
            if valid and len(prods) == expected_count:
                return new_seq, True
        
        return None, False

    def build_seq(seed_order):
        """Build a sequence starting with primes, then greedily add composites."""
        seq = primes[:]
        used = set(seq)
        remaining = [x for x in composites if x not in used]
        
        random.Random(seed_order).shuffle(remaining)
        
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for c in remaining[:]:
                new_seq, success = try_insertion(seq, c)
                if success:
                    seq = new_seq
                    remaining.remove(c)
                    progress = True
                    break
        return seq

    best_seq = primes[:]
    best_len = len(best_seq)
    
    seeds_used = 0
    while time.time() < deadline:
        seed_val = random.randint(0, 2**31 - 1)
        seeds_used += 1
        seq = build_seq(seed_val)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
            # Local search: try to add more
            improved = True
            while improved and time.time() < deadline:
                improved = False
                used = set(best_seq)
                remaining2 = [x for x in composites if x not in used]
                random.shuffle(remaining2)
                for c in remaining2:
                    new_seq, success = try_insertion(best_seq, c)
                    if success:
                        best_seq = new_seq
                        best_len = len(best_seq)
                        improved = True
                        break
            
            # Local search: try "remove and re-add" to escape local optima
            # This allows swapping out a "bad" element for two "good" ones
            removed_improved = True
            while removed_improved and time.time() < deadline:
                removed_improved = False
                # Try removing each element
                for idx in range(len(best_seq)):
                    # Skip if removing a prime? No, maybe removing a prime helps?
                    # But primes are very "safe". Let's try all.
                    orig_elem = best_seq[idx]
                    temp_seq = best_seq[:idx] + best_seq[idx+1:]
                    
                    # Try to add any remaining composite
                    used_temp = set(temp_seq)
                    candidates = [x for x in composites if x not in used_temp]
                    random.shuffle(candidates)
                    
                    found_better = False
                    for c in candidates:
                        new_seq, success = try_insertion(temp_seq, c)
                        if success:
                            # We removed 1, added 1. Length is same. 
                            # This is a swap. It might help future insertions.
                            # To actually increase length, we'd need to add 2.
                            # But try_insertion only adds 1.
                            # So we check if the new sequence allows further additions?
                            # That's expensive.
                            # Let's just accept the swap if it doesn't reduce length (it doesn't)
                            # and hope it opens up space.
                            # Actually, if we just swap, length is same.
                            # We should only keep it if it leads to a longer sequence later.
                            # For now, let's just try to add a SECOND element to temp_seq 
                            # using the new configuration.
                            
                            # Try to add another element to new_seq
                            used_new = set(new_seq)
                            candidates2 = [x for x in composites if x not in used_new]
                            random.shuffle(candidates2)
                            for c2 in candidates2:
                                new_seq2, success2 = try_insertion(new_seq, c2)
                                if success2:
                                    # We removed 1, added 2. Net gain +1.
                                    if len(new_seq2) > best_len:
                                        best_seq = new_seq2
                                        best_len = len(best_seq)
                                        removed_improved = True
                                        found_better = True
                                        break
                            if found_better:
                                break
                            
                    if found_better:
                        break
    
    return best_seq
