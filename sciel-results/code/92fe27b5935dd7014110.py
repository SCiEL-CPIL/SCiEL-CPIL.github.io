import random
import math

def construct():
    n = 256
    candidates = list(range(2, n + 1))
    
    # Precompute prime factors or just use mod
    # Precompute bad_pairs_bitset for each a
    # Bit position for pair (b, c) with b < c is b * 257 + c
    # Max index: 256 * 257 + 256 = 65888
    # Python int handles this easily.
    
    bad_pairs_bitset = [0] * (n + 1)
    
    def is_bad_triple(a, b, c):
        if (b * c) % a == 0:
            return True
        if (a * c) % b == 0:
            return True
        if (a * b) % c == 0:
            return True
        return False

    # Precomputation
    for a in range(2, n + 1):
        bitset = 0
        for b in range(2, n + 1):
            for c in range(b + 1, n + 1):
                if is_bad_triple(a, b, c):
                    idx = b * 257 + c
                    bitset |= (1 << idx)
        bad_pairs_bitset[a] = bitset

    # Initial set: All primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(n)
    S = primes[:]
    S_set = set(S)
    
    # Build initial S_pairs bitset
    S_pairs = 0
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            b, c = S[i], S[j]
            if b > c: b, c = c, b
            idx = b * 257 + c
            S_pairs |= (1 << idx)

    best_S = S[:]
    best_size = len(S)
    
    # Simulated Annealing
    T = 10.0
    T_end = 0.01
    lambda_v = 50.0
    iterations = 20000
    decay = (T_end / T) ** (1.0 / iterations)
    
    import time
    start_time = time.time()
    
    for it in range(iterations):
        if time.time() - start_time > 50:
            break
            
        # Choose move type: 0: Add, 1: Remove
        move_type = random.choice([0, 1])
        
        if move_type == 0:
            # Add a random element not in S
            if len(S) >= n: continue
            # Pick random candidate not in S
            # Simple rejection sampling or pick from list
            # Since S is small, pick random from candidates and check
            a = random.choice(candidates)
            if a in S_set:
                continue
                
            # Calculate delta V
            # Violations added = popcount(S_pairs & bad_pairs_bitset[a])
            inter = S_pairs & bad_pairs_bitset[a]
            # Python 3.8+ has int.bit_count()
            delta_v = inter.bit_count()
            delta_size = 1
            delta_e = -1 + lambda_v * delta_v
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                # Accept
                # Update S_pairs
                new_bits = 0
                for b in S:
                    if b < a:
                        idx = b * 257 + a
                    else:
                        idx = a * 257 + b
                    new_bits |= (1 << idx)
                S_pairs |= new_bits
                S.append(a)
                S_set.add(a)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S[:]
                    
        else:
            # Remove a random element from S
            if len(S) <= 1: continue
            idx_rm = random.randint(0, len(S) - 1)
            a = S[idx_rm]
            
            # Calculate delta V
            # Violations removed = popcount(S_pairs & bad_pairs_bitset[a])
            inter = S_pairs & bad_pairs_bitset[a]
            delta_v = -inter.bit_count()
            delta_size = -1
            delta_e = 1 + lambda_v * delta_v
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                # Accept
                # Update S_pairs: clear bits involving a
                bits_to_clear = 0
                for b in S:
                    if b == a: continue
                    if b < a:
                        idx = b * 257 + a
                    else:
                        idx = a * 257 + b
                    bits_to_clear |= (1 << idx)
                S_pairs &= ~bits_to_clear
                
                S.pop(idx_rm)
                S_set.remove(a)
                
        T *= decay

    # Return best_S
    # Verify best_S is valid
    def is_valid_set(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                for k in range(j + 1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if is_bad_triple(a, b, c):
                        return False
        return True
        
    if not is_valid_set(best_S):
        # Fallback to primes if SA broke something (shouldn't happen if logic is correct, but safe)
        return primes
        
    return list(best_S)
