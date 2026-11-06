import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute prime exponent vectors for 2..128
    vecs = {}
    for x in range(2, 129):
        v = [0] * num_primes
        n = x
        for i, p in enumerate(primes):
            while n % p == 0:
                v[i] += 1
                n //= p
        vecs[x] = tuple(v)
        
    def add_vec(v1, v2):
        return tuple(a + b for a, b in zip(v1, v2))
        
    def greedy_search(order):
        seq = []
        prod_set = set()
        # We need to efficiently compute suffix products
        # Let's just recalculate from the end of seq for each candidate x
        # This is O(L) per candidate, which is fine.
        
        for x in order:
            if time.time() > deadline:
                break
                
            # Calculate new block products ending at x
            new_prods = []
            current_prod = vecs[x]
            new_prods.append(current_prod)
            
            # Multiply by suffixes of current seq
            # Suffixes are: seq[-1], seq[-2]*seq[-1], ...
            suffix_prod = vecs[x]
            for i in range(len(seq) - 1, -1, -1):
                suffix_prod = add_vec(suffix_prod, vecs[seq[i]])
                new_prods.append(suffix_prod)
                
            # Check for collisions
            valid = True
            for p in new_prods:
                if p in prod_set:
                    valid = False
                    break
                    
            if valid:
                seq.append(x)
                for p in new_prods:
                    prod_set.add(p)
                    
        return seq
        
    best_seq = []
    
    # 1. Try increasing order
    seq_inc = greedy_search(range(2, 129))
    if len(seq_inc) > len(best_seq):
        best_seq = seq_inc
        
    # 2. Try decreasing order
    seq_dec = greedy_search(range(128, 1, -1))
    if len(seq_dec) > len(best_seq):
        best_seq = seq_dec
        
    # 3. Try random orders
    # We have plenty of time, so we can do many random restarts
    while time.time() < deadline:
        order = list(range(2, 129))
        random.shuffle(order)
        seq_rand = greedy_search(order)
        if len(seq_rand) > len(best_seq):
            best_seq = seq_rand
            
    return best_seq
