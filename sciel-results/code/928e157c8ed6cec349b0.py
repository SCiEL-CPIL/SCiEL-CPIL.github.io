import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0
    candidates = list(range(2, 41))
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute exponent vectors for each candidate
    def get_vec(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
        
    vecs = {n: get_vec(n) for n in candidates}
    
    # Initial best set: all primes
    best_set = primes[:]
    best_size = len(best_set)
    
    # Faster check for adding ONE element to an existing set of vectors
    def check_add_single(current_vectors, x):
        xv = vecs[x]
        # Check collision: for all v in current_vectors, (v + xv) NOT IN current_vectors
        for v in current_vectors:
            new_v = tuple(v[i] + xv[i] for i in range(num_primes))
            if new_v in current_vectors:
                return False, None
        
        # Build new set
        new_vectors = set(current_vectors)
        for v in current_vectors:
            new_v = tuple(v[i] + xv[i] for i in range(num_primes))
            new_vectors.add(new_v)
            
        return True, new_vectors
    
    # Check if we can remove y and add x1, x2 (net +1 size)
    def check_1remove_2add(current_vectors, y, x1, x2):
        # First, compute vectors without y
        # We need to subtract y from all vectors that have y in them
        # This is complex. Instead, let's just rebuild from scratch for the smaller set.
        # But we don't have the set of elements, only the vectors.
        # So this approach requires passing the element list too.
        # For now, let's skip this and rely on other improvements.
        return False, None

    random.seed(42)
    
    # Phase 1: Randomized Greedy with multiple restarts
    for restart in range(2000):
        if time.time() > deadline - 2.0:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        current_vectors = {tuple([0] * num_primes)}
        
        for x in shuffled:
            valid, new_vectors = check_add_single(current_vectors, x)
            if valid:
                current_set.append(x)
                current_vectors = new_vectors
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 18:
                pass
    
    # Phase 2: Deterministic Greedy with different heuristics
    def count_pf(n):
        c = 0
        for p in primes:
            while n % p == 0:
                c += 1
                n //= p
        return c
        
    sorted_by_pf = sorted(candidates, key=count_pf)
    
    # Try greedy with this order
    current_set = []
    current_vectors = {tuple([0] * num_primes)}
    for x in sorted_by_pf:
        valid, new_vectors = check_add_single(current_vectors, x)
        if valid:
            current_set.append(x)
            current_vectors = new_vectors
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]
        
    # Try reverse order
    current_set = []
    current_vectors = {tuple([0] * num_primes)}
    for x in sorted_by_pf[::-1]:
        valid, new_vectors = check_add_single(current_vectors, x)
        if valid:
            current_set.append(x)
            current_vectors = new_vectors
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]
        
    # Try greedy sorted by value
    for order in [sorted(candidates), sorted(candidates, reverse=True)]:
        current_set = []
        current_vectors = {tuple([0] * num_primes)}
        for x in order:
            valid, new_vectors = check_add_single(current_vectors, x)
            if valid:
                current_set.append(x)
                current_vectors = new_vectors
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]

    # Phase 3: Local Search - Try to add elements to the best set
    if time.time() < deadline - 5.0:
        valid, best_vectors = check_add_single({tuple([0] * num_primes)}, best_set[0])
        for x in best_set[1:]:
            valid, best_vectors = check_add_single(best_vectors, x)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for x in candidates:
                if x in best_set:
                    continue
                valid, new_vectors = check_add_single(best_vectors, x)
                if valid:
                    best_set.append(x)
                    best_vectors = new_vectors
                    best_size += 1
                    improved = True
                    break

    # Phase 4: More intensive random restarts with different strategies
    if time.time() < deadline - 2.0:
        for restart in range(1000):
            if time.time() > deadline - 1.0:
                break
            
            # Mix of random and heuristic
            if restart % 3 == 0:
                order = candidates[:]
                random.shuffle(order)
            elif restart % 3 == 1:
                order = sorted_by_pf
            else:
                order = sorted(candidates, key=lambda x: (count_pf(x), x))
            
            current_set = []
            current_vectors = {tuple([0] * num_primes)}
            for x in order:
                valid, new_vectors = check_add_single(current_vectors, x)
                if valid:
                    current_set.append(x)
                    current_vectors = new_vectors
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]

    return [int(x) for x in best_set]
