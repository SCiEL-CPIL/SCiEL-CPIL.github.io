import itertools
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def is_sidon_mask(mask, S, n):
        """
        Check if the subset defined by bitmask is Sidon.
        S is a list of 12 integers.
        """
        # Extract elements
        elems = [S[i] for i in range(n) if (mask >> i) & 1]
        k = len(elems)
        if k <= 2:
            return True
            
        sums = set()
        # To speed up, we can just check all pairs
        for i in range(k):
            a = elems[i]
            for j in range(i, k):
                b = elems[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def largest_sidon_size(S):
        """
        Find the size of the largest Sidon subset of S.
        S is a list of 12 distinct integers.
        """
        n = len(S)
        # We want to find the max k such that there exists a mask of weight k that is Sidon.
        # Iterate k from n down to 1.
        # For each k, iterate all masks of weight k.
        
        # Optimization: Precompute which masks are Sidon?
        # There are 4096 masks. Checking each takes ~O(k^2).
        # Total work ~ 4096 * 100 ops = 400k ops. Very fast.
        
        # Let's just check all masks from high weight to low weight.
        # Generate masks by weight.
        
        # A faster way: 
        # 1. Check if full set is Sidon.
        # 2. If not, try removing 1 element (12 choices).
        # 3. If not, try removing 2 elements (66 choices).
        # ...
        
        # Actually, iterating all 4096 masks and finding the max weight Sidon mask is robust and fast enough.
        max_size = 0
        for mask in range(1 << n):
            # Count bits
            # Python 3.10+ has int.bit_count()
            w = bin(mask).count('1')
            if w <= max_size:
                continue
            if is_sidon_mask(mask, S, n):
                max_size = w
                # Can we break early? No, a larger mask might exist.
                # But if we iterate masks in decreasing order of weight, we can break.
                # Let's generate masks by weight.
                
        # Better approach: Iterate by weight
        for w in range(n, 0, -1):
            # Iterate combinations of indices of size w
            found = False
            # Use itertools.combinations for indices
            for combo in itertools.combinations(range(n), w):
                mask = 0
                for i in combo:
                    mask |= (1 << i)
                if is_sidon_mask(mask, S, n):
                    return w
            # If no mask of size w is Sidon, try w-1
        return 0

    def largest_sidon_size_fast(S):
        """
        Optimized: Iterate masks from 2^n-1 down to 0 is not sorted by weight.
        Let's stick to the weight-based search but optimize the Sidon check.
        
        Even better: 
        For n=12, total subsets 4096.
        We can precompute the "conflict" structure?
        
        Let's just use the weight-based search. It's fast enough for local search steps if we cache or if the check is fast.
        The check is O(k^2). Max k=12, so 78 sums.
        4096 * 78 ~ 300k operations per evaluation.
        If we do 1000 evaluations, 300M ops. Might be tight in Python.
        
        Let's optimize `is_sidon_mask` using bit operations or precomputed sums?
        
        Alternative:
        A subset is NOT Sidon if there is a collision.
        Collision: a_i + a_j = a_k + a_l with (i,j) != (k,l).
        
        Let's try to improve the search strategy first.
        """
        n = len(S)
        # Check size n
        if is_sidon_mask((1<<n)-1, S, n):
            return n
        
        # Check size n-1
        for i in range(n):
            mask = ((1<<n)-1) ^ (1<<i)
            if is_sidon_mask(mask, S, n):
                return n-1
        
        # Check size n-2
        for i in range(n):
            for j in range(i+1, n):
                mask = ((1<<n)-1) ^ (1<<i) ^ (1<<j)
                if is_sidon_mask(mask, S, n):
                    return n-2
                    
        # Fallback to full search for smaller sizes
        for w in range(n-3, 0, -1):
            for combo in itertools.combinations(range(n), w):
                mask = 0
                for i in combo:
                    mask |= (1 << i)
                if is_sidon_mask(mask, S, n):
                    return w
        return 0

    # Initial best
    best_S = list(range(12))
    best_score = largest_sidon_size_fast(best_S)
    
    # Generate structured candidates
    candidates = set()
    
    # 1. Arithmetic Progressions
    for start in range(0, 117):
        for d in range(1, 12):
            S = [start + i*d for i in range(12)]
            if max(S) <= 128:
                candidates.add(tuple(S))
                
    # 2. Random
    random.seed(42)
    for _ in range(2000):
        S = random.sample(range(129), 12)
        candidates.add(tuple(S))
        
    # 3. "Dense" clusters to force collisions
    # E.g. two interleaved APs
    for start in range(0, 50):
        for d1 in range(1, 10):
            for d2 in range(1, 10):
                S = []
                for i in range(12):
                    if i % 2 == 0:
                        S.append(start + (i//2)*d1)
                    else:
                        S.append(start + 1 + ((i-1)//2)*d2)
                if len(set(S)) == 12 and max(S) <= 128:
                    candidates.add(tuple(S))

    # Evaluate candidates
    for S_tuple in candidates:
        S = list(S_tuple)
        score = largest_sidon_size_fast(S)
        if score < best_score:
            best_score = score
            best_S = S[:]
            
        if time.time() > deadline:
            break

    # Local Search
    def get_neighbor(S):
        """
        Modify one element of S.
        """
        S_new = S[:]
        idx = random.randint(0, 11)
        # Choose a new value not in S_new
        while True:
            v = random.randint(0, 128)
            if v not in S_new:
                S_new[idx] = v
                return S_new

    # Hill Climbing / Simulated Annealing
    current_S = best_S[:]
    current_score = best_score
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Random restart occasionally
        if random.random() < 0.05:
            current_S = random.sample(range(129), 12)
            current_score = largest_sidon_size_fast(current_S)
            if current_score < best_score:
                best_score = current_score
                best_S = current_S[:]
        else:
            neighbor = get_neighbor(current_S)
            n_score = largest_sidon_size_fast(neighbor)
            
            # Accept if better, or with some probability if equal (to escape plateaus)
            if n_score < current_score or (n_score == current_score and random.random() < 0.1):
                current_S = neighbor
                current_score = n_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_S = current_S[:]
                    if best_score == 3: # Theoretical lower bound might be 3 or 4?
                         # If we hit 3, it's very good.
                         pass

    return best_S

# EVOLVE-BLOCK-END
