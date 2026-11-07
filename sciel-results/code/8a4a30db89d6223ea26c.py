import random
import time
from itertools import combinations

def construct():
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Generate exponent vectors for 2..40
    candidates = []
    for n in range(2, 41):
        vec = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        # temp should be 1
        candidates.append((n, tuple(vec)))
        
    # Function to check if a list of vectors is dissociated
    # We check if all subset sums are distinct.
    # For k vectors, there are 2^k sums.
    def is_dissociated(vecs):
        k = len(vecs)
        if k == 0:
            return True
        sums = set()
        # Use bitmask to iterate subsets
        # For k up to 18, 2^18 = 262144, which is fast.
        for mask in range(1 << k):
            s = [0] * num_primes
            # Optimization: build sums incrementally or just sum
            # Simple sum is fine for small k and dim 12
            for i in range(k):
                if mask & (1 << i):
                    for j in range(num_primes):
                        s[j] += vecs[i][j]
            sums.add(tuple(s))
        return len(sums) == (1 << k)
        
    # Faster check for adding one vector to an existing set
    # Maintain the set of existing subset sums.
    # When adding v, new sums are old_sums + v.
    # If any new_sum is in old_sums, collision.
    class DissociatedSet:
        def __init__(self):
            self.vecs = []
            self.sums = {tuple([0]*num_primes)}
            
        def can_add(self, vec):
            # Check if any existing sum + vec is in existing sums
            # Equivalently, if any existing sum is in (existing sums - vec)
            # Or just: for s in sums, if s+vec in sums: return False
            # But sums is a set of tuples.
            for s in self.sums:
                new_s = tuple(s[i] + vec[i] for i in range(num_primes))
                if new_s in self.sums:
                    return False
            return True
            
        def add(self, vec):
            # Add vec and update sums
            new_sums = set(self.sums)
            for s in self.sums:
                new_s = tuple(s[i] + vec[i] for i in range(num_primes))
                new_sums.add(new_s)
            self.sums = new_sums
            self.vecs.append(vec)
            
    best_set = []
    best_size = 0
    
    # Time limit 60s.
    start_time = time.time()
    deadline = start_time + 55
    
    # 1. Baseline: Construct by prime groups
    # For each prime, pick best dissociated subset of its powers
    baseline_vecs = []
    baseline_nums = []
    for p in primes:
        powers = [(n, v) for n, v in candidates if all(v[i] == 0 for i in range(num_primes) if primes[i] != p) and v[prime_to_idx[p]] > 0]
        # Actually, just filter candidates that are powers of p
        p_powers = []
        for n, v in candidates:
            # Check if n is a power of p
            is_power = True
            for i, pp in enumerate(primes):
                if pp != p and v[i] != 0:
                    is_power = False
                    break
                if pp == p and v[i] == 0:
                    # Must be > 0 for power >= p
                    if n != 1: # n is in 2..40
                        is_power = False
            if is_power:
                p_powers.append(v)
        
        # Find largest dissociated subset of p_powers
        # p_powers size is small (<=5)
        max_sub = []
        for r in range(len(p_powers), 0, -1):
            found = False
            for comb in combinations(p_powers, r):
                if is_dissociated(list(comb)):
                    max_sub = list(comb)
                    found = True
                    break
            if found:
                break
        baseline_vecs.extend(max_sub)
        
    # Map vecs back to nums for return
    # We need to return the numbers, not vectors
    # The baseline construction above loses the number mapping easily if we just take vectors.
    # Let's just run randomized greedy on the full candidate list.
    
    # Randomized Greedy
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        ds = DissociatedSet()
        for n, v in candidates:
            if ds.can_add(v):
                ds.add(v)
                
        if len(ds.vecs) > best_size:
            best_size = len(ds.vecs)
            # Store the numbers corresponding to ds.vecs
            # We need to map vec back to num.
            # Multiple nums can have same vec? No, distinct integers have distinct prime factorizations.
            # So vec -> num is unique.
            vec_to_num = {v: n for n, v in candidates}
            best_set = [vec_to_num[v] for v in ds.vecs]
            
        # Local improvement: try to add more
        # This is implicitly done by the greedy if we iterate, but greedy is one pass.
        # We can try to add remaining candidates in random order again.
        remaining = [c for c in candidates if c[1] not in ds.vecs]
        random.shuffle(remaining)
        for n, v in remaining:
            if time.time() > deadline:
                break
            if ds.can_add(v):
                ds.add(v)
        
        if len(ds.vecs) > best_size:
            best_size = len(ds.vecs)
            vec_to_num = {v: n for n, v in candidates}
            best_set = [vec_to_num[v] for v in ds.vecs]
            
    # Ensure we return a valid list of ints
    # If best_set is empty (shouldn't be), return primes
    if not best_set:
        return [p for p in primes]
        
    return best_set
