import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Precompute prime factorization vectors for numbers 2..128
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
    
    primes = [x for x in range(2, 129) if is_prime(x)]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Exponent vector for each number 2..128
    exp_vecs = {}
    for n in range(2, 129):
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(primes):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)
        
    # Packed integer representation for fast hashing
    BITS = 10
    def pack_vec(vec):
        res = 0
        for i, e in enumerate(vec):
            res |= e << (i * BITS)
        return res

    def check_valid(seq):
        """Check if sequence has all distinct consecutive block products using prefix sums."""
        n = len(seq)
        if n == 0:
            return True
        # Build prefix sum vectors: prefix[i] = sum of exp_vecs[seq[0:i]]
        prefix = [[0] * num_primes]
        for i in range(n):
            ej = exp_vecs[seq[i]]
            prev = prefix[-1]
            prefix.append([prev[k] + ej[k] for k in range(num_primes)])
        
        prods = set()
        for i in range(n):
            pi = prefix[i]
            for j in range(i + 1, n + 1):
                pj = prefix[j]
                # Pack the difference vector
                packed = 0
                for k in range(num_primes):
                    diff = pj[k] - pi[k]
                    packed |= diff << (k * BITS)
                if packed in prods:
                    return False
                prods.add(packed)
        return True
    
    def try_append(seq, c):
        """Try to append c to seq (c must be > seq[-1]). Return new seq or None."""
        if seq and c <= seq[-1]:
            return None
        new_seq = seq + [c]
        if check_valid(new_seq):
            return new_seq
        return None
    
    # Greedy: iterate through 2..128, add if possible
    seq = []
    for n in range(2, 129):
        if time.time() > deadline:
            break
        new_seq = try_append(seq, n)
        if new_seq is not None:
            seq = new_seq
    
    # Local search: try to improve by swapping elements
    def local_search(seq, max_iter=2000):
        current = seq[:]
        used = set(current)
        remaining = [x for x in range(2, 129) if x not in used]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            # Try insertions
            for c in remaining[:]:
                # Try inserting c at each valid position
                new_seq = None
                for i in range(len(current) + 1):
                    if i > 0 and current[i-1] >= c:
                        continue
                    if i < len(current) and c >= current[i]:
                        continue
                    candidate = current[:i] + [c] + current[i:]
                    if check_valid(candidate):
                        new_seq = candidate
                        break
                if new_seq is not None:
                    current = new_seq
                    used = set(current)
                    remaining = [x for x in range(2, 129) if x not in used]
                    improved = True
                    break
            
            # Try swaps: remove one, add another
            if not improved:
                for i in range(len(current)):
                    removed = current[i]
                    base = current[:i] + current[i+1:]
                    for c in remaining:
                        if c <= removed:
                            continue
                        # Try inserting c into base
                        for j in range(len(base) + 1):
                            if j > 0 and base[j-1] >= c:
                                continue
                            if j < len(base) and c >= base[j]:
                                continue
                            candidate = base[:j] + [c] + base[j:]
                            if check_valid(candidate) and len(candidate) >= len(current):
                                current = candidate
                                used = set(current)
                                remaining = [x for x in range(2, 129) if x not in used]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current
    
    # Improve with local search
    improved_seq = local_search(seq, max_iter=3000)
    if len(improved_seq) > len(seq):
        seq = improved_seq
    
    # Try multiple greedy variants with different orderings
    for _ in range(50):
        if time.time() > deadline:
            break
        # Try greedy with slight randomization
        rng = random.Random()
        candidates = list(range(2, 129))
        # Bias towards primes
        rng.shuffle(candidates)
        # Put primes first with some probability
        primes_first = [x for x in candidates if is_prime(x)]
        composites_first = [x for x in candidates if not is_prime(x)]
        if rng.random() < 0.5:
            order = primes_first + composites_first
        else:
            order = candidates
        rng.shuffle(order)
        
        s2 = []
        for n in order:
            if time.time() > deadline:
                break
            ns = try_append(s2, n)
            if ns is not None:
                s2 = ns
        
        s2 = local_search(s2, max_iter=1000)
        if len(s2) > len(seq):
            seq = s2
    
    return seq
