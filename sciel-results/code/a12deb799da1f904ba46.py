import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    max_val = 10210
    min_val = 211
    N = max_val - min_val + 1  # 10000 numbers

    # 1. Sieve primes up to max_val
    is_prime = bytearray([1]) * (max_val + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = 0
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    pidx = {p: i for i, p in enumerate(primes)}
    d = len(primes)

    # 2. Compute parity vectors
    def get_vector(n):
        v = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt & 1:
                    v |= (1 << pidx[p])
        if temp > 1:
            v |= (1 << pidx[temp])
        return v

    target = get_vector(210)
    
    # vectors[i] corresponds to number min_val + i
    vectors = [0] * N
    for i in range(N):
        vectors[i] = get_vector(min_val + i)

    # 3. Find minimal M such that target is in span of vectors[0..M-211]
    # We maintain a basis in RREF-like form or just a list of basis vectors
    # To check membership efficiently, we can use a dictionary mapping MSB to basis vector
    basis = {}
    min_M = None
    
    for i in range(N):
        if time.time() > deadline - 5:
            break
            
        v = vectors[i]
        
        # Try to reduce v by existing basis
        r = v
        while r:
            msb = r.bit_length() - 1
            if msb in basis:
                r ^= basis[msb]
            else:
                # r is a new basis vector
                basis[msb] = r
                break
        
        # Check if target is in span
        # Reduce target by basis
        t = target
        while t:
            msb = t.bit_length() - 1
            if msb in basis:
                t ^= basis[msb]
            else:
                break
        
        if t == 0:
            min_M = min_val + i
            break
            
    if min_M is None:
        # Fallback if not found in time (should not happen)
        min_M = max_val

    # 4. Solve for the subset using numbers 211..min_M
    K = min_M - min_val + 1
    # System: A * x = b
    # A is d x K, columns are vectors[0..K-1]
    # We want x in {0,1}^K
    
    # Build augmented matrix rows: (coeffs of x | b)
    # coeffs are bits 0..K-1, b is bit K
    rows = []
    for j in range(d):
        row = 0
        for i in range(K):
            if (vectors[i] >> j) & 1:
                row |= (1 << i)
        if (target >> j) & 1:
            row |= (1 << K)
        rows.append(row)
        
    # Gaussian Elimination
    pivot_row = 0
    pivot_cols = []
    for c in range(K):
        if time.time() > deadline - 2:
            break
        found = -1
        for r in range(pivot_row, d):
            if (rows[r] >> c) & 1:
                found = r
                break
        if found == -1:
            continue
        rows[found], rows[pivot_row] = rows[pivot_row], rows[found]
        for r in range(d):
            if r != pivot_row and ((rows[r] >> c) & 1):
                rows[r] ^= rows[pivot_row]
        pivot_cols.append(c)
        pivot_row += 1
        if pivot_row >= d:
            break
            
    # Particular solution (free vars = 0)
    c_p = 0
    for r, c in enumerate(pivot_cols):
        if (rows[r] >> K) & 1:
            c_p |= (1 << c)
            
    # Null space basis
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(K) if c not in pivot_set]
    null_basis = []
    for f in free_cols:
        n = 1 << f
        for r, c in enumerate(pivot_cols):
            if (rows[r] >> f) & 1:
                n |= (1 << c)
        null_basis.append(n)
        
    # Minimize weight of c_p using null space
    current = c_p
    best_c = current
    best_weight = bin(current).count('1')
    
    # Greedy reduction
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for b in null_basis:
            nc = current ^ b
            nw = bin(nc).count('1')
            if nw < best_weight:
                current = nc
                best_c = nc
                best_weight = nw
                improved = True
                
    # If weight is still > 100, we might have a problem, 
    # but for minimal M, K is likely small enough or solution is sparse.
    # If K <= 100, max weight is K <= 100.
    if best_weight > 100:
        # Try random restarts in null space
        import random
        rng = random.Random(42)
        for _ in range(1000):
            if time.time() > deadline - 1:
                break
            # Pick random null space vector combination
            # Just try single flips from current
            # Or random subset of null basis
            k = rng.randint(1, 5)
            if len(null_basis) < k:
                k = len(null_basis)
            idxs = rng.sample(range(len(null_basis)), k)
            delta = 0
            for idx in idxs:
                delta ^= null_basis[idx]
            nc = c_p ^ delta
            nw = bin(nc).count('1')
            if nw < best_weight:
                best_c = nc
                best_weight = nw

    # Extract solution
    selected = []
    for i in range(K):
        if (best_c >> i) & 1:
            selected.append(min_val + i)
            
    # Final check
    if not selected:
        # Should not happen
        return [min_val]
        
    return selected

if __name__ == "__main__":
    sol = construct()
    print(f"Solution size: {len(sol)}")
    print(f"Max element: {max(sol) if sol else 0}")
    if sol:
        prod = 210
        for x in sol:
            prod *= x
        sqrt_prod = math.isqrt(prod)
        is_square = (sqrt_prod * sqrt_prod == prod)
        print(f"Is square: {is_square}")
        print(f"Raw value: {max(sol) - 210}")
