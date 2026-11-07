import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0
    
    max_val = 10210
    min_val = 211
    
    # Sieve
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_to_index = {p: i for i, p in enumerate(primes)}
    d = len(primes)
    
    # Target vector
    t = 0
    n = 210
    for p in primes:
        if n % p == 0:
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count % 2 == 1:
                t |= (1 << prime_to_index[p])
                
    # Vectors
    num_vecs = max_val - 210
    vectors = [0] * num_vecs
    for i in range(num_vecs):
        x = 211 + i
        v = 0
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                count = 0
                while n % p == 0:
                    n //= p
                    count += 1
                if count % 2 == 1:
                    v |= (1 << prime_to_index[p])
        if n > 1:
            if n in prime_to_index:
                v |= (1 << prime_to_index[n])
        vectors[i] = v
        
    def bit_count(x):
        return bin(x).count('1')

    def solve_for_K(K):
        if time.time() > deadline:
            return None, None
            
        basis_vec = [0] * d
        basis_comb = [0] * d
        kernel_basis = []
        
        # Build basis and collect kernel vectors
        for i in range(K):
            v = vectors[i]
            comb = 1 << i
            while v != 0:
                msb = v.bit_length() - 1
                if basis_vec[msb] != 0:
                    v ^= basis_vec[msb]
                    comb ^= basis_comb[msb]
                else:
                    basis_vec[msb] = v
                    basis_comb[msb] = comb
                    break
            if v == 0:
                if comb != 0:
                    kernel_basis.append(comb)
                    
        # Reduce t
        vt = t
        comb_t = 0
        while vt != 0:
            msb = vt.bit_length() - 1
            if basis_vec[msb] != 0:
                vt ^= basis_vec[msb]
                comb_t ^= basis_comb[msb]
            else:
                return None, None
                
        return comb_t, kernel_basis

    # Find minimal K for existence
    low, high = 1, num_vecs
    min_K_solvable = num_vecs + 1
    while low <= high:
        mid = (low + high) // 2
        sol, kern = solve_for_K(mid)
        if sol is not None:
            min_K_solvable = mid
            high = mid - 1
        else:
            low = mid + 1
            
    if min_K_solvable > num_vecs:
        return []
        
    best_solution = None
    best_max_val = max_val + 1
    
    # Search for low weight solution
    # Start from min_K_solvable
    for K in range(min_K_solvable, min(min_K_solvable + 100, num_vecs + 1)):
        if time.time() > deadline:
            break
            
        sol, kernel_basis = solve_for_K(K)
        if sol is None:
            continue
            
        current_weight = bit_count(sol)
        
        # Local search
        # Limit iterations to avoid timeout
        for _ in range(20):
            improved = False
            for k_vec in kernel_basis:
                new_sol = sol ^ k_vec
                new_weight = bit_count(new_sol)
                if new_weight < current_weight:
                    sol = new_sol
                    current_weight = new_weight
                    improved = True
            if not improved:
                break
                
        if current_weight <= 100:
            # Calculate actual max value
            max_idx = -1
            for i in range(K):
                if (sol >> i) & 1:
                    max_idx = i
            actual_max = 211 + max_idx
            if actual_max < best_max_val:
                best_max_val = actual_max
                best_solution = sol
                break # Minimal K implies minimal max bound, so first valid is best?
                # Actually, a solution for K=10 might have max index 9.
                # A solution for K=11 might have max index 5.
                # But if a solution for K=11 has max index 5, it is also a solution for K=10 (since indices 0..9 are available).
                # So the minimal K where *any* valid solution exists will yield the minimal max index.
                # Wait, "valid solution" means weight <= 100.
                # If for K=10, all solutions have weight > 100, but for K=11, a solution has weight 50 and max index 5,
                # then the max value is 211+5 = 216.
                # For K=10, no valid solution.
                # So we must find the first K where a valid solution exists.
                # And among valid solutions for that K, pick the one with min max index?
                # No, we just need *a* valid solution.
                # The max value is determined by the highest index used.
                # We want to minimize this.
                # So if we find a valid solution for K, we should check its max index.
                # Could a larger K yield a smaller max index?
                # Yes, if the solution for K uses only small indices.
                # But if a solution uses only indices < K, it is also a solution for any K' > K.
                # So the minimal K where a valid solution exists is the one we want.
                # However, we might find a solution for K=100 with max index 99.
                # And for K=101, we might find a solution with max index 10.
                # Is that possible?
                # If a solution uses indices {0..10}, it is a valid solution for K=101.
                # It is also a valid solution for K=11.
                # So if it existed, it would have been found at K=11.
                # So we should find the *first* K where a valid solution exists.
                # And return that solution.
                break
                
    if best_solution is None:
        return []
        
    selected = []
    for i in range(K):
        if (best_solution >> i) & 1:
            selected.append(211 + i)
            
    return selected
