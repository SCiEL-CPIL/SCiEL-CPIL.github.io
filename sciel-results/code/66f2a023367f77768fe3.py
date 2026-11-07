import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 210
    max_val = 10210
    
    # Precompute prime factors for numbers up to max_val
    # We only care about primes up to max_val, but actually we only need primes 
    # that appear in the factorization of numbers in (210, max_val].
    # Max prime factor of 10210 is 10210 itself (if prime) or less.
    # 10210 = 2 * 5 * 1021. 1021 is prime? sqrt(1021) ~ 31.9. 
    # Primes: 2,3,5,7,11,13,17,19,23,29,31.
    # 1021 / 7 = 145.8, /11=92, /13=78, /17=60, /19=53, /23=44, /29=35, /31=32.
    # 1021 is prime.
    
    # To be safe and fast, let's just find primes up to max_val.
    # But we can optimize: we only need primes up to max_val.
    # Actually, for GF(2) representation, we need a unique index for each prime.
    
    def sieve(max_n):
        is_prime = [True] * (max_n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(max_n)) + 1):
            if is_prime[i]:
                for j in range(i*i, max_n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(max_val)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Calculate square-free kernel (parity vector) for each number
    # We represent the vector as a Python int (bitset).
    # Bit i is 1 if prime[i] appears an odd number of times in the factorization.
    
    # First, compute kernel for 210
    def get_kernel(x):
        temp = x
        kernel = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    kernel |= (1 << prime_to_idx[p])
        if temp > 1:
            # temp is a prime factor
            if temp in prime_to_idx:
                kernel |= (1 << prime_to_idx[temp])
            else:
                # This shouldn't happen if sieve covers max_val
                pass
        return kernel

    target_kernel = get_kernel(210)
    
    # We want to find a subset S of {211, ..., M} such that XOR of their kernels == target_kernel.
    # This is equivalent to finding a subset of vectors that sums to target in GF(2).
    # We can treat this as a linear algebra problem.
    # Let v_i be the kernel of i.
    # We want sum(c_i * v_i) = target, where c_i in {0,1}.
    
    # Strategy:
    # Iterate M from 211 upwards.
    # Maintain a basis for the vectors {v_211, ..., v_M}.
    # Also maintain the "reachable" space? No, that's expensive.
    # Better: Standard Gaussian elimination to find if target is in the span of {v_211, ..., v_M}.
    # If yes, we can backtrack to find the actual subset.
    
    # However, we want to MINIMIZE M.
    # So we check M = 211, 212, ...
    # For each M, we add v_M to the system.
    # If target becomes reachable, we have a solution with max element M.
    # We need to extract the subset.
    
    # To extract the subset, we can store the dependencies during elimination.
    # Each basis vector b_j can be represented as a set of original indices that form it.
    # Or simpler: since we only need ONE solution, and we are adding vectors one by one,
    # we can keep track of how each new vector is reduced.
    
    # Let's use a dictionary: basis[leading_bit] = (vector, set_of_indices)
    # When we add a new vector v with index i:
    #   current = v, current_set = {i}
    #   while current != 0:
    #       lb = highest set bit
    #       if lb in basis:
    #           current ^= basis[lb][0]
    #           current_set ^= basis[lb][1]
    #       else:
    #           basis[lb] = (current, current_set)
    #           break
    #   If current == 0, it means v is in the span of previous vectors.
    #   This doesn't directly tell us if target is reachable, but it updates the basis.
    
    # Wait, if we just maintain the basis, how do we know if target is reachable?
    # We can try to reduce target using the current basis.
    # If target reduces to 0, it is reachable.
    # If so, the set of indices used in the reduction + the indices from the basis vectors used
    # gives the solution.
    
    # To make extraction easy, let's store for each basis vector the set of original numbers that constitute it.
    
    basis = {} # leading_bit -> (kernel_int, set_of_indices)
    
    # Precompute kernels for all candidates 211..max_val
    # This might be slow if done naively for each M, but we can precompute once.
    # 10000 numbers * ~100 primes = 1M ops. Very fast.
    
    kernels = {}
    for i in range(211, max_val + 1):
        kernels[i] = get_kernel(i)
        
    # Iterate M
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        v = kernels[M]
        current = v
        current_set = {M}
        
        # Reduce v using existing basis
        while current != 0:
            # Find highest set bit
            lb = current.bit_length() - 1
            if lb in basis:
                b_vec, b_set = basis[lb]
                current ^= b_vec
                current_set ^= b_set
            else:
                basis[lb] = (current, current_set)
                break
        
        # Now check if target is reachable
        # Try to reduce target using the current basis
        t = target_kernel
        t_set = set() # Indices from basis vectors used to form target
        
        # We need to be careful: if we use a basis vector, we must include its constituent indices.
        # But if we use the same basis vector twice, it cancels out.
        # In GF(2), we just XOR.
        
        # To find the specific subset, we reduce target.
        # If we successfully reduce target to 0, the XOR of the sets of the basis vectors used
        # is the solution.
        
        # However, we must ensure the indices are distinct?
        # The problem says "distinct integers".
        # The linear combination gives a set of indices. Since we are in GF(2), each index appears 0 or 1 times.
        # So the resulting set is automatically distinct.
        
        # Let's perform the reduction of target.
        # We need to know WHICH basis vectors were used.
        
        t_current = target_kernel
        t_used_indices = set()
        
        # We iterate through the basis? No, we reduce t_current.
        # But we need to track the set.
        # The basis vectors are linearly independent.
        # If t_current reduces to 0, the set of indices is the symmetric difference of the sets of the basis vectors used.
        
        # Let's do the reduction and track the sets.
        # Since the basis is fixed for this M, we can just do it.
        
        # Optimization: Only check if target is in span if we added a new vector or if we haven't checked yet?
        # Actually, if target was reachable with M-1, we would have found it then.
        # So we only need to check if the NEW vector allowed the target to be reached.
        # This happens if the new vector was added to the basis (increasing rank) OR if it completed a cycle?
        # No, if the new vector was dependent (reduced to 0), it didn't expand the span.
        # So we only need to check if the new vector was added to the basis.
        
        if current != 0:
            # New basis vector added, span expanded. Check target.
            t_current = target_kernel
            t_used_indices = set()
            
            # Reduce t_current
            # We need to map leading bits to the sets.
            # We can just use the basis dict.
            
            # To avoid modifying t_current in a way that loses track, we just simulate.
            # Since we need the set, we XOR the sets as we go.
            
            # Note: The basis vectors themselves are built from original indices.
            # So if we use basis[lb], we XOR its set into t_used_indices.
            
            # Is it possible that t_current reduces to 0?
            # Yes, if target is in the span.
            
            # Let's do the reduction.
            # We need a copy of the basis states? No, basis is read-only here.
            
            # Wait, the reduction process:
            # while t_current != 0:
            #   lb = highest bit
            #   if lb in basis:
            #      t_current ^= basis[lb][0]
            #      t_used_indices ^= basis[lb][1]
            #   else:
            #      break # Not reachable
            
            # If we break, not reachable.
            # If t_current becomes 0, reachable.
            
            # Let's run this check.
            # To save time, we can pre-compute the "rank" or just do this.
            # The loop runs at most len(primes) times.
            
            check_t = target_kernel
            check_set = set()
            
            while check_t != 0:
                lb = check_t.bit_length() - 1
                if lb in basis:
                    b_vec, b_set = basis[lb]
                    check_t ^= b_vec
                    check_set ^= b_set
                else:
                    break
            
            if check_t == 0:
                # Found a solution!
                # The set of indices is check_set.
                # All indices are in [211, M].
                # They are distinct.
                # Product * 210 is a square.
                
                # Convert to list of ints
                result = [int(x) for x in check_set]
                
                # Verify just in case
                # prod = 210
                # for x in result:
                #     prod *= x
                # s = math.isqrt(prod)
                # if s*s == prod:
                #     return result
                # else:
                #     # Error, continue?
                #     pass
                
                # Return the list
                return result
                
    # If no solution found up to max_val (should not happen), return a fallback
    # A known solution is [840] (840*210 = 420^2)
    # Wait, 840 is in the range (210, 10210].
    # Is 840 distinct from others? It's a single element, so yes.
    # But we want to minimize max - 210.
    # If the loop didn't find anything, it means no subset of {211..10210} works?
    # That's impossible because {840} works.
    # Why would the loop miss it?
    # If M=840, v_840 is added.
    # Target is kernel(210).
    # kernel(840) = kernel(210 * 4) = kernel(210).
    # So v_840 == target.
    # When M=840, we add v_840.
    # It reduces to 0? No, if basis is empty or doesn't have this vector.
    # If basis is empty, v_840 becomes a basis vector.
    # Then we check target.
    # Target reduces to 0 using basis vector v_840.
    # check_set will be {840}.
    # So it should find it.
    
    return [840]

if __name__ == "__main__":
    res = construct()
    print(res)
