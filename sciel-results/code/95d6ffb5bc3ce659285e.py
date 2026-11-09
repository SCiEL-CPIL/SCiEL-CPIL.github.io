import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s wall-clock limit

    n = 210
    max_val = 10210
    max_size = 100

    # Target square-free kernel for 210 = 2 * 3 * 5 * 7
    target = 2 * 3 * 5 * 7

    def get_square_free_kernel(x):
        """
        Returns the square-free kernel of x: product of primes with odd exponent.
        """
        temp = x
        kernel = 1
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    kernel *= p
            p += 1 if p == 2 else 2
        if temp > 1:
            kernel *= temp
        return kernel

    # Precompute square-free kernels for all candidates 211 to max_val
    # This is fast enough to do upfront
    kernels = {}
    for x in range(211, max_val + 1):
        if time.time() > deadline - 5:
            break
        kernels[x] = get_square_free_kernel(x)

    # If precomputation didn't finish (unlikely), we have a problem, but let's assume it does.
    # To be safe, we only use what we have.
    if not kernels:
        return [840] # Fallback

    # Binary search for the minimal M such that a valid subset exists
    # The subset must consist of distinct integers from (210, M]
    # Condition: XOR of their square-free kernels must equal target.
    # This is equivalent to: target is in the linear span (over GF(2)) of the vectors
    # corresponding to the square-free kernels of numbers in (210, M].
    
    # We represent each square-free kernel as a vector in GF(2)^P, where P is the set of primes.
    # Since we only care about whether 'target' is in the span, we can just check if
    # the rank of the set {k1, ..., km, target} is equal to the rank of {k1, ..., km}.
    # Or simpler: reduce 'target' by the basis of {k1, ..., km}. If it reduces to 1, it's in the span.
    # Wait, the operation is XOR of kernels.
    # Let v(x) be the vector of prime exponents mod 2 for x.
    # We want sum(v(x_i)) = v(target) (mod 2).
    # This is a linear system over GF(2).
    
    # We can maintain a basis of vectors as we add more numbers.
    # A vector is a set of primes (or an integer representing the square-free kernel).
    # XOR of square-free kernels corresponds to symmetric difference of their prime sets.
    
    # Let's implement a Gaussian elimination that maintains a basis of square-free kernels.
    # Basis: list of (vector, original_number) or just vectors.
    # We want to check if target is in the span.
    
    def can_form_target(max_m):
        """
        Checks if there exists a subset of numbers in (210, max_m]
        whose product * 210 is a perfect square.
        Returns (True, subset) if found, (False, None) otherwise.
        """
        # Collect all numbers in (210, max_m]
        # We need to check if target is in the span of their square-free kernels.
        # We also need to reconstruct the subset.
        
        # To reconstruct, we need to keep track of which numbers contributed to the basis.
        # Standard Gaussian elimination can track this.
        
        # Basis: dict mapping leading prime (or just the vector) to the vector and the set of numbers.
        # Actually, simpler: 
        # 1. Add vectors one by one to a basis.
        # 2. Keep track of the "combination" for each basis vector.
        
        # Let's use a list of basis vectors, each associated with a bitmask or set of indices.
        # Since max_size is 100, but the range can be large, we can't use a simple bitmask of size M.
        # However, we only need *one* valid subset.
        
        # Alternative: 
        # Perform Gaussian elimination on the set of vectors.
        # If target reduces to 1, then target is in the span.
        # To find the actual subset, we can store for each basis vector the set of original numbers that XOR to it.
        
        # Let's try this:
        # basis: list of (vector, set_of_numbers)
        # When adding a new vector v with number x:
        #   w = v
        #   S = {x}
        #   For each (b, B) in basis:
        #       if leading term of w is divisible by leading term of b (or just check if w XOR b < w? No, need consistent ordering)
        #       Let's order primes. The "leading" prime is the largest prime factor.
        #       If w has the same leading prime as b:
        #           w = w XOR b
        #           S = S XOR B (symmetric difference)
        #       If w becomes 1, we found a dependency? No, we want to reduce target.
        
        # Actually, let's just build the basis and then reduce target.
        
        # To make it efficient, we only iterate up to max_m.
        # We need to be careful with time.
        
        # Let's collect all kernels first for this max_m?
        # No, we can do it incrementally if we were doing a sweep, but binary search requires re-checking.
        # Since max_m changes, we have to recompute the basis for each check.
        # The range is up to 10000. 10000 vectors. Gaussian elimination is O(N * P) where P is number of primes.
        # P is small (< 100). So 10000 * 100 = 1e6 ops. Very fast.
        
        # Let's implement the check.
        
        # We need to handle the "set of numbers" part to return the subset.
        # Using sets of integers for the combination is fine as long as the subset size is small.
        # The subset size will be at most the rank of the matrix, which is <= P (number of primes involved).
        # P is the number of distinct primes <= 10210. There are ~1200 primes.
        # But the rank is bounded by the number of distinct prime factors in the kernels.
        # Actually, the rank is at most the number of primes that appear in any kernel.
        # This could be large.
        
        # Wait, if the rank is large, the subset size could be large.
        # But the problem allows up to 100 integers.
        # If the minimal subset found by Gaussian elimination has size > 100, it's invalid.
        # However, usually the rank is small for these kinds of problems?
        # Let's assume the subset size is manageable. If not, we might need a more complex search,
        # but let's try the simple Gaussian elimination first.
        
        # Optimization: We can stop early if we find the target in the span.
        
        # Let's define the reduction function.
        # We will maintain a basis of vectors.
        # Each basis vector is a square-free integer (kernel).
        # We also maintain a dictionary mapping each basis vector to the set of numbers that form it.
        
        # Actually, a cleaner way:
        # 1. Build basis of kernels for numbers in (210, max_m].
        # 2. Reduce target using this basis.
        # 3. If target reduces to 1, it is in the span.
        # 4. The reduction process tells us which basis vectors were used.
        # 5. We then XOR the sets of numbers corresponding to those basis vectors.
        
        # Let's implement this.
        
        # Step 1: Build Basis
        # We need to iterate through numbers 211 to max_m.
        # For each number x, get kernel k.
        # Try to add k to the basis.
        
        # Basis representation:
        # A list of tuples (kernel, set_of_numbers)
        # To make reduction fast, we should sort basis vectors by their "leading" prime (largest prime factor).
        # Or just keep them in a dict keyed by the kernel itself? No, kernels can repeat.
        # Standard Gaussian elimination on vectors over GF(2):
        # We want a basis where each vector has a unique "pivot" (e.g., largest prime factor).
        
        # Let's use a list of (pivot, kernel, set_of_numbers).
        # Pivot is the largest prime factor of the kernel.
        
        basis = [] # list of (pivot, kernel, set_of_numbers)
        
        def get_largest_prime_factor(n):
            if n == 1: return 1
            temp = n
            p = 2
            largest = 1
            while p * p <= temp:
                if temp % p == 0:
                    largest = p
                    while temp % p == 0:
                        temp //= p
                p += 1 if p == 2 else 2
            if temp > 1:
                largest = temp
            return largest

        # Precompute largest prime factor for all possible kernels?
        # Kernels are square-free products of primes <= 10210.
        # The number of possible kernels is 2^P, which is huge.
        # But we only see kernels of numbers <= 10210.
        # So we can just compute LPF on the fly or cache it.
        
        lpf_cache = {}
        def lpf(n):
            if n in lpf_cache:
                return lpf_cache[n]
            res = get_largest_prime_factor(n)
            lpf_cache[n] = res
            return res

        # Build basis
        for x in range(211, max_m + 1):
            if time.time() > deadline - 5:
                return (False, None)
            k = kernels[x]
            if k == 1:
                # 1 is the zero vector in this context? No, kernel 1 means x is a perfect square.
                # v(x) = 0. It doesn't contribute to the span.
                # So we can ignore it for basis construction.
                continue
            
            # Try to reduce k using existing basis
            # We iterate through basis in descending order of pivot
            # If basis is not sorted, we should sort it or pick the one with matching pivot.
            # Let's keep basis sorted by pivot descending.
            
            current_k = k
            current_set = {x}
            
            # We need to find a basis vector with the same pivot as current_k's pivot
            # Or more generally, reduce current_k.
            
            # Efficient reduction:
            # Find the basis vector with the same pivot as current_k.
            # If found, XOR them.
            # If not, add current_k to basis.
            
            # Since we want to maintain a reduced row echelon form (RREF) like structure:
            # Each basis vector has a unique pivot, and no other basis vector has that pivot.
            
            # Let's find the pivot of current_k
            piv = lpf(current_k)
            
            # Search for a basis vector with pivot == piv
            found = False
            for i, (b_piv, b_k, b_set) in enumerate(basis):
                if b_piv == piv:
                    # XOR
                    current_k ^= b_k
                    current_set = current_set.symmetric_difference(b_set)
                    found = True
                    # After XOR, the pivot might change (if the leading term cancels out)
                    # We need to repeat the reduction process with the new current_k
                    break
            
            if not found:
                # No basis vector with this pivot. Add to basis.
                # But wait, what if current_k has a different pivot now?
                # The loop above only checks for the *original* pivot of k?
                # No, we need to keep reducing until we either hit 1 or find a pivot that isn't in the basis.
                
                # Let's rewrite the reduction loop properly.
                
                current_k = k
                current_set = {x}
                while current_k != 1:
                    piv = lpf(current_k)
                    # Find basis vector with this pivot
                    matched = False
                    for i, (b_piv, b_k, b_set) in enumerate(basis):
                        if b_piv == piv:
                            current_k ^= b_k
                            current_set = current_set.symmetric_difference(b_set)
                            matched = True
                            break
                    if not matched:
                        # Add to basis
                        # Insert in sorted order? Or just append and sort later?
                        # For correctness of "unique pivot", we just need to ensure we check all.
                        # Since we check all basis vectors for a match, the order doesn't strictly matter for correctness,
                        # but for efficiency, we want to find the match quickly.
                        # Given small basis size, linear search is fine.
                        basis.append((piv, current_k, current_set))
                        break
                # If current_k becomes 1, it means x is in the span of previous numbers.
                # We don't add it to the basis, but we found a dependency.
                # We don't need the dependency for the basis, but we might need it later?
                # No, the basis spans the same space.
        
        # Now, reduce target using the basis
        target_k = target
        target_set = set()
        
        while target_k != 1:
            piv = lpf(target_k)
            matched = False
            for i, (b_piv, b_k, b_set) in enumerate(basis):
                if b_piv == piv:
                    target_k ^= b_k
                    target_set = target_set.symmetric_difference(b_set)
                    matched = True
                    break
            if not matched:
                # Target is not in the span
                return (False, None)
        
        # If we get here, target_k is 1, so target is in the span.
        # The subset is target_set.
        # We must check if the subset size is <= 100 and all elements are distinct (they are, as they are sets).
        # Also, we must ensure all elements are > 210 and <= max_m.
        # They are by construction.
        
        if len(target_set) > max_size:
            # This subset is too large.
            # Is there a smaller one?
            # The Gaussian elimination gives *a* solution. It might not be the smallest.
            # However, usually the rank is small.
            # If this fails, we might need a more sophisticated search.
            # But let's assume for now that if a solution exists, this one is valid or close.
            # Actually, if len > 100, we cannot use this specific subset.
            # We would need to search for a smaller combination.
            # This is a potential failure mode.
            # Let's check if there's a way to find a smaller subset.
            # The minimal weight solution in a linear code is NP-hard.
            # But with rank < 20 or so, we can use meet-in-the-middle or just hope the Gaussian elimination gives a small one.
            # Given the structure of square-free kernels, the rank is likely small.
            
            # For now, if it's too large, we return False? Or try to find another?
            # Let's return False for this max_m and increase it? No, if a solution exists, it must be in the span.
            # If the only solutions are large, we are stuck.
            # But the problem asks to minimize max(selected). It doesn't say minimize size.
            # Wait, "1..100 distinct integers". So size MUST be <= 100.
            # If the Gaussian elimination gives a subset of size 101, it's invalid.
            # We need to find a subset of size <= 100.
            
            # This is a constraint satisfaction problem.
            # Let's hope the rank is small enough that the Gaussian elimination subset is <= 100.
            # The number of primes <= 10210 is ~1200.
            # But the kernels are products of primes.
            # The rank of the set of kernels of numbers 211..M is at most the number of distinct primes appearing in them.
            # This could be large.
            
            # However, notice that we are looking for a specific target.
            # Let's proceed with this implementation and see if it works.
            # If it fails due to size, we might need to add a step to minimize weight.
            
            return (False, None)

        # Verify the subset
        # Check distinctness (guaranteed by set)
        # Check range (guaranteed by construction)
        # Check product * 210 is square
        prod = 210
        for x in target_set:
            prod *= x
        sq = math.isqrt(prod)
        if sq * sq == prod:
            return (True, list(target_set))
        else:
            # Should not happen
            return (False, None)

    # Binary search for minimal M
    # Lower bound: 211 (smallest possible max)
    # Upper bound: 10210
    
    low = 211
    high = max_val
    best_m = max_val
    best_subset = None
    
    # We know 840 works (single element 210*4). So high can be 840 initially?
    # No, there might be a 2-element solution with max < 840.
    # Let's start high at 840 to save time, but check if we can go lower.
    # Actually, just binary search 211 to 10210.
    
    # Optimization: We can cache the basis building?
    # No, because the set of numbers changes.
    
    # To speed up, we can pre-compute the kernels (done).
    # The Gaussian elimination is O(M * Rank * CostOfXOR).
    # M is up to 10000. Rank is small. CostOfXOR is small.
    # 10000 * 100 = 1e6 ops.
    # Binary search takes log2(10000) ~ 14 steps.
    # Total 14e6 ops. Should be well within 60s.
    
    while low <= high:
        if time.time() > deadline - 5:
            break
        mid = (low + high) // 2
        feasible, subset = can_form_target(mid)
        if feasible:
            best_m = mid
            best_subset = subset
            high = mid - 1
        else:
            low = mid + 1
    
    if best_subset is None:
        # Fallback to known solution
        best_subset = [840]
    else:
        # Final verification
        prod = 210
        for x in best_subset:
            prod *= x
        sq = math.isqrt(prod)
        if sq * sq != prod or len(best_subset) > 100 or len(set(best_subset)) != len(best_subset):
            best_subset = [840]
        else:
            # Ensure types
            best_subset = [int(x) for x in best_subset]
            
    # Final checks
    assert isinstance(best_subset, list)
    assert 1 <= len(best_subset) <= 100
    assert len(set(best_subset)) == len(best_subset)
    for x in best_subset:
        assert isinstance(x, int) and not isinstance(x, bool)
        assert 210 < x <= 10210
        
    return best_subset
