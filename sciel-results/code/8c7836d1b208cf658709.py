import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Sieve primes up to 10210
    max_val = 10210
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Target mask for 210 = 2 * 3 * 5 * 7
    # We need the product P such that 210*P is a square.
    # This means P must have odd exponents for primes 2, 3, 5, 7 and even for others.
    target_mask = 0
    for p in [2, 3, 5, 7]:
        if p in prime_to_idx:
            target_mask |= (1 << prime_to_idx[p])
            
    # 3. Iterate M from 211 upwards
    # We expect to find a solution for M <= 230
    # Basis for Gaussian elimination
    # We store basis vectors and the subset of numbers that sum to them
    # basis is a list of (mask, subset_list)
    # We want to keep basis vectors linearly independent and reduced
    
    # To efficiently check if target is in span, we maintain a reduced basis.
    # Actually, simpler: just maintain a list of all masks seen? No, dimension is ~50.
    # Let's maintain a standard row-echelon form basis.
    
    def get_mask(x):
        m = 0
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                cnt = 0
                while n % p == 0:
                    n //= p
                    cnt += 1
                if cnt % 2 == 1:
                    m |= (1 << prime_to_idx[p])
        if n > 1:
            # n is prime
            if n in prime_to_idx:
                m |= (1 << prime_to_idx[n])
            else:
                # Should not happen if primes list is complete
                pass
        return m

    # We will iterate M and maintain the basis incrementally
    # However, adding a vector to a reduced basis is O(dim).
    # Let's just collect all masks for a given M and do Gaussian elimination from scratch?
    # M is small (up to 230, so 20 numbers). 20 numbers is tiny.
    # So for each M, we can just take the list of masks for 211..M and run elimination.
    
    for M in range(211, 10211):
        if time.time() > deadline:
            break
            
        # Collect masks for numbers 211 to M
        masks = []
        nums = []
        for x in range(211, M + 1):
            masks.append(get_mask(x))
            nums.append(x)
            
        # Gaussian elimination to find if target_mask is in span
        # We want to represent target_mask as a sum of some masks.
        # Let's perform elimination on the system:
        # We have vectors v_1, ..., v_k. We want to find coefficients c_i such that sum c_i v_i = target.
        # This is equivalent to checking if target is in the span.
        
        # Standard approach:
        # Maintain a list of basis vectors in row echelon form.
        # For each vector, reduce it by existing basis.
        # If it becomes 0, it's dependent.
        # If not, add to basis.
        # We also need to track the combination.
        
        # Let's use a dictionary or list to track the "parent" combination.
        # basis[i] = mask
        # combo[i] = list of indices in 'nums' that sum to basis[i]
        
        basis = []
        combos = []
        
        # To make reduction fast, we can keep basis sorted by highest bit set.
        # But k is small (< 50).
        
        for idx, m in enumerate(masks):
            curr_mask = m
            curr_combo = [idx]
            
            # Reduce curr_mask using basis
            # Iterate through basis, if highest bit of curr_mask matches highest bit of basis vector, XOR them
            # To do this efficiently, we need basis sorted by highest bit.
            
            # Let's just iterate and check
            # A simpler method for small k:
            # Maintain a list of basis vectors.
            # For each new vector, try to reduce it.
            
            changed = True
            while changed:
                changed = False
                for i, b_mask in enumerate(basis):
                    # Check if b_mask helps reduce curr_mask
                    # Find highest bit in curr_mask
                    if curr_mask == 0:
                        break
                    # Find highest set bit in curr_mask
                    # Python 3.11+ has int.bit_length
                    # Let's find the pivot
                    # We want to eliminate the highest bit of curr_mask if possible
                    # Let's find the highest bit in curr_mask
                    hb = curr_mask.bit_length() - 1
                    # Check if any basis vector has this bit set
                    # To make this O(1), we should maintain a map from pivot_bit to basis_index
                    # Let's re-implement with a pivot map
                    
                    # Actually, for k <= 100, O(k^2) is fine.
                    pass
            
            # Let's restart the inner loop logic properly with a pivot map
            # This is getting complicated to do incrementally in the loop.
            # Since M is small, let's just do a full Gaussian elimination for each M.
            # Complexity: O(M * dim^2). M <= 100, dim <= 50. 100 * 2500 = 250,000 ops. Very fast.
            
        # Re-doing the logic inside the M loop with full elimination
        # This is cleaner.
        
        # Build matrix
        # We want to solve A x = b over F2
        # A is k x n (k=dim, n=M-210)
        # Actually, we just need to check if b is in column space of A.
        # This is equivalent to row space of A^T containing b? No.
        # It's equivalent to b being in the span of the rows of A (if we treat masks as row vectors).
        # Yes, span of masks.
        
        # Gaussian Elimination on the set of vectors {masks} + {target}
        # If target reduces to 0, it's in the span.
        # We need to track which original vectors were used.
        
        # Let's use a list of (mask, indices_set)
        # indices_set is a set of indices from 'nums'
        
        vecs = []
        for i, m in enumerate(masks):
            vecs.append([m, {i}])
            
        # Add target as a special vector to be reduced
        # We want to see if target can be reduced to 0 by XOR-ing with some of vecs.
        # Let's add target to the list and try to eliminate all of them?
        # No, we want to eliminate target using the others.
        
        # Standard Gaussian Elimination to find if target is in span:
        # 1. Create a list of vectors: masks.
        # 2. Perform row reduction to get a basis.
        # 3. Try to reduce target using this basis.
        # 4. If target becomes 0, it is in the span.
        
        # Let's maintain basis as list of (mask, combo_set)
        # And keep them in reduced row echelon form (RREF) to make reduction easy.
        
        basis = []
        
        def reduce_vec(m, combo):
            # Reduce m using basis
            # Returns (new_m, new_combo)
            for i, (b_mask, b_combo) in enumerate(basis):
                if b_mask == 0:
                    continue
                # Find pivot of b_mask (highest bit)
                # We maintain basis sorted by pivot?
                # Let's just check if the highest bit of m is covered by b_mask
                if m == 0:
                    break
                hb = m.bit_length() - 1
                # Check if b_mask has this bit set
                if (b_mask >> hb) & 1:
                    m ^= b_mask
                    # Update combo: XOR the sets
                    # Since we are working over F2, adding a vector corresponds to symmetric difference of indices
                    # However, if an index appears in both, it cancels out (count 2 -> 0)
                    # We can use sets.
                    # new_combo = combo ^ b_combo
                    # In Python, set symmetric difference is ^
                    combo = combo ^ b_combo
            return m, combo

        # Insert vectors into basis
        for i, m in enumerate(masks):
            curr_m, curr_c = m, {i}
            curr_m, curr_c = reduce_vec(curr_m, curr_c)
            if curr_m != 0:
                # Add to basis
                basis.append([curr_m, curr_c])
                # To make future reductions faster, we could sort basis by pivot bit
                # But size is small.
                
        # Now try to reduce target
        t_m = target_mask
        t_c = set()
        t_m, t_c = reduce_vec(t_m, t_c)
        
        if t_m == 0:
            # Solution found
            # The subset of indices is t_c
            # Convert indices to numbers
            selected_nums = [nums[i] for i in t_c]
            
            # Verify
            if not selected_nums:
                # Empty set product is 1. 210*1=210 not square.
                # But target mask non-zero implies non-empty solution usually.
                pass
                
            prod = 1
            for n in selected_nums:
                prod *= n
            
            val = 210 * prod
            s = math.isqrt(val)
            if s * s == val:
                # Check distinct and range
                if len(set(selected_nums)) == len(selected_nums):
                    if all(210 < n <= 10210 for n in selected_nums):
                        # Return list of ints
                        return [int(n) for n in selected_nums]

    # Fallback: k=1 solution
    # x = 210 * t^2
    # 210 < x <= 10210
    # t=1 -> 210 (not > 210)
    # t=2 -> 840
    # 840 - 210 = 630.
    # This is much worse than 21.
    # But if no small solution found, return this.
    
    for t in range(2, 100):
        x = 210 * t * t
        if x > 10210:
            break
        if x > 210:
            return [x]
            
    return [840]
