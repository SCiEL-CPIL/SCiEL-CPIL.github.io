import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 210
    max_val = 10210

    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Precompute parity masks using a sieve-like approach
    # mask[x] = XOR of masks of its prime factors (with multiplicity)
    masks = [0] * (max_val + 1)
    for p in primes:
        p_bit = 1 << prime_index[p]
        for multiple in range(p, max_val + 1, p):
            # Count how many times p divides multiple
            cnt = 0
            t = multiple
            while t % p == 0:
                cnt += 1
                t //= p
            if cnt % 2 == 1:
                masks[multiple] ^= p_bit

    # Parity mask of 210
    target = masks[210]

    def verify_solution(sol):
        """Verify that product of sol * 210 is a perfect square."""
        prod = 210
        for x in sol:
            prod *= x
        s = math.isqrt(prod)
        return s * s == prod

    def find_solution_for_M(M):
        """
        Find a subset of {211, ..., M} whose XOR of masks equals target,
        with at most 100 elements. Returns None if not found.
        Uses Gaussian elimination to find one solution, then tries to reduce size.
        """
        # Build basis with history tracking
        basis_vec = {}   # pivot -> vector
        basis_hist = {}  # pivot -> history mask

        for M2 in range(211, M + 1):
            v = masks[M2]
            h = 1 << (M2 - 211)
            tv = v
            th = h
            while tv:
                p = tv.bit_length() - 1
                if p in basis_vec:
                    tv ^= basis_vec[p]
                    th ^= basis_hist[p]
                else:
                    basis_vec[p] = tv
                    basis_hist[p] = th
                    break

        # Check if target is in span
        tv = target
        th = 0
        while tv:
            p = tv.bit_length() - 1
            if p in basis_vec:
                tv ^= basis_vec[p]
                th ^= basis_hist[p]
            else:
                return None

        if tv != 0:
            return None

        # Extract initial solution
        sol_set = set()
        hh = th
        while hh:
            lsb = hh & -hh
            idx = lsb.bit_length() - 1
            sol_set.add(211 + idx)
            hh ^= lsb

        if len(sol_set) <= 100:
            sol = list(sol_set)
            if verify_solution(sol):
                return sol

        # Try to reduce solution size using kernel vectors
        # Find kernel vectors (nonzero vectors in null space)
        # A vector in kernel has XOR of masks = 0
        # We can find kernel vectors by looking at dependencies
        
        # Build full row-reduced form to find kernel vectors
        # Actually, let's use a different approach:
        # For each element in sol_set, try removing it and see if we can compensate
        # with a small set of other elements
        
        # Simpler approach: try greedy removal
        # For each element e in sol_set, check if there's a small subset of 
        # (sol_set \ {e}) union (candidates not in sol_set) that can replace e
        
        # Even simpler: try to find pairs/triples that cancel out
        # If a, b, c are in sol_set and mask[a]^mask[b]^mask[c] = 0, we can remove all three
        # Or if mask[a] = mask[b], we can remove both (they cancel)
        
        # Find duplicate masks in sol_set
        mask_to_elements = {}
        for e in sol_set:
            m = masks[e]
            if m not in mask_to_elements:
                mask_to_elements[m] = []
            mask_to_elements[m].append(e)
        
        removed = set()
        for m, elems in mask_to_elements.items():
            if m == 0:
                # These elements don't affect the XOR, remove them
                for e in elems:
                    if e not in removed:
                        removed.add(e)
            elif len(elems) >= 2:
                # Keep only one, remove the rest (pairs cancel)
                keep = elems[0]
                for e in elems[1:]:
                    if e not in removed:
                        removed.add(e)
        
        if removed:
            sol_set -= removed
            if len(sol_set) <= 100:
                sol = list(sol_set)
                if verify_solution(sol):
                    return sol
        
        # Try triple cancellations: find a,b,c in sol_set with mask[a]^mask[b]^mask[c]=0
        # This is expensive; only do if still > 100
        if len(sol_set) > 100:
            # Look for small kernel vectors involving elements in sol_set
            # For each pair in sol_set, check if their XOR matches mask of some other element in sol_set
            sol_list = list(sol_set)
            mask_map = {}
            for e in sol_list:
                mask_map[masks[e]] = e
            
            new_removed = set()
            for i in range(len(sol_list)):
                if sol_list[i] in removed or sol_list[i] in new_removed:
                    continue
                for j in range(i + 1, len(sol_list)):
                    if sol_list[j] in removed or sol_list[j] in new_removed:
                        continue
                    xor_mask = masks[sol_list[i]] ^ masks[sol_list[j]]
                    if xor_mask in mask_map:
                        k = mask_map[xor_mask]
                        if k != sol_list[i] and k != sol_list[j] and k not in removed and k not in new_removed:
                            new_removed.add(sol_list[i])
                            new_removed.add(sol_list[j])
                            new_removed.add(k)
        
        if new_removed:
            sol_set -= new_removed
            if len(sol_set) <= 100:
                sol = list(sol_set)
                if verify_solution(sol):
                    return sol
        
        # If still too large, return None (try larger M)
        if len(sol_set) <= 100:
            sol = list(sol_set)
            if verify_solution(sol):
                return sol
        
        return None

    # Main search: find minimum M where a valid solution exists
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        
        sol = find_solution_for_M(M)
        if sol is not None:
            return [int(x) for x in sol]

    # Fallback (should not be reached)
    return [840]
