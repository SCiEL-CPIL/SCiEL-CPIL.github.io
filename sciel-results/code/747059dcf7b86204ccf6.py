"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    def squarefree_kernel(n):
        """Return the squarefree kernel of n."""
        res = 1
        i = 2
        while i * i <= n:
            cnt = 0
            while n % i == 0:
                n //= i
                cnt += 1
            if cnt % 2 == 1:
                res *= i
            i += 1
        if n > 1:
            res *= n
        return res

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # 210 = 2 * 3 * 5 * 7 (squarefree)
    # We need: sf(210 * prod) = 1, i.e., sf(prod) = 210
    # Equivalently, the XOR of prime signatures must equal {2,3,5,7}
    
    # Use signature as integer bitmask over primes relevant to our range
    # Primes up to ~10210 that could appear: let's use the actual squarefree kernel approach
    
    target_kernel = 210  # sf(210) = 210
    
    # Search from smallest possible max value upward
    # For each candidate max M (from 211 upward), check if we can find a subset
    # of {211, ..., M} whose product has sf = 210
    
    # Precompute squarefree kernels
    # We'll extend the range gradually
    
    # First, try small sizes with expanding range
    best_result = None
    best_max = float('inf')
    
    # Try size 1: need sf(n) = 210, so n = 210 * k^2 for some k
    # n > 210, so k >= 1: n = 210, 840, 1890, 3360, 5250, 7560, 10290 (too big)
    # 210 is not > 210, so try 840, 1890, 3360, 5250, 7560
    for k in range(1, 100):
        n = 210 * k * k
        if n > 210 and n <= 10210:
            if best_result is None or n < best_max:
                best_max = n
                best_result = [n]
    
    # Try size 2: need sf(a) * sf(b) to be a perfect square times 210
    # i.e., sf(a * b) = 210
    # More precisely: sf(a) XOR sf(b) in prime exponent parity = sf(210)
    # Using the kernel: sf(a*b) = sf(sf(a)*sf(b)) should equal 210
    
    # Let's search by increasing max value
    # For max = M, candidates are 211..M
    # Try all pairs and triples
    
    # Efficient approach: for each M from 212 upward, add M to candidate pool
    # and check if any subset of the pool has kernel 210
    
    # But that's expensive. Instead, let's be smart:
    # Precompute kernels for a range, then search
    
    max_search = 400  # search up to here initially
    candidates = list(range(211, max_search + 1))
    kernels = {}
    for n in candidates:
        kernels[n] = squarefree_kernel(n)
    
    # Size 1 check (already done above, but verify)
    for n in candidates:
        if kernels[n] == target_kernel:
            if best_result is None or n < best_max:
                best_max = n
                best_result = [n]
    
    # Size 2: sf(a*b) = 210
    # sf(a*b) = sf(kernels[a] * kernels[b])
    # We need sf(kernels[a] * kernels[b]) == 210
    # Since kernels are squarefree, sf(kernels[a] * kernels[b]) = kernels[a] * kernels[b] / gcd(kernels[a], kernels[b])^2... 
    # Actually for squarefree x, y: sf(x*y) = x*y/gcd(x,y)^2 = (x/gcd(x,y))*(y/gcd(x,y))
    # This equals 210 iff x*y/gcd(x,y)^2 = 210
    
    # Simpler: just compute and check
    n_cands = len(candidates)
    for i in range(n_cands):
        if time.time() > deadline:
            break
        a = candidates[i]
        ka = kernels[a]
        for j in range(i + 1, n_cands):
            b = candidates[j]
            kb = kernels[b]
            # Compute sf(ka * kb)
            prod = ka * kb
            if squarefree_kernel(prod) == target_kernel:
                if b < best_max:
                    best_max = b
                    best_result = [a, b]
                break  # no need to check larger b for this a
    
    # Size 3
    for i in range(n_cands):
        if time.time() > deadline:
            break
        a = candidates[i]
        ka = kernels[a]
        for j in range(i + 1, n_cands):
            if time.time() > deadline:
                break
            b = candidates[j]
            kb = kernels[b]
            kab = squarefree_kernel(ka * kb)
            for k in range(j + 1, n_cands):
                c = candidates[k]
                kc = kernels[c]
                if c >= best_max:
                    break
                if squarefree_kernel(kab * kc) == target_kernel:
                    if c < best_max:
                        best_max = c
                        best_result = [a, b, c]
                    break  # no need for larger c
    
    # If we found something with max <= 231 (score <= 21), we're at least tied
    # Try to improve by searching smaller ranges more carefully
    
    # Let's do a focused search: for each possible max value M from 211 to best_max-1,
    # check if there's a valid subset with all elements <= M
    
    if best_result is not None:
        # Try to find better (smaller max)
        for M in range(211, best_max):
            if time.time() > deadline:
                break
            sub_cands = [n for n in range(211, M + 1)]
            sub_kernels = {n: squarefree_kernel(n) for n in sub_cands}
            n_sub = len(sub_cands)
            
            # Size 1
            found = False
            for n in sub_cands:
                if sub_kernels[n] == target_kernel:
                    best_max = M
                    best_result = [n]
                    found = True
                    break
            if found:
                break
            
            # Size 2
            found = False
            for i in range(n_sub):
                a = sub_cands[i]
                ka = sub_kernels[a]
                for j in range(i + 1, n_sub):
                    b = sub_cands[j]
                    kb = sub_kernels[b]
                    if squarefree_kernel(ka * kb) == target_kernel:
                        best_max = M
                        best_result = [a, b]
                        found = True
                        break
                if found:
                    break
            if found:
                break
            
            # Size 3
            found = False
            for i in range(n_sub):
                a = sub_cands[i]
                ka = sub_kernels[a]
                for j in range(i + 1, n_sub):
                    b = sub_cands[j]
                    kb = sub_kernels[b]
                    kab = squarefree_kernel(ka * kb)
                    for k in range(j + 1, n_sub):
                        c = sub_cands[k]
                        kc = sub_kernels[c]
                        if squarefree_kernel(kab * kc) == target_kernel:
                            best_max = M
                            best_result = [a, b, c]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    
    if best_result is None:
        # Fallback: known valid solution
        best_result = [224, 240]
    
    # Verify the result
    prod = 210
    for x in best_result:
        prod *= x
    if not is_perfect_square(prod):
        # Should not happen, but fallback
        best_result = [224, 240]
    
    # Ensure types are correct
    return [int(x) for x in best_result]
