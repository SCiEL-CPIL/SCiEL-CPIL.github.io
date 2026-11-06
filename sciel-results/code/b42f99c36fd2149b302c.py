import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # 1. Generate primes <= 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    num_primes = len(primes)
    
    # We will try to cover as much as possible.
    # Let's set a high limit for the "covered" array to allow dynamic extension.
    # But for speed, we'll precompute lists up to a max L, say 5000.
    L_MAX = 5000
    
    # Precompute cover_lists for all primes and residues up to L_MAX
    # cover_lists[p_idx][r] = list of numbers
    cover_lists = []
    for p_idx, p in enumerate(primes):
        residues = [[] for _ in range(p)]
        for r in range(p):
            start = r if r > 0 else p
            # Only include numbers up to L_MAX
            residues[r] = [x for x in range(start, L_MAX + 1, p)]
        cover_lists.append(residues)
        
    best_classes = []
    best_L = 0
    
    # Greedy construction
    covered = [False] * (L_MAX + 1)
    cost = 0.0
    used_primes = set()
    classes = []
    
    # We want to cover 1, 2, 3...
    # We iterate through k = 1 to L_MAX
    # If k is not covered, we must add a class to cover it.
    # We pick the class (p, r=k%p) that maximizes future coverage (gain)
    # subject to cost constraint.
    
    for k in range(1, L_MAX + 1):
        if time.time() - start_time > time_limit:
            break
            
        if covered[k]:
            # Update best if we are extending the prefix
            if k > best_L:
                best_L = k
                best_classes = classes.copy()
            continue
        
        # k is uncovered. We need to cover it.
        best_p_idx = -1
        best_r = -1
        best_gain = -1
        
        for p_idx in range(num_primes):
            if p_idx in used_primes:
                continue
            p = primes[p_idx]
            r = k % p
            
            new_cost = cost + 1.0 / p
            if new_cost > 1.0:
                continue
            
            # Calculate gain: number of currently uncovered numbers in cover_lists[p_idx][r]
            # Optimization: We only care about numbers >= k?
            # No, we care about any uncovered number, but since we are scanning k upwards,
            # numbers < k are already covered (by definition of k being the current pointer?
            # No, k is the current index in the loop. If covered[k] is False,
            # it means no previous class covered k.
            # But previous classes might have covered numbers > k.
            # So "gain" should count uncovered numbers in the list.
            
            gain = 0
            for x in cover_lists[p_idx][r]:
                if not covered[x]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_p_idx = p_idx
                best_r = r
                
        if best_p_idx == -1:
            # Cannot cover k with any available prime within budget
            break
            
        # Apply the best choice
        p = primes[best_p_idx]
        r = best_r
        classes.append([p, r])
        used_primes.add(best_p_idx)
        cost += 1.0 / p
        
        # Update covered array
        for x in cover_lists[best_p_idx][best_r]:
            covered[x] = True
            
        # Check if we covered k (should be true)
        # And update best_L if the prefix is extended
        # The prefix is the largest L such that 1..L are all covered.
        # Since we just covered k, and 1..k-1 were covered (by loop invariant),
        # the new prefix is at least k.
        # But maybe we covered k+1, k+2... as well?
        # We can check how far the prefix extends.
        new_L = k
        while new_L < L_MAX and covered[new_L + 1]:
            new_L += 1
            
        if new_L > best_L:
            best_L = new_L
            best_classes = classes.copy()

    if not best_classes:
        # Fallback: Return a valid but minimal solution?
        # Or empty?
        # Empty is valid (L=0).
        return []
        
    return best_classes
# EVOLVE-BLOCK-END
