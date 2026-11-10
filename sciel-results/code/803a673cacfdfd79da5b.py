import time

def construct():
    S = 123457
    deadline = time.time() + 80  # 80s safety margin (evaluator kills at 90s)
    
    # First, try the smooth-number approach for a quick answer
    # Generate B-smooth numbers and check for pairs summing to S
    # Start with small B and increase
    
    # Precompute primes up to S for reference
    # Use a simple sieve for smallest prime factor to get largest prime factor
    P = [1] * (S + 1)
    for i in range(2, S + 1):
        if P[i] == 1:  # i is prime
            for j in range(i, S + 1, i):
                P[j] = i
    
    # Smooth-number approach: try increasing B
    # For each B, generate all B-smooth numbers <= S
    # Check if any pair sums to S
    # The first B where this works gives max(P(a),P(b)) <= B
    
    # Get primes up to S
    primes = []
    is_prime = [True] * (S + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, S + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, S + 1, i):
                is_prime[j] = False
    
    # Try increasing smoothness bounds
    best_score = float('inf')
    best_a = 1
    best_b = S - 1
    
    # Quick smooth-number search
    for B_idx in range(len(primes)):
        if time.time() > deadline:
            break
        B = primes[B_idx]
        
        # Generate all B-smooth numbers <= S
        smooths = {1}
        current = [1]
        p = B
        # Use all primes up to B
        for pi in range(B_idx + 1):
            p = primes[pi]
            new_vals = []
            for x in current:
                val = x
                while val * p <= S:
                    val *= p
                    new_vals.append(val)
                    smooths.add(val)
            current.extend(new_vals)
        
        # Check pairs
        found = False
        for a in smooths:
            if a >= 1 and a <= S - 1:
                b = S - a
                if b >= 1 and b in smooths:
                    # Both a and b are B-smooth, so max(P(a),P(b)) <= B
                    score = B
                    if score < best_score:
                        best_score = score
                        best_a = a
                        best_b = b
                    found = True
                    break
        
        if found:
            # We found a pair with max prime factor <= B
            # Since we're going in increasing order of B, this is optimal
            # But let's verify with the P array to be sure
            pa = P[best_a]
            pb = P[best_b]
            actual_score = max(pa, pb)
            if actual_score <= B:
                return [best_a, best_b]
    
    # Fallback: full linear scan
    for a in range(1, S // 2 + 1):
        if time.time() > deadline:
            break
        b = S - a
        pa = P[a]
        pb = P[b]
        score = pa if pa > pb else pb
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            if best_score == 2:
                break
    
    return [best_a, best_b]
