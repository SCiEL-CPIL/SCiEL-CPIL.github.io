import numpy as np

def construct():
    N = 10000000

    # Compute smallest prime factor (SPF) sieve
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Helper to compute sigma(n) using SPF
    def sigma(n):
        if n <= 1:
            return 1 if n == 1 else 0
        result = 1
        temp = n
        while temp > 1:
            p = spf[temp]
            power = 1
            while temp % p == 0:
                temp //= p
                power *= p
            # sigma(p^k) = (p^(k+1) - 1) / (p - 1)
            result *= (power * p - 1) // (p - 1)
        return result

    # Search over k = |2a - 3b| starting from 1
    # For a given k, we want to find a, b in [1, N] with 2a - 3b = ±k and sigma(a) = sigma(b)
    # Minimizing |a/b - 3/2| = k/(2b), so for fixed k we want largest b
    # 2a - 3b = k => a = (3b + k)/2, requires 3b + k even
    # 2a - 3b = -k => a = (3b - k)/2, requires 3b - k even, and a >= 1 => b >= (k+1)/3
    
    # We'll iterate k from 1 upward, and for each k iterate b from N down
    # But that's too slow. Instead, for each k, the constraint is:
    # Case 1: 3b + k even => b and k have opposite parity (since 3b parity = b parity)
    #   Actually 3b + k even means b + k even (since 3b ≡ b mod 2)
    #   a = (3b + k) // 2, need 1 <= a <= N => 2 <= 3b+k <= 2N => (2-k)//3 <= b <= (2N-k)//3
    # Case 2: 3b - k even => b - k even => b and k same parity
    #   a = (3b - k) // 2, need 1 <= a <= N => 2 <= 3b-k <= 2N => (k+2)//3 <= b <= (2N+k)//3
    #   Also b >= 1
    
    best_error = float('inf')
    best_pair = None

    # We want smallest k, and for each k, largest b to minimize k/(2b)
    # For each k, the maximum b in each case:
    # Case 1: b_max = floor((2N - k) / 3), with b ≡ k (mod 2)
    # Case 2: b_max = floor((2N + k) / 3), with b ≡ k (mod 2), and b >= ceil((k+2)/3)
    
    max_k_to_check = 100000  # reasonable upper bound
    
    for k in range(1, max_k_to_check + 1):
        # Case 2: 2a - 3b = -k, so a = (3b-k)/2, b and k same parity
        # b_max for case 2: floor((2N + k) / 3)
        b_max_c2 = (2 * N + k) // 3
        b_min_c2 = max(1, (k + 2) // 3)
        
        # Find largest b <= b_max_c2 with b ≡ k (mod 2) and b >= b_min_c2
        b_candidate = b_max_c2
        if b_candidate % 2 != k % 2:
            b_candidate -= 1
        if b_candidate >= b_min_c2:
            a = (3 * b_candidate - k) // 2
            if 1 <= a <= N and a != b_candidate:
                sa = sigma(a)
                sb = sigma(b_candidate)
                if sa == sb:
                    error = k / (2.0 * b_candidate)
                    if error < best_error:
                        best_error = error
                        best_pair = (a, b_candidate)
                        # Since we want to minimize k/(2b), and k is increasing,
                        # we could break early if we find a very good one, but let's continue
                        # Actually for fixed k, this is the best b. For larger k, error increases unless b increases enough.
                        # We can break if k/(2*b_candidate) >= best_error is impossible for larger k... 
                        # But let's just check a few more k values.
        
        # Case 1: 2a - 3b = k, so a = (3b+k)/2, b and k opposite parity
        # b_max for case 1: floor((2N - k) / 3)
        b_max_c1 = (2 * N - k) // 3
        b_min_c1 = max(1, 1)
        
        if b_max_c1 >= 1:
            b_candidate = b_max_c1
            if b_candidate % 2 == k % 2:
                b_candidate -= 1
            if b_candidate >= b_min_c1:
                a = (3 * b_candidate + k) // 2
                if 1 <= a <= N and a != b_candidate:
                    sa = sigma(a)
                    sb = sigma(b_candidate)
                    if sa == sb:
                        error = k / (2.0 * b_candidate)
                        if error < best_error:
                            best_error = error
                            best_pair = (a, b_candidate)
        
        # If we've found a pair, check if further k can improve
        # For k' > k, min error is k'/(2*N) (if b can be N). 
        # If k/(2*b_candidate) is already very small, we might be done.
        # But let's be safe and check up to max_k or until no improvement for a while.
        if best_pair is not None:
            # Theoretical minimum for any larger k is k/(2*N)
            if (k + 1) / (2.0 * N) >= best_error:
                break

    if best_pair is None:
        # Fallback: find any valid pair
        # Search for small pairs
        for b in range(1, N + 1):
            for a in range(1, N + 1):
                if a == b:
                    continue
                if 2 * a == 3 * b:
                    continue
                if sigma(a) == sigma(b):
                    best_pair = (a, b)
                    best_error = abs(a / b - 1.5)
                    break
            if best_pair:
                break
    
    return [int(best_pair[0]), int(best_pair[1])]
