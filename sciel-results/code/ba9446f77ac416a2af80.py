import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Generate powerful numbers up to a reasonable limit.
    # We need enough powerful numbers to have a good density for verification.
    # L = 10^12 gives ~ 10^4 powerful numbers. 
    # Verification of N against 10^4 numbers is fast (O(M)).
    # If we generate up to 10^14, M ~ 10^5. Verification is still fast.
    # Let's try L = 10^12 first to ensure speed, then maybe expand if time permits.
    # Actually, to find N up to 10^18, we need powerful numbers up to 10^18.
    # But we can't generate all of them.
    
    # Strategy:
    # A. Generate powerful numbers up to L_gen = 10^12.
    # B. Identify "rich" powerful numbers: those that are sums of other powerful numbers? No.
    #    Rich powerful numbers are likely those of the form $u^2 v$ where $v$ has many small prime factors?
    #    Actually, the number of representations $r(N)$ is high if $N$ can be written as $a+b$ in many ways.
    #    This happens if $a$ and $b$ are "small" relative to $N$? No.
    #    It happens if $N$ is in the "core" of the distribution of sums.
    
    # Let's generate powerful numbers up to L = 10^12.
    L = 10**12
    
    def generate_powerful(limit):
        powerful = set()
        max_b = int(limit ** (1.0/3.0)) + 2
        
        # Precompute prime squares for square-free check
        # Need primes up to sqrt(max_b) ~ 1000
        def get_prime_squares(limit):
            is_prime = [True] * (limit + 1)
            if limit >= 0: is_prime[0] = False
            if limit >= 1: is_prime[1] = False
            for i in range(2, int(limit ** 0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, limit + 1, i):
                        is_prime[j] = False
            return [i*i for i in range(2, limit + 1) if is_prime[i]]
        
        max_b_sqrt = int(max_b ** 0.5) + 2
        prime_squares = get_prime_squares(max_b_sqrt)
        
        def is_square_free(n):
            if n < 2:
                return True
            for ps in prime_squares:
                if ps > n:
                    break
                if n % ps == 0:
                    return False
            return True

        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b3 = b ** 3
            if b3 > limit:
                break
            max_a = int(math.isqrt(limit // b3))
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)
        
        return sorted(powerful)

    P = generate_powerful(L)
    P_np = np.array(P, dtype=np.int64)
    P_set = set(P)
    
    # 2. Generate candidate N values.
    # We want N such that many pairs (a,b) sum to N.
    # If we pick N from the sums of the "most central" powerful numbers, we might find high counts.
    # Or, we can pick N values that are sums of powerful numbers with high "multiplicity" in their generation?
    
    # Alternative:
    # Pick a set of "seed" powerful numbers S.
    # For each pair (s1, s2) in S, compute N = s1 + s2.
    # Then for each unique N, count how many pairs in P sum to N.
    # To make this efficient, we only verify Ns that appear frequently in the seed sums.
    
    # Let's take the top K powerful numbers by "divisibility" or just the largest ones?
    # Actually, small powerful numbers are more dense.
    # Let's take the first K powerful numbers (smallest ones) and the largest K?
    # Sums of small numbers are small. Sums of large numbers are large.
    
    # Let's try a mixed approach:
    # Sample M_sample pairs from P to generate candidate Ns.
    # Count frequencies of these Ns.
    # Then verify the top candidates against the full P.
    
    num_samples = 200000  # 200k samples
    # We need to be careful with memory and time.
    # Generating 200k sums is fast.
    
    # To bias towards "good" Ns, maybe we should sample from the middle of the distribution?
    # Or just random? Random might miss the peaks.
    
    # Better:
    # The number of representations is related to the additive energy.
    # Let's just generate a large set of candidate Ns by summing subsets of P.
    # If we sum the first 100 powerful numbers with the next 100, we get 10k candidates.
    # Let's do a few blocks.
    
    candidates = set()
    
    # Block 1: Small + Small
    small_P = P[:500]
    for i in range(len(small_P)):
        for j in range(i, len(small_P)):
            candidates.add(small_P[i] + small_P[j])
            
    # Block 2: Small + Medium
    med_P = P[500:1500]
    for i in range(len(small_P)):
        for j in range(len(med_P)):
            candidates.add(small_P[i] + med_P[j])
            
    # Block 3: Medium + Medium
    for i in range(len(med_P)):
        for j in range(i, len(med_P)):
            candidates.add(med_P[i] + med_P[j])
            
    # Block 4: Large + Large (from the end of P)
    large_P = P[-500:]
    for i in range(len(large_P)):
        for j in range(i, len(large_P)):
            candidates.add(large_P[i] + large_P[j])
            
    # Block 5: Small + Large
    for i in range(len(small_P)):
        for j in range(len(large_P)):
            candidates.add(small_P[i] + large_P[j])
            
    # Also, let's add some random large Ns to see if big Ns are better?
    # But we can't generate powerful numbers > L easily for verification.
    # Wait, if N > 2*L, we can't verify it with P (which only goes up to L).
    # So we are limited to N <= 2*L = 2*10^12.
    # Is that enough? The task allows N up to 10^18.
    # If the best N is > 2*10^12, we miss it.
    # However, the density of powerful numbers drops.
    # It is likely that the maximum number of representations is achieved for smaller Ns where the density is higher?
    # Let's assume N <= 2*10^12 is a good search space.
    
    # 3. Verify candidates.
    # For each candidate N, count pairs (a,b) in P such that a+b=N and a<=b.
    # We can do this efficiently using binary search or NumPy.
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Sort candidates by some heuristic?
    # Or just iterate and keep track of best.
    # Number of candidates:
    # 500*500/2 = 125k
    # 500*1000 = 500k
    # 1000*1000/2 = 500k
    # 500*500/2 = 125k
    # 500*500 = 250k
    # Total ~ 1.5M candidates.
    # Verification of 1.5M candidates * O(log M) or O(M) is too slow.
    # O(M) verification for 1.5M candidates is 1.5M * 10^4 = 1.5 * 10^10 ops. Too slow.
    
    # We need to filter candidates.
    # Only verify candidates that are "likely" to have many reps.
    # How to estimate?
    # We can pre-compute a "score" for each candidate based on the number of times it appeared in our sampling?
    # But we didn't count frequencies in the blocks above, we just added to a set.
    
    # Let's change strategy:
    # Count frequencies of sums in a smaller sample, then verify the top K.
    
    counts_sample = defaultdict(int)
    
    # Sample 100k pairs
    indices = np.random.choice(len(P), size=(100000, 2), replace=True)
    # This is 100k random pairs.
    # Sums = P_np[indices[:,0]] + P_np[indices[:,1]]
    # But we want a<=b? For counting reps, order doesn't matter for the sum value.
    # We just want to find Ns that are "popular".
    
    sums = P_np[indices[:,0]] + P_np[indices[:,1]]
    
    # Count unique sums
    # Using numpy unique with return_counts
    unique_sums, counts = np.unique(sums, return_counts=True)
    
    # Get top 1000 candidates
    top_indices = np.argsort(counts)[::-1][:1000]
    top_candidates = unique_sums[top_indices]
    
    # Now verify these 1000 candidates against the full P.
    # For each N in top_candidates:
    #   Find all a in P such that N-a is in P and a <= N-a.
    #   This can be done by iterating a in P (or a subset) and checking if N-a is in P_set.
    #   O(M) per candidate. 1000 * 10^4 = 10^7 ops. Fast.
    
    for N in top_candidates:
        if N > 2 * L:
            continue
        if N < 2:
            continue
            
        pairs = []
        # Iterate a in P
        # To speed up, we can stop when a > N/2
        # P is sorted.
        # Find index where P > N/2
        # Use binary search
        # Or just loop and break.
        
        # Let's use a loop with break
        for a in P:
            if a > N // 2:
                break
            b = N - a
            if b in P_set:
                pairs.append([a, b])
                
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
            
        if time.time() > deadline - 10:
            break
            
    # If we didn't find anything better than 1, we should probably try a different set of candidates.
    # But 1 is the baseline (N=2, 1+1).
    # Usually we find > 1.
    
    # One more check: what if the best N is not in the top 1000 of random samples?
    # We can do a second pass with different sampling or different blocks.
    # But time is limited.
    
    # Let's also check the candidates from the "blocks" we generated earlier if we have time.
    # The set 'candidates' has ~1.5M items.
    # We can't verify all.
    # But we can verify the top 500 from the blocks?
    # We didn't count frequencies in blocks.
    # Let's just stick to the top 1000 from random sampling.
    # Random sampling of sums should capture the "dense" regions of the sum distribution.
    
    return {'N': best_N, 'pairs': best_pairs}
