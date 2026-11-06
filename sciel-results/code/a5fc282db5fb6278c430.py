"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    if limit < 1:
        return [1]
    powerful = set()
    powerful.add(1)
    
    # b^3 * a^2 <= limit
    # b up to limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # a^2 <= limit / b3
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def _is_powerful(n):
    """Check if n is powerful by exact factorization."""
    if n <= 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count == 1:
                return False
        d += 1 if d == 2 else 2  # skip even numbers after 2
    if temp > 1:
        # temp is a prime factor appearing once
        return False
    return True

def _count_representations(N, powerful_list, powerful_set):
    """Count number of pairs (a,b) with a<=b, a+b=N, both powerful."""
    pairs = []
    # Iterate through powerful numbers a <= N//2
    # Since powerful_list is sorted, we can stop when a > N//2
    for a in powerful_list:
        if a > N // 2:
            break
        b = N - a
        if b in powerful_set:
            pairs.append([a, b])
    return pairs

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate powerful numbers up to a large limit
    # 10^15 is safe and fast to generate (~10^5 numbers)
    # The number of powerful numbers up to N is approx N^0.5? No, N^(1/3) * N^(1/2) ? 
    # Count is roughly sum_{b<=N^(1/3)} N^(1/2)/b^(3/2) ~ N^(1/2) * N^(1/6) = N^(2/3)? 
    # Actually, the count of powerful numbers up to X is O(X^(1/3) * X^(1/2)) is wrong.
    # It is dominated by squares. There are X^(1/2) squares. 
    # Plus cubes etc.
    # For X=10^15, sqrt(X)=10^7.5 ~ 3*10^7. This might be too many to store in a list if we go higher.
    # Let's stick to 10^12 or 10^14.
    # 10^14 -> sqrt is 10^7. 10^7 numbers is manageable in memory (80MB for list of ints).
    
    limit_gen = 10**14
    powerful_list = _generate_powerful(limit_gen)
    powerful_set = set(powerful_list)
    M = len(powerful_list)
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_score = 1
    
    # Strategy: Find candidate N by summing pairs from a subset of powerful numbers.
    # We will try several subsets of different sizes.
    
    # Subset sizes to try
    # Smaller subsets are faster to process (K^2 sums), but might miss the best N.
    # Larger subsets are slower but more likely to hit a high-density N.
    # K=1000 -> 500k sums. Very fast.
    # K=5000 -> 12.5M sums. Might take a few seconds.
    # K=10000 -> 50M sums. Might be too slow in Python.
    
    subset_sizes = [1000, 2000, 5000]
    
    # We can also try random subsets or subsets of larger powerful numbers
    # But smallest powerful numbers are most frequent in sums?
    # Actually, if we want N to be large, we might want larger a,b.
    # However, the density of powerful numbers decreases.
    # Let's stick to the smallest K powerful numbers first, as they are "densest".
    
    for K in subset_sizes:
        if time.time() > deadline - 20:
            break
            
        if K > M:
            K = M
        
        # Use the first K powerful numbers (smallest)
        # This biases towards smaller N.
        # To find larger N, we might need to sample from the whole range.
        # But for now, let's see what this yields.
        
        P_sub = powerful_list[:K]
        
        # Count sums
        sum_counts = defaultdict(int)
        # Optimization: only a <= b
        for i in range(K):
            a = P_sub[i]
            # b >= a
            for j in range(i, K):
                b = P_sub[j]
                s = a + b
                sum_counts[s] += 1
        
        # Find top candidates
        # Sort by count descending
        # We only need top few
        top_candidates = sorted(sum_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        for N_cand, count in top_candidates:
            if time.time() > deadline - 10:
                break
            
            # Verify N_cand with full powerful set
            # Note: N_cand might be larger than limit_gen?
            # If N_cand > limit_gen, we need to generate more powerful numbers.
            # But our P_sub is from limit_gen. So N_cand <= 2 * limit_gen.
            # If N_cand > limit_gen, we need to check powerful numbers up to N_cand.
            # We only generated up to limit_gen.
            # If N_cand > limit_gen, we must extend powerful_list.
            
            if N_cand > limit_gen:
                # Extend generation
                new_limit = N_cand
                # This is expensive if done repeatedly.
                # But K is small, so N_cand is sum of two small powerfuls, so N_cand is small.
                # Max N_cand from K=5000 (smallest 5000 powerfuls)
                # The 5000th powerful number is roughly ?
                # Count ~ N^(1/3) * N^(1/2) ? No.
                # Let's just check if it's within current limit.
                # If not, skip or regenerate.
                # Given K is small, N_cand is likely < limit_gen.
                pass
            
            pairs = _count_representations(N_cand, powerful_list, powerful_set)
            
            # Verify each pair
            valid_pairs = []
            for a, b in pairs:
                if a <= b and a + b == N_cand and a in powerful_set and b in powerful_set:
                    # Double check powerfulness with exact factorization for safety?
                    # The set was generated correctly, so membership is sufficient.
                    # But _is_powerful is the "exact factorization" check requested.
                    # For speed, we trust the set generation.
                    # However, the task says "checked by exact factorization".
                    # We should probably run _is_powerful on the pairs to be strict, 
                    # or ensure _generate_powerful is correct.
                    # _generate_powerful generates a^2 * b^3. This is the definition.
                    # So membership in set is equivalent to being powerful.
                    valid_pairs.append([a, b])
            
            if len(valid_pairs) > best_score:
                best_score = len(valid_pairs)
                best_N = N_cand
                best_pairs = valid_pairs
    
    # If we have time, try to search for larger N by sampling sums from larger indices
    if time.time() < deadline - 30:
        # Try a subset of larger powerful numbers
        # Take indices from M//2 to M
        if M > 1000:
            start_idx = max(0, M // 2)
            end_idx = M
            K2 = min(2000, end_idx - start_idx)
            P_sub2 = powerful_list[start_idx:start_idx + K2]
            
            sum_counts2 = defaultdict(int)
            for i in range(K2):
                a = P_sub2[i]
                for j in range(i, K2):
                    b = P_sub2[j]
                    s = a + b
                    sum_counts2[s] += 1
            
            top_candidates2 = sorted(sum_counts2.items(), key=lambda x: x[1], reverse=True)[:20]
            
            for N_cand, count in top_candidates2:
                if time.time() > deadline - 10:
                    break
                
                # Ensure we have powerful numbers up to N_cand
                if N_cand > limit_gen:
                    # Regenerate powerful numbers up to N_cand
                    # This is expensive, but only if N_cand is large.
                    # With K2=2000, N_cand is sum of two large powerfuls.
                    # Might exceed limit_gen.
                    # Let's just skip if N_cand > limit_gen to avoid expensive regeneration
                    # unless we are sure we can afford it.
                    continue
                
                pairs = _count_representations(N_cand, powerful_list, powerful_set)
                valid_pairs = []
                for a, b in pairs:
                    if a <= b and a + b == N_cand and a in powerful_set and b in powerful_set:
                        valid_pairs.append([a, b])
                
                if len(valid_pairs) > best_score:
                    best_score = len(valid_pairs)
                    best_N = N_cand
                    best_pairs = valid_pairs

    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
