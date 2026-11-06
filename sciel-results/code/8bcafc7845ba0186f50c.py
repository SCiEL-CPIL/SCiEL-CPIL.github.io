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
    
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def _is_powerful(n):
    """Check if n is powerful using exact factorization."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count == 1:
                return False
        p += 1
    if temp > 1:
        return False
    return True

def _count_representations(N, powerful_list, powerful_set):
    """Count how many pairs (a,b) with a<=b, a+b=N, both powerful."""
    count = 0
    for a in powerful_list:
        if 2 * a > N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    # Generate powerful numbers up to a reasonable limit
    # The key insight: we want to search for N values that have many representations
    # Strategy: use a hash map to count sums, but be smart about memory
    
    # Phase 1: Generate powerful numbers up to ~2*10^7
    # This gives us ~100k powerful numbers
    limit = 2 * 10**7
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    M = len(powerful)
    
    # Phase 2: Count all pairwise sums using a dictionary
    # But we need to be careful about memory and time
    # Use a more efficient approach: for each a, iterate b >= a, count sums
    
    sum_counts = defaultdict(int)
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_result
    
    # Iterate through pairs, but with early termination
    for i in range(M):
        a = powerful[i]
        if 2 * a > limit:
            break
        
        # For this a, b ranges from a to limit - a
        # Binary search for the largest index j where powerful[j] <= limit - a
        max_b = limit - a
        
        # Find j_max: largest j such that powerful[j] <= max_b and j >= i
        lo, hi = i, M - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if powerful[mid] <= max_b:
                lo = mid + 1
            else:
                hi = mid - 1
        j_max = hi
        
        if j_max < i:
            continue
        
        # Count sums for this a with b from i to j_max
        # To avoid O(M^2) in the worst case, we limit the inner loop
        # but this is already bounded by the structure
        
        for j in range(i, j_max + 1):
            s = a + powerful[j]
            sum_counts[s] += 1
        
        # Check time periodically
        if i % 500 == 0 and time.time() - start_time > deadline - 25:
            break
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        # Find best from what we have
        if sum_counts:
            best_N = max(sum_counts, key=sum_counts.get)
            best_count_here = sum_counts[best_N]
            if best_count_here > best_count:
                best_count = best_count_here
                pairs = []
                for a in powerful:
                    if 2 * a > best_N:
                        break
                    b = best_N - a
                    if b in powerful_set:
                        pairs.append([a, b])
                best_result = {'N': best_N, 'pairs': pairs}
        return best_result
    
    # Find the N with maximum representations
    if sum_counts:
        best_N = max(sum_counts, key=sum_counts.get)
        best_count_here = sum_counts[best_N]
        
        if best_count_here > best_count:
            best_count = best_count_here
            pairs = []
            for a in powerful:
                if 2 * a > best_N:
                    break
                b = best_N - a
                if b in powerful_set:
                    pairs.append([a, b])
            best_result = {'N': best_N, 'pairs': pairs}
    
    # Phase 3: Try to improve by searching around the best N found
    # Check nearby N values that might have similar or better counts
    if best_count > 1:
        base_N = best_result['N']
        # Check a window around base_N
        for delta in range(-50, 51):
            candidate_N = base_N + delta
            if candidate_N < 2 or candidate_N > limit:
                continue
            if candidate_N == base_N:
                continue
            
            count = _count_representations(candidate_N, powerful, powerful_set)
            if count > best_count:
                best_count = count
                pairs = []
                for a in powerful:
                    if 2 * a > candidate_N:
                        break
                    b = candidate_N - a
                    if b in powerful_set:
                        pairs.append([a, b])
                best_result = {'N': candidate_N, 'pairs': pairs}
            
            if time.time() - start_time > deadline - 5:
                break
    
    return best_result

# EVOLVE-BLOCK-END
