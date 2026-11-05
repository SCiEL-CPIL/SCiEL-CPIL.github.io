# EVOLVE-BLOCK-START
def construct():
    n = 512
    max_card = 48

    # Start with a Sidon (B2) sequence to guarantee no 3-term arithmetic progression.
    # x_i = i * B + i^2 for i = 1..m, with B large enough to ensure distinct pairwise sums.
    # For i, j in [1, m], pairwise sums x_i + x_j = (i+j)*B + i^2 + j^2.
    # To ensure uniqueness, we need B > 2*m^2 (roughly).
    # Let's use B = 2 * m * (m + 1) + 1 to be safe.
    # We want to fit as many as possible in [1, 512].
    # x_m = m*B + m^2 <= 512.
    # With B ~ 2m^2, x_m ~ 2m^3 + m^2.
    # 2m^3 <= 512 => m^3 <= 256 => m <= 6.
    # Let's try m=6, B = 2*6*7+1 = 85.
    # x_1 = 85+1=86, x_2=170+4=174, x_3=255+9=264, x_4=340+16=356, x_5=425+25=450, x_6=510+36=546 > 512.
    # So m=5: x_5 = 5*85+25 = 450. x_6 would be too big.
    # Let's try smaller B. B = 2m+1? No, need B > 2m^2 for Sidon property to hold easily.
    # Actually, for x_i = i*B + i^2, the condition for Sidon is that all (i+j)*B + i^2+j^2 are distinct.
    # If (i+j)*B + i^2+j^2 = (k+l)*B + k^2+l^2, then (i+j-k-l)*B = k^2+l^2-i^2-j^2.
    # If i+j != k+l, then B divides the RHS. If B is large enough, this can't happen.
    # If i+j = k+l, then i^2+j^2 = k^2+l^2. For distinct pairs with same sum, this implies {i,j}={k,l}.
    # So we just need B > max |k^2+l^2-i^2-j^2| / |i+j-k-l| for i+j != k+l.
    # The max difference in i^2+j^2 is about 2m^2. So B > 2m^2 should work.
    
    # Let's use a simpler construction: powers of 2 are Sidon, but too sparse.
    # Let's just use a greedy approach with bitset checks for all subset sizes, 
    # but start with a good seed to make it faster.
    
    # Actually, let's just do the greedy with bitset checks from scratch.
    # The key insight from the previous attempt is that we can use bitsets for subset sum.
    # But we need to be careful with the size constraint (>=2).
    
    # Let's implement a function to check if a target is a subset sum of a list of numbers,
    # with the subset size at least min_size.
    # We can use a dictionary mapping sum -> min_size to achieve that sum.
    # Or, since the numbers are small (differences in [-511, 511]) and the list is short (<=48),
    # we can use a bitset where each bit represents a possible sum, and we track the size separately?
    # Actually, for each sum, we only care if it's achievable with size >= min_size.
    # We can maintain a set of (sum, size) pairs, but that might blow up.
    # Better: for each sum, keep the minimum size to achieve it.
    # Then at the end, check if target is in the dict and min_size[target] >= min_size_required.
    
    def can_achieve(diffs, target, min_size):
        # diffs: list of integers
        # target: integer
        # min_size: minimum number of elements to use
        # Returns True if there's a subset of diffs with sum == target and size >= min_size.
        # Note: empty subset has sum 0 and size 0.
        
        # dp[sum] = min size to achieve that sum
        # Initialize: dp[0] = 0
        dp = {0: 0}
        
        for d in diffs:
            new_dp = dict(dp)
            for s, sz in dp.items():
                new_s = s + d
                new_sz = sz + 1
                if new_s not in new_dp or new_sz < new_dp[new_s]:
                    new_dp[new_s] = new_sz
            dp = new_dp
        
        return target in dp and dp[target] >= min_size

    S = []
    
    for x in range(1, n + 1):
        valid = True
        
        # Check 1: Is x the mean of any subset of S of size >= 2?
        # Equivalent: 0 is a subset sum of {s - x for s in S} with size >= 2.
        # Since s != x, no single element is 0. So size >= 1 implies size >= 2?
        # No, size 1 means s-x=0 => s=x, which is false since x not in S.
        # So any non-empty subset summing to 0 has size >= 2.
        # So we just check if there's a non-empty subset summing to 0.
        # This is equivalent to can_achieve(diffs, 0, 1).
        if S:
            diffs = [s - x for s in S]
            if can_achieve(diffs, 0, 1):
                valid = False
        
        if valid:
            # Check 2: For each a in S, is a the mean of a subset of S U {x} excluding a?
            # Subsets containing x: T = T0 U {x}, T0 subset S\{a}, |T0| >= 1.
            # sum(T0) + x = (|T0|+1)*a => sum(T0) - |T0|*a = a - x.
            # Let y_i = s_i - a. sum(y_i) = a - x.
            # So check if (a - x) is a subset sum of {s - a for s in S, s != a} with size >= 1.
            for a in S:
                diffs_a = [s - a for s in S if s != a]
                target = a - x
                if can_achieve(diffs_a, target, 1):
                    valid = False
                    break
        
        if valid:
            S.append(x)
            if len(S) >= max_card:
                break
    
    return S
# EVOLVE-BLOCK-END
