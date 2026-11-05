"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 512
    # We want to maximize the number of distinct positive integers in [1, 512]
    # such that no element is the exact arithmetic mean of any subset of at least two OTHER elements.
    #
    # Condition: For any a in S, there is no subset T of S \ {a} with |T| >= 2 such that sum(T) = |T| * a.
    # This is equivalent to saying that a is not in the set of all subset averages of S\{a} (for subsets of size >= 2).
    #
    # A known sufficient condition for such sets is to use a "sum-free" like structure or powers of 2,
    # but we can do better.
    #
    # Let's try a greedy approach with a more efficient check.
    # However, the current program scores 1.0, which likely means it finds a small set.
    # The prompt says "Maximize cardinality. At most 48 integers".
    # So the target is likely close to 48.
    #
    # Let's analyze the constraint more carefully.
    # If we pick numbers that are far apart, we are less likely to have averages.
    # Consider the set of powers of 2: {1, 2, 4, 8, ..., 2^k}.
    # 1, 2, 4, 8, 16, 32, 64, 128, 256. (9 elements)
    # Is 4 the average of any subset of others?
    # Others: 1, 2, 8, 16, ...
    # Subsets of size 2: 1+2=3, 1+8=9, ... none is 4*2=8. Wait, average is sum/2. So we need sum = 2*4 = 8.
    # Subset {8} is size 1, not allowed. Subset {1,2,?} no.
    # Actually, for powers of 2, the sum of any subset is unique (binary representation).
    # Can a power of 2 be the average of others?
    # a = (sum of others) / k  => sum of others = k*a.
    # Since others are powers of 2 different from a, their sum is a number with bits set at positions different from a's bit.
    # k*a is a multiple of a.
    # This structure is very safe. But 9 is small.
    #
    # We need up to 48 elements.
    # Let's try a random greedy approach with a faster validity check.
    # Instead of checking all subsets, we can use the property that if the set is "convex" or has large gaps, it's easier.
    #
    # Alternative approach:
    # The condition is: a != mean(T) for any T subset of S\{a}, |T|>=2.
    # This is equivalent to: a is not in the convex hull of any subset of S\{a}? No, arithmetic mean is specific.
    #
    # Let's try to construct a set with large gaps.
    # If we pick numbers such that the difference between consecutive elements is large,
    # it's harder to form averages.
    #
    # Let's try a greedy algorithm that picks the smallest available number, but checks validity efficiently.
    # How to check validity efficiently?
    # For a candidate set S, and a new element x, we need to check:
    # 1. x is not the average of any subset of S (size >= 2).
    # 2. No existing element a in S is the average of any subset of (S \ {a}) U {x} (size >= 2).
    #
    # Check 1: Is x in the set of subset averages of S?
    # This can be checked by DP on subset sums, but S can be up to 48 elements. 2^48 is too big.
    # However, we only care about averages.
    #
    # Let's look at the structure of "no element is the mean of others".
    # This is related to "1-sum-free" or similar additive combinatorics concepts.
    #
    # Let's try a different strategy:
    # Pick numbers that are very large, e.g., near 512.
    # Or pick numbers that are powers of a base, but scaled.
    #
    # Let's try a random search with local improvement, or a smarter greedy.
    # Given the time limit (60s) and the need for a strong algorithm, let's try a greedy with a heuristic:
    # Prefer numbers that are "far" from existing elements in terms of potential averages.
    #
    # Actually, a known result is that the maximum size of such a set in [1, N] is roughly N / log N or similar?
    # For N=512, 48 is about 512/10.
    #
    # Let's try a greedy that picks the largest possible numbers first?
    # Or the smallest?
    #
    # Let's try a simple greedy: iterate from 1 to 512, add if valid.
    # But we need a fast validity check.
    #
    # Fast validity check for adding x to S:
    # We need to ensure no a in S U {x} is the average of a subset of (S U {x}) \ {a} of size >= 2.
    #
    # Let's use a bitset DP for subset sums of S.
    # Max sum of 48 elements <= 48 * 512 = 24576.
    # Bitset of size 24576 is small.
    # We can maintain a bitset `sums` where sums[i] is True if i is a subset sum of S.
    # Also maintain a count of how many subsets yield each sum? No, we just need existence.
    # But we need to check if x = sum(T)/k for some T subset of S, |T|>=2.
    # This means sum(T) = k*x for some k >= 2.
    # So we need to check if any multiple of x (k*x) is a subset sum of S, and the subset has size >= 2.
    #
    # We can maintain a 2D DP: dp[k][s] = True if there is a subset of size k with sum s.
    # k up to 48, s up to 24576. 48 * 24576 ~ 1.2M booleans. This is fine.
    #
    # When adding x:
    # 1. Check if x is an average of a subset of S.
    #    For k in 2..len(S): if dp[k][k*x] is True, then invalid.
    # 2. Check if any existing a in S is an average of a subset of (S \ {a}) U {x}.
    #    This is harder because the subset can include x.
    #    Subsets of (S \ {a}) U {x} are either subsets of S\{a} or subsets of S\{a} plus {x}.
    #    Case 2a: Subset T is from S\{a}. We already know S was valid, so a is not the average of any subset of S\{a}.
    #    Case 2b: Subset T = T' U {x}, where T' is a subset of S\{a}.
    #    Then mean(T) = (sum(T') + x) / (|T'| + 1) = a.
    #    => sum(T') + x = a * (|T'| + 1)
    #    => sum(T') = a * |T'| + a - x
    #    We need to check if there exists k' >= 0 (size of T') such that there is a subset of S\{a} of size k' with sum = a*k' + a - x.
    #    Note: |T| = k' + 1 >= 2 => k' >= 1.
    #    So we need to check if for any a in S, and any k' in 1..len(S)-1,
    #    there is a subset of S\{a} of size k' with sum = a*k' + a - x.
    #
    # This requires checking for each a in S, which is O(|S|) per candidate x.
    # And for each a, we need to check if a specific sum is achievable by a subset of size k' from S\{a}.
    # We can precompute DP for S, but removing 'a' from DP is not trivial.
    #
    # Alternative:
    # Instead of incremental addition, we can use a local search or simulated annealing.
    # Or, we can use a greedy that is slower but correct, and hope it finds a good set.
    #
    # Let's try a different angle.
    # If we pick all numbers in [1, 512] that are congruent to 1 mod 3?
    # Or use a random set and repair.
    #
    # Let's try a greedy with the 2D DP approach, but optimize the check for existing elements.
    # We can maintain the DP table for the current set S.
    # When adding x, we update the DP table.
    # But checking validity for existing elements requires "removing" a from the DP, which is hard.
    #
    # Let's simplify:
    # We'll use a greedy that adds elements one by one.
    # For each candidate x (from 1 to 512), we check if S U {x} is valid.
    # To check validity, we can iterate over all a in S U {x} and check if a is the average of a subset of others.
    # For a given a, we need to know if there is a subset T of (S U {x}) \ {a} with sum(T) = k*a for some k >= 2.
    #
    # We can use a bitset for subset sums of (S U {x}) \ {a}.
    # Since |S| <= 48, we can compute the subset sum bitset for each a on the fly?
    # That would be O(|S| * 2^{|S|}) which is too slow.
    #
    # Let's try a heuristic:
    # Pick numbers that are far apart.
    # Start with 1, then 512, then pick numbers in between that are "safe".
    #
    # Let's try a random greedy:
    # 1. Start with an empty set.
    # 2. Shuffle the numbers 1..512.
    # 3. For each number x in the shuffled list, try to add it.
    # 4. Check validity using a slow but correct method (since we only add if valid, and we stop when we have 48 or no more can be added).
    #
    # To make it faster, we can try multiple random orders and keep the best.
    #
    # Let's implement the slow validity check first, and see how many elements we can get.
    # If it's too slow, we optimize.
    #
    # Validity check for a set S:
    # For each a in S:
    #   others = S \ {a}
    #   For each subset T of others with |T| >= 2:
    #     if sum(T) == len(T) * a: return False
    # Return True
    #
    # This is exponential in |S|. For |S|=48, it's impossible.
    #
    # We need a better check.
    # Let's use the DP approach for the whole set S.
    # dp[k][s] = True if there is a subset of S of size k with sum s.
    # Then, a is the average of a subset T of S\{a} of size k if and only if:
    #   There is a subset of S of size k+1 with sum (k+1)*a, and a is in that subset.
    #   OR there is a subset of S of size k with sum k*a, and a is NOT in that subset (and k>=2).
    #
    # This is getting complicated.
    #
    # Let's try a different construction:
    # Use a set of numbers that are "superincreasing" or similar.
    # For example, a_i = 2^i. We saw this gives 9 elements.
    # What if we use a_i = c * 2^i? Still 9 elements.
    #
    # What if we use a_i = i * M for large M?
    # If M is large enough, the averages won't hit exactly.
    # For example, if we take {1*M, 2*M, ..., k*M}, the average of a subset is (sum of indices * M) / k.
    # This is an integer only if sum of indices is divisible by k.
    # And it equals j*M only if sum of indices / k = j.
    # So we need a set of indices {1..k} such that no index is the average of a subset of others.
    # This is the same problem but with smaller numbers!
    #
    # So the problem reduces to finding a set of indices I such that no i in I is the average of a subset of I\{i}.
    # Then we can scale by M to fit in [1, 512].
    #
    # Let's solve the smaller problem first.
    # Find a large set I of positive integers such that no element is the average of a subset of others.
    # Then scale it to fit in [1, 512].
    #
    # Let's try to find a large such set I.
    # We can use a greedy algorithm on small numbers.
    #
    # Let's try to find a set I of size 48? That would require the max element to be at least 48.
    # If we scale by M, the max element is 48*M <= 512 => M <= 10.
    # So we need a set I of size 48 with max element <= 10? Impossible.
    #
    # So we can't just scale a small set.
    #
    # Let's go back to the original problem.
    # We need 48 numbers in [1, 512].
    #
    # Let's try a greedy that picks numbers with large gaps.
    # Start with 1, 512.
    # Then pick 256, 128, 64, ...
    #
    # Let's try a random search with a fast validity check using the DP table for the current set.
    # We'll maintain the DP table for the current set S.
    # When adding x, we check if x is an average of a subset of S.
    # And we check if any existing a is an average of a subset of S\{a} U {x}.
    #
    # For the second part, we can approximate by checking if a is an average of a subset of S U {x} that includes x.
    # This is equivalent to checking if a - x is an average of a subset of S\{a} with a specific size adjustment.
    #
    # This is too complex for a simple greedy.
    #
    # Let's try a different approach:
    # Use a local search.
    # Start with a random set of 48 numbers.
    # Try to swap elements to reduce the number of violations.
    #
    # Or, start with a small valid set and grow it.
    #
    # Given the time constraints, let's try a greedy that picks the largest possible numbers first.
    # This might create large gaps and avoid averages.
    #
    # Let's implement a greedy that iterates from 512 down to 1, and adds x if S U {x} is valid.
    # We'll use a slow validity check, but since we're adding from the top, the set might remain small or grow slowly.
    #
    # If this is too slow, we'll optimize.
    #
    # Let's try it.
    
    def is_valid(S):
        S = list(S)
        n = len(S)
        if n < 2:
            return True
        # DP for subset sums and sizes
        # dp[k][s] = True if there is a subset of size k with sum s
        max_sum = sum(S)
        dp = [[False] * (max_sum + 1) for _ in range(n + 1)]
        dp[0][0] = True
        for x in S:
            for k in range(n, 0, -1):
                for s in range(max_sum, -1, -1):
                    if dp[k-1][s-x]:
                        dp[k][s] = True
        
        for a in S:
            others = [x for x in S if x != a]
            # Check if a is the average of a subset of others
            # a = sum(T) / k => sum(T) = k*a
            # T is a subset of others, |T| = k >= 2
            # We need to check if there is a subset of S of size k+1 with sum (k+1)*a that includes a,
            # or a subset of S of size k with sum k*a that does not include a.
            #
            # Simpler: iterate over possible k and check if k*a is a subset sum of others.
            # We can use the DP table, but we need to exclude a.
            #
            # Let's just check all subsets of others for small |others|.
            # But |others| can be up to 47.
            #
            # Let's use the DP table for S, and for each a, we can't easily exclude it.
            #
            # Let's try a different check:
            # For each a in S, for each k in 2..n-1:
            #   if dp[k][k*a] is True, we need to check if a is in such a subset.
            #   If a is in a subset of size k with sum k*a, then the rest of the subset has sum k*a - a = (k-1)*a and size k-1.
            #   This means a is the average of the rest? No.
            #
            # This is getting too complicated. Let's use a simple check for small sets.
            # If |S| <= 20, we can use brute force.
            # If |S| > 20, we assume it's valid? No, that's risky.
            #
            # Let's try a heuristic:
            # If the set is "well-spread", it's likely valid.
            #
            # Let's try a different strategy:
            # Use a set of numbers that are all odd.
            # The average of an even number of odd numbers is an integer.
            # The average of an odd number of odd numbers is a half-integer.
            # So if we only pick odd numbers, the average of a subset of size k is an integer only if k is even.
            # So we only need to check subsets of even size.
            # This reduces the search space by half.
            #
            # Let's try a greedy that picks odd numbers only.
            #
            # Let's try a random greedy with odd numbers.
            
            return True  # Placeholder for now
        
    # Let's try a simple greedy with odd numbers, and a fast check.
    selected = []
    for x in range(512, 0, -1):
        if x % 2 == 0:
            continue
        candidate = selected + [x]
        # Simple check: for each a in candidate, check if a is the average of a subset of others.
        # We'll use a brute force check for small candidate sizes.
        valid = True
        for a in candidate:
            others = [y for y in candidate if y != a]
            # Check subsets of others of size 2 and 3 (common cases)
            # If |others| is large, we skip the check? No, we need to be correct.
            # Let's use a DP check for each a.
            # Since candidate size is small (we stop at 48), we can use a DP for each a.
            # But this is O(n^2 * sum) which might be slow.
            #
            # Let's try a different check:
            # For a given a, we need to check if there is a subset T of others with sum(T) = k*a for some k >= 2.
            # We can use a bitset for subset sums of others.
            others_set = set(others)
            max_sum = sum(others)
            bitset = 1  # bitset[0] = 1
            for y in others:
                bitset |= (bitset << y)
            # Now check if any k*a is in the bitset, for k >= 2.
            for k in range(2, len(others) + 1):
                target = k * a
                if target > max_sum:
                    break
                if (bitset >> target) & 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
            if len(selected) >= 48:
                break
    
    return selected

# EVOLVE-BLOCK-END
