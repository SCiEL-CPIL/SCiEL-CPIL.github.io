"""Rewritten program: exact minimum-cardinality subset-sum over divisors of 5040."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    target = 4979
    deadline = time.time() + 55.0  # safety margin before 60s kill

    # All distinct positive divisors of 5040
    n = 5040
    divisors = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            divisors.append(d)
            if d != n // d:
                divisors.append(n // d)
    divisors.sort(reverse=True)  # largest first

    # Precompute: we want the minimum number of distinct divisors summing to target.
    # The divisors are all positive integers dividing 5040.
    # Max divisor is 5040, target is 4979 < 5040, so we can't use 5040.
    # Filter out divisors > target (they can't be part of a sum that equals target with positive terms)
    # Actually, any single divisor > target can't be used. But sums of smaller ones can.
    # Let's keep only divisors <= target.
    usable = [d for d in divisors if d <= target]
    usable.sort(reverse=True)

    # We need to find the minimum k such that some k distinct elements from 'usable' sum to target.
    # Strategy: iterate k from 1 upward, and for each k, check if any k-subset sums to target.
    # For small k, use combinations. For larger k, use meet-in-the-middle.

    # Upper bound on k: worst case is using the smallest divisors.
    # But let's just iterate k from 1 to len(usable) with a reasonable cap.
    # Since target is 4979 and the largest usable divisor is 2520 (next is 1680, etc.),
    # let's think: 2520 + 1680 = 4200, need 779 more. 779 is not a divisor.
    # Let's just do a systematic search.

    # Meet-in-the-middle for k up to ~10 should be fine with 60 divisors.
    # Number of divisors of 5040: 5040 = 2^4 * 3^2 * 5 * 7, so (4+1)*(2+1)*2*2 = 60 divisors.
    
    best_solution = None

    # Try k = 1, 2, 3, ...
    max_k = min(len(usable), 20)  # cap to be safe

    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break

        if k == 1:
            if target in usable:
                return [target]
            continue

        if k == 2:
            # Check pairs
            s = set(usable)
            found = False
            for d in usable:
                complement = target - d
                if complement in s and complement != d:
                    return [d, complement]
            if time.time() > deadline:
                break
            continue

        if k <= 4:
            # Direct combinations for small k
            for combo in combinations(usable, k):
                if sum(combo) == target:
                    return list(combo)
            if time.time() > deadline:
                break
            continue

        # Meet-in-the-middle for k >= 5
        half1 = k // 2
        half2 = k - half1

        # To keep it tractable, limit the pool of divisors we consider.
        # For larger k, we might need smaller divisors. Let's use all usable divisors.
        # But combinations of 5 from 60 is 5.4M, which is okay. 6 from 60 is 50M, too many.
        # So for k=5, direct combos might work. For k>=6, we need smarter approach.
        
        if k <= 5:
            for combo in combinations(usable, k):
                if sum(combo) == target:
                    return list(combo)
            if time.time() > deadline:
                break
            continue

        # For k >= 6, use meet-in-the-middle with a restricted pool
        # or use a different strategy.
        # Let's try a greedy + local search approach as fallback, but first try MITM on a subset.
        
        # Actually, let's try a different approach for larger k:
        # Since we want minimum k, and we've exhausted k up to 5, let's try k=6 with MITM.
        # But 6 from 60 is too many. Let's restrict to the largest divisors first.
        
        # Alternative: use a recursive backtracking with pruning for k up to 8.
        # Or, since the number of divisors is only 60, we can use a bitset approach for subset sum
        # but we need exactly k elements.
        
        # Let's try a more efficient approach:
        # For k >= 6, use meet-in-the-middle but with a limited set of divisors.
        # Take the top 30 divisors (largest) for the first half and top 30 for the second half.
        # This might miss solutions, but it's a heuristic.
        
        # Better: use a complete search with pruning.
        # Since k is small (6-8), and we have 60 divisors, we can use backtracking.
        
        def backtrack(start, remaining, count, current):
            if count == 0:
                return current if remaining == 0 else None
            if remaining <= 0:
                return None
            if time.time() > deadline:
                return None
            # Pruning: if even taking the largest 'count' divisors from start can't reach 'remaining', stop
            # The largest possible sum with 'count' more elements from index 'start'
            # is the sum of the largest 'count' elements from usable[start:]
            if start + count > len(usable):
                return None
            max_possible = sum(usable[start:start+count])
            if max_possible < remaining:
                return None
            min_possible = sum(usable[start:start+count])  # actually min is smallest, but let's skip for now
            
            for i in range(start, len(usable) - count + 1):
                d = usable[i]
                if d > remaining:
                    continue
                result = backtrack(i + 1, remaining - d, count - 1, current + [d])
                if result is not None:
                    return result
            return None

        solution = backtrack(0, target, k, [])
        if solution is not None:
            return solution

    # If we didn't find a solution with the above, try a more exhaustive search
    # or return the best we can.
    
    # As a last resort, try all k up to 12 with backtracking
    for k in range(max_k + 1, 13):
        if time.time() > deadline:
            break
        
        def backtrack2(start, remaining, count, current):
            if count == 0:
                return current if remaining == 0 else None
            if remaining <= 0:
                return None
            if time.time() > deadline:
                return None
            if start + count > len(usable):
                return None
            max_possible = sum(usable[start:start+count])
            if max_possible < remaining:
                return None
            
            for i in range(start, len(usable) - count + 1):
                d = usable[i]
                if d > remaining:
                    continue
                result = backtrack2(i + 1, remaining - d, count - 1, current + [d])
                if result is not None:
                    return result
            return None

        solution = backtrack2(0, target, k, [])
        if solution is not None:
            return solution

    # Fallback: return an empty list or best effort
    return []
# EVOLVE-BLOCK-END
