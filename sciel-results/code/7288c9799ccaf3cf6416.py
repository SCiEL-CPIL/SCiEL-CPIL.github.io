"""Rewritten program: minimize number of distinct divisors of 5040 summing to 4979."""
import time
import sys

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    divisors.sort(reverse=True)  # largest first for greedy initial solution

    # Greedy initial solution
    def greedy():
        rem = target
        chosen = []
        for d in divisors:
            if d <= rem:
                chosen.append(d)
                rem -= d
                if rem == 0:
                    break
        if rem == 0:
            return chosen
        return None

    best = greedy()
    if best is None:
        raise RuntimeError("Greedy failed")

    best_count = len(best)

    # Precompute divisor set for O(1) membership
    div_set = set(divisors)

    # Try to find solutions with fewer terms
    # Use a BFS/DFS approach with pruning
    # Since target is ~5000 and divisors are up to 5040, the number of terms will be small
    # Let's try k from 1 upward, but we want to minimize, so start from 1 and go up
    # Actually, we have a greedy solution, so we can try to improve it
    
    # Better approach: use dynamic programming or meet-in-the-middle
    # Given the small target and divisor set, let's try a recursive search with pruning
    
    # Sort divisors in descending order for better pruning
    divs_desc = sorted(divisors, reverse=True)
    
    # Maximum possible sum with k largest divisors
    # For a given k, the max sum is sum of k largest divisors
    # The min sum is sum of k smallest divisors
    
    # Let's use a recursive backtracking with memoization on (index, remaining)
    # But the state space might be large. Let's try a different approach.
    
    # Since we want to minimize the number of terms, let's try k = 1, 2, 3, ...
    # For each k, check if there exists a subset of size k that sums to target
    
    # For small k, we can use combinations or recursive search
    # k=1: just check if target is a divisor
    if target in div_set:
        return [target]
    
    # k=2: check pairs
    # For each divisor d, check if target - d is also a divisor and d != target - d
    for i, d1 in enumerate(divs_desc):
        if d1 >= target:
            continue
        d2 = target - d1
        if d2 in div_set and d2 != d1:
            return [d1, d2]
    
    # k=3: check triples
    # This is O(n^2) where n is number of divisors
    # Number of divisors of 5040: 5040 = 2^4 * 3^2 * 5 * 7, so (4+1)(2+1)(1+1)(1+1) = 5*3*2*2 = 60 divisors
    # So 60^2 = 3600 checks for k=3, which is fine
    
    divs_set = set(divs_desc)
    for i, d1 in enumerate(divs_desc):
        if d1 >= target:
            continue
        for j in range(i + 1, len(divs_desc)):
            d2 = divs_desc[j]
            if d1 + d2 >= target:
                continue
            d3 = target - d1 - d2
            if d3 in divs_set and d3 > d2:  # ensure distinct and avoid duplicates
                return [d1, d2, d3]
    
    # k=4: check quadruples
    # 60^3 = 216000, still manageable
    for i, d1 in enumerate(divs_desc):
        if d1 >= target:
            continue
        for j in range(i + 1, len(divs_desc)):
            d2 = divs_desc[j]
            if d1 + d2 >= target:
                continue
            for k in range(j + 1, len(divs_desc)):
                d3 = divs_desc[k]
                if d1 + d2 + d3 >= target:
                    continue
                d4 = target - d1 - d2 - d3
                if d4 in divs_set and d4 > d3:
                    return [d1, d2, d3, d4]
    
    # k=5: 60^4 = 12,960,000, might be slow in Python but let's try with pruning
    # Add time check
    if time.time() > deadline - 10:
        return best
    
    for i, d1 in enumerate(divs_desc):
        if d1 >= target:
            continue
        for j in range(i + 1, len(divs_desc)):
            d2 = divs_desc[j]
            if d1 + d2 >= target:
                continue
            for k in range(j + 1, len(divs_desc)):
                d3 = divs_desc[k]
                if d1 + d2 + d3 >= target:
                    continue
                for l in range(k + 1, len(divs_desc)):
                    d4 = divs_desc[l]
                    if d1 + d2 + d3 + d4 >= target:
                        continue
                    d5 = target - d1 - d2 - d3 - d4
                    if d5 in divs_set and d5 > d4:
                        return [d1, d2, d3, d4, d5]
                    if time.time() > deadline - 5:
                        return best
    
    # If we didn't find a solution with <= 5 terms, return the greedy solution
    return best

if __name__ == "__main__":
    result = construct()
    print(result)
    print(f"Count: {len(result)}, Sum: {sum(result)}")
