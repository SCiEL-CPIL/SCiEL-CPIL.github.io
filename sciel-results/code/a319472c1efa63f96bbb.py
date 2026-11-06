from fractions import Fraction
import time

def construct():
    target = Fraction(5, 31)
    best_solution = [8, 31, 248]
    best_max = 248
    
    deadline = time.time() + 50.0

    # We know d must be >= 7
    # Try to find a solution with max denominator < best_max
    
    # Meet in the Middle search
    # We iterate M from a lower bound up to best_max - 1
    # Lower bound: sum_{d=7}^M 1/d >= 5/31
    # 1/7 + 1/8 = 15/56 ~ 0.26 > 0.16
    # So M can be as low as 8?
    # If M=8, D={7,8}. Sums: 0, 1/7, 1/8, 15/56.
    # None is 5/31.
    
    # We will try M from 50 to 80 first.
    # If M is too large, 2^(M/2) is too big.
    
    for M in range(50, min(81, best_max)):
        if time.time() > deadline:
            break
            
        D = list(range(7, M + 1))
        mid = len(D) // 2
        D1 = D[:mid]
        D2 = D[mid:]
        
        # Generate subset sums for D1
        sums1 = {Fraction(0): ()}
        for d in D1:
            new_sums = {}
            for s, subset in sums1.items():
                # We can prune sums that are already > target
                if s > target:
                    continue
                # Add d
                new_s = s + Fraction(1, d)
                if new_s <= target:
                    # Store the new sum
                    # If multiple subsets yield the same sum, we keep the one with smaller max?
                    # Or just any one? We want to find *a* solution.
                    # Storing the subset allows reconstruction.
                    # To save memory, we might just store the sum and a bitset or tuple.
                    # Tuple is fine for small M.
                    new_sums[new_s] = subset + (d,)
            sums1.update(new_sums)
            
        # Check D2
        found = False
        # Generate subset sums for D2 on the fly?
        # Or generate all and check?
        # Generating all is 2^(len(D2)).
        # For M=50, len(D2)=22. 4M items.
        # We can iterate recursively or iteratively.
        
        # Iterative generation for D2
        sums2 = {Fraction(0): ()}
        for d in D2:
            new_sums = {}
            for s, subset in sums2.items():
                if s > target:
                    continue
                new_s = s + Fraction(1, d)
                if new_s <= target:
                    new_sums[new_s] = subset + (d,)
            sums2.update(new_sums)
            
        # Now check for match
        # For each sum in sums2, check if target - sum in sums1
        for s2, sub2 in sums2.items():
            rem = target - s2
            if rem in sums1:
                sub1 = sums1[rem]
                solution = list(sub1) + list(sub2)
                # Verify distinctness (guaranteed by construction from disjoint sets)
                # Verify sum
                # Verify max < best_max
                if max(solution) < best_max:
                    best_max = max(solution)
                    best_solution = solution
                    found = True
                    break
        
        if found:
            # We found a solution with max < previous best.
            # Can we find an even better one?
            # The loop continues to higher M, but we want LOWER max.
            # So we should iterate M from LOW to HIGH?
            # No, we want the SMALLEST max denominator.
            # So we should iterate M from 50 upwards?
            # If we find a solution for M=50, its max is <= 50.
            # If we find a solution for M=60, its max is <= 60.
            # So we should stop at the FIRST M where we find a solution?
            # Yes, because we are iterating M in increasing order.
            # The first solution found will have the smallest possible max denominator among the searched M's.
            # Wait, is it possible that a solution exists for M=50 but we missed it?
            # No, MITM is exhaustive for the set {7..M}.
            # So if we find a solution for M=50, the max denominator is <= 50.
            # We can return it immediately.
            return best_solution

    return best_solution

if __name__ == "__main__":
    print(construct())
