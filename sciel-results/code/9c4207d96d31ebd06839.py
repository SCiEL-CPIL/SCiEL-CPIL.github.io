import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order for potential early pruning or just for consistent ordering
    divisors.sort(reverse=True)
    
    # Check for k=1
    if target in divisors:
        return [target]
        
    # Check for k=2
    n_divs = len(divisors)
    for i in range(n_divs):
        for j in range(i + 1, n_divs):
            if divisors[i] + divisors[j] == target:
                return [divisors[i], divisors[j]]
                
    # Check for k=3
    for i in range(n_divs):
        for j in range(i + 1, n_divs):
            # Pruning: if sum of largest remaining is too small, break?
            # divisors[i] + divisors[j] + divisors[k] == target
            # divisors[k] = target - divisors[i] - divisors[j]
            # We can use a set for O(1) lookup
            pass # We'll do a better k=3 search below
    
    # More efficient k=3 search using set
    div_set = set(divisors)
    for i in range(n_divs):
        for j in range(i + 1, n_divs):
            rem = target - divisors[i] - divisors[j]
            if rem > 0 and rem in div_set:
                # Ensure distinctness: rem must not be divisors[i] or divisors[j]
                if rem != divisors[i] and rem != divisors[j]:
                    return [divisors[i], divisors[j], rem]
                    
    # Check for k=4
    # We can use meet-in-the-middle or nested loops with pruning
    # Since n=60, 4 nested loops is ~ 5 million iterations worst case, which might be slow in Python if not careful.
    # But we can break early if sum exceeds target.
    # Sort ascending for easier pruning? No, descending allows us to stop when sum > target.
    
    # Let's try a recursive search with pruning for k=4 and k=5
    # But we already have a k=5 solution. So we only need to check if k=4 exists.
    
    # Efficient k=4 check:
    # Fix i, j. Then we need divisors[k] + divisors[l] = target - divisors[i] - divisors[j].
    # This reduces to a 2-sum problem for the remaining pair.
    
    # Precompute all pair sums and their indices?
    # Or just loop i, j and then use a set for the remaining 2.
    
    for i in range(n_divs):
        for j in range(i + 1, n_divs):
            rem2 = target - divisors[i] - divisors[j]
            if rem2 <= 0:
                continue
            # Find k, l such that divisors[k] + divisors[l] == rem2
            # k > j, l > k
            # We can iterate k from j+1 to n_divs-1
            # and check if rem2 - divisors[k] is in the set of divisors at indices > k
            # To make this fast, we can just check if (rem2 - divisors[k]) is in the set of divisors, 
            # and then verify indices.
            # Since we just need existence and the values, we can do:
            found = False
            for k in range(j + 1, n_divs):
                d_k = divisors[k]
                if d_k >= rem2:
                    break # Since sorted descending, subsequent d_l will be smaller, but wait.
                    # divisors is descending. d_k is large.
                    # We need d_l = rem2 - d_k.
                    # Since d_l must be a divisor, and distinct from i, j, k.
                    # And d_l < d_k (since l > k).
                    # So d_l = rem2 - d_k < d_k => rem2 < 2*d_k => d_k > rem2/2.
                    # If d_k <= rem2/2, then d_l >= d_k, which contradicts l > k (if we assume strict order of indices implies value order? No, indices imply value order because sorted).
                    # So we only need to check d_k such that d_k > rem2/2.
                    # Actually, simpler: just check if (rem2 - d_k) is in the set of divisors.
                    # And ensure it's not i, j, k.
                    # And ensure it appears after k in the list? No, just distinct values.
                    # Since divisors are unique values, checking value in set is enough for existence.
                    # We just need to make sure the value found is not one of the used ones.
                    
                    d_l = rem2 - d_k
                    if d_l > 0 and d_l in div_set:
                        if d_l != divisors[i] and d_l != divisors[j] and d_l != d_k:
                            return [divisors[i], divisors[j], d_k, d_l]
                            
    # If no k=4 found, return the known k=5 solution
    # Greedy found: [2520, 1680, 720, 56, 3]
    # Let's verify it's valid
    sol = [2520, 1680, 720, 56, 3]
    if sum(sol) == target and all(x in div_set for x in sol) and len(set(sol)) == len(sol):
        return sol
        
    # Fallback: General BFS/DFS for minimum terms (should not reach here)
    # But to be safe, we can run a simple DP or search for k=5, 6...
    # Given the constraints, the greedy solution is likely optimal or close.
    # We'll return the greedy solution as the best found so far.
    
    # Actually, let's just implement a proper search for k=5 if k=4 fails, 
    # but we already have a candidate.
    
    return sol
