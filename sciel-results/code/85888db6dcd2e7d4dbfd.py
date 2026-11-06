import time
import sys

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    
    # Greedy baseline (Size 5)
    # This is fast and always works for this specific instance
    rem = target
    greedy_solution = []
    for d in divisors:
        if d <= rem:
            greedy_solution.append(d)
            rem -= d
            if rem == 0:
                break
    # Verify greedy solution
    if sum(greedy_solution) == target and len(greedy_solution) == len(set(greedy_solution)):
        best_solution = greedy_solution
    else:
        best_solution = None # Should not happen, but fallback

    # Try to find smaller solution (Size 4)
    # Since size 3 is impossible (max sum of 3 distinct divisors < 4979? 
    # Max 3: 2520+1680+1260 = 5460. 
    # We checked specific cases above. 
    # Let's just search size 4.
    
    found_size_4 = None
    
    # Optimization: d1 must be large enough
    # d1 + d2 + d3 + d4 = 4979
    # d1 > d2 > d3 > d4
    # d1 >= 4979/4 = 1244.75
    
    # Filter divisors for d1
    d1_candidates = [d for d in divisors if d >= 1245]
    
    for d1 in d1_candidates:
        if time.time() - start_time > timeout:
            break
        rem1 = target - d1
        if rem1 <= 0:
            continue
        # d2 < d1, d2 + d3 + d4 = rem1
        # d2 >= rem1 / 3
        d2_candidates = [d for d in divisors if d < d1 and d >= rem1 / 3]
        
        for d2 in d2_candidates:
            if time.time() - start_time > timeout:
                break
            rem2 = rem1 - d2
            if rem2 <= 0:
                continue
            # d3 < d2, d3 + d4 = rem2
            # d3 >= rem2 / 2
            d3_candidates = [d for d in divisors if d < d2 and d >= rem2 / 2]
            
            for d3 in d3_candidates:
                if time.time() - start_time > timeout:
                    break
                d4 = rem2 - d3
                if d4 <= 0:
                    continue
                if d4 >= d3:
                    continue
                # Check if d4 is a divisor
                if n % d4 == 0:
                    # Found a valid solution of size 4
                    found_size_4 = [d1, d2, d3, d4]
                    break
            if found_size_4:
                break
        if found_size_4:
            break
    
    if found_size_4:
        return found_size_4
    elif best_solution:
        return best_solution
    else:
        # Fallback: return any valid solution or empty list
        # But we know greedy works
        return greedy_solution if greedy_solution else []
