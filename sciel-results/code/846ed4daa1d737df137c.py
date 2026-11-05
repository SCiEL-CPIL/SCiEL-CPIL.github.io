import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Generate all divisors of 5040
    n = 5040
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    
    # Sort in ascending order for combination generation
    divisors.sort()
    target = 4979
    best_count = float('inf')
    best_solution = None
    
    # Greedy attempt first for a quick valid solution
    divisors_desc = sorted(divisors, reverse=True)
    remainder = target
    chosen = []
    for d in divisors_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    if remainder == 0 and len(chosen) < best_count:
        best_count = len(chosen)
        best_solution = list(chosen)
    
    # Try smaller counts first: 1, 2, 3, 4
    # For each count, enumerate combinations to find if any sum to target
    for count in range(1, min(best_count, 5)):
        if time.time() - start_time > DEADLINE:
            break
        
        if count == 1:
            if target in divisors:
                best_count = 1
                best_solution = [target]
                break
        elif count == 2:
            div_set = set(divisors)
            found = False
            for d in divisors:
                if time.time() - start_time > DEADLINE:
                    break
                complement = target - d
                if complement in div_set and complement > d:
                    best_count = 2
                    best_solution = [d, complement]
                    found = True
                    break
            if found:
                break
        elif count == 3:
            div_set = set(divisors)
            found = False
            for i, d1 in enumerate(divisors):
                if time.time() - start_time > DEADLINE:
                    break
                rem2 = target - d1
                for d2 in divisors[i+1:]:
                    if d2 >= rem2:
                        break
                    complement = rem2 - d2
                    if complement in div_set and complement > d2:
                        best_count = 3
                        best_solution = [d1, d2, complement]
                        found = True
                        break
                if found:
                    break
            if found:
                break
        elif count == 4:
            div_set = set(divisors)
            found = False
            for i, d1 in enumerate(divisors):
                if time.time() - start_time > DEADLINE:
                    break
                rem3 = target - d1
                if rem3 <= 0:
                    break
                for j in range(i+1, len(divisors)):
                    d2 = divisors[j]
                    if d2 >= rem3:
                        break
                    rem2 = rem3 - d2
                    if rem2 <= 0:
                        break
                    for k in range(j+1, len(divisors)):
                        d3 = divisors[k]
                        if d3 >= rem2:
                            break
                        d4 = rem2 - d3
                        if d4 > d3 and d4 in div_set:
                            best_count = 4
                            best_solution = [d1, d2, d3, d4]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: return greedy solution
    remainder = target
    chosen = []
    for d in divisors_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
