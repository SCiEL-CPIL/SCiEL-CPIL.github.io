# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Generate all 2^a * 3^b * 5^c numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # For each target, find minimum-term antichain representation
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: use a simple valid representation
            answers.append([n] if n in smooth else [1, n-1] if n-1 in smooth and n-1 != 1 and (n-1) % 1 != 0 else [n])
            continue
        
        # Try 1 term
        if n in smooth:
            answers.append([n])
            continue
        
        # Try 2 terms: find a, b in smooth, a < b, a + b = n, a does not divide b
        found = False
        for a in smooth:
            if a >= n:
                break
            b = n - a
            if b <= a or b not in smooth:
                continue
            # Check antichain: a should not divide b
            if b % a != 0:
                answers.append([a, b])
                found = True
                break
        if found:
            continue
        
        # Try 3 terms: a < b < c, a+b+c=n, no term divides another
        found = False
        smooth_le_n = [x for x in smooth if x < n]
        for i in range(len(smooth_le_n)):
            a = smooth_le_n[i]
            if a >= n:
                break
            rem1 = n - a
            for j in range(i + 1, len(smooth_le_n)):
                b = smooth_le_n[j]
                if b >= rem1:
                    break
                c = rem1 - b
                if c <= b or c not in smooth:
                    continue
                # Check antichain: no pair divides another
                if b % a == 0 or c % a == 0 or c % b == 0:
                    continue
                answers.append([a, b, c])
                found = True
                break
            if found:
                break
        if found:
            continue
        
        # Try 4 terms with DFS
        found = False
        smooth_le_n = [x for x in smooth if x < n]
        best = None
        
        def dfs(terms, remaining, min_idx, target):
            nonlocal best
            if best is not None:
                return True
            if len(terms) >= 4:
                if sum(terms) == target:
                    best = list(terms)
                    return True
                return False
            if not terms:
                # Need to pick from smooth_le_n
                pass
            
            for idx in range(min_idx, len(smooth_le_n)):
                val = smooth_le_n[idx]
                if val > remaining:
                    break
                # Check antichain with existing terms
                valid = True
                for t in terms:
                    if t > val:
                        if t % val == 0:
                            valid = False
                            break
                    else:
                        if val % t == 0:
                            valid = False
                            break
                if not valid:
                    continue
                new_terms = terms + [val]
                if dfs(new_terms, remaining - val, idx + 1, target):
                    return True
            return False
        
        if dfs([], n, 0, n):
            answers.append(best)
            continue
        
        # Fallback for 5+ terms - use greedy approach
        # Just find any valid antichain summing to n
        # For small n, brute force with more terms
        found = False
        for count in range(5, 8):
            smooth_le_n = [x for x in smooth if x <= n]
            for combo in combinations(smooth_le_n, count):
                if sum(combo) != n:
                    continue
                # Check antichain
                valid = True
                for i in range(count):
                    for j in range(i + 1, count):
                        a, b = combo[i], combo[j]
                        if a > b:
                            a, b = b, a
                        if b % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    answers.append(list(combo))
                    found = True
                    break
            if found:
                break
        if not found:
            # Last resort: single term if possible, otherwise [1, n-1] adjusted
            if n in smooth:
                answers.append([n])
            else:
                # Find any valid representation
                answers.append([1, n - 1] if n - 1 in smooth and n > 2 else [n])
    
    return answers
# EVOLVE-BLOCK-END
