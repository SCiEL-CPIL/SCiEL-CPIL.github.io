# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
    max_target = 128
    
    # Generate all 2,3,5-smooth numbers up to max_target
    smooth = []
    for x in range(1, max_target + 1):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # Precompute divisibility: div[i][j] = True if smooth[i] divides smooth[j]
    n_smooth = len(smooth)
    div = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if smooth[j] % smooth[i] == 0:
                div[i][j] = True
    
    # For each target n (1..128), find minimum number of terms
    # such that we have a valid antichain summing to n
    # We search in increasing order of count: 1, 2, 3, ...
    
    answers = [None] * (max_target + 1)
    
    # Pre-filter smooth numbers by value for each target
    # For target n, only smooth numbers <= n can be used
    
    # Try count = 1 first for all targets
    for n in range(1, max_target + 1):
        for s in smooth:
            if s == n:
                answers[n] = [s]
                break
            if s > n:
                break
    
    # Try count = 2 for remaining targets
    for n in range(1, max_target + 1):
        if answers[n] is not None:
            continue
        if time.time() > deadline:
            break
        candidates = [s for s in smooth if s <= n]
        # Sort by value for early termination
        for i in range(len(candidates)):
            a = candidates[i]
            if a * 2 > n:  # since terms are distinct and positive, min other term > a
                # Actually we need b = n - a, and b > a for distinctness, but also b must be in candidates
                # Let's just check directly
                pass
            b = n - a
            if b <= a:
                continue
            if b > n:
                continue
            # Check if b is smooth
            if b not in candidates:
                continue
            # Check antichain: a does not divide b and b does not divide a
            # Since b > a, b cannot divide a. So just check a doesn't divide b.
            if b % a == 0:
                continue
            answers[n] = [a, b]
            break
    
    # Try count = 3 for remaining targets
    for n in range(1, max_target + 1):
        if answers[n] is not None:
            continue
        if time.time() > deadline:
            break
        candidates = [s for s in smooth if s <= n]
        # Use combinations of 3
        for combo in combinations(candidates, 3):
            if sum(combo) == n:
                # Check antichain: no element divides another
                valid = True
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = combo[i], combo[j]
                        if b > a and b % a == 0:
                            valid = False
                            break
                        if a > b and a % b == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    answers[n] = list(combo)
                    break
    
    # Try count = 4 for remaining targets
    for n in range(1, max_target + 1):
        if answers[n] is not None:
            continue
        if time.time() > deadline:
            break
        candidates = [s for s in smooth if s <= n]
        # For 4 terms, we need to be more careful about performance
        # Use a smarter approach: fix first two, then find two more that sum to remaining
        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                rem = n - a - b
                if rem <= 0:
                    break
                # Find two distinct smooth numbers from candidates (excluding a, b) that sum to rem
                # and form an antichain with a and b
                found = False
                for k in range(len(candidates)):
                    c = candidates[k]
                    if c == a or c == b:
                        continue
                    if c >= rem:
                        break
                    d = rem - c
                    if d <= c:
                        continue
                    # Check if d is in candidates
                    if d not in candidates:
                        continue
                    if d == a or d == b or d == c:
                        continue
                    # Check antichain among {a, b, c, d}
                    terms = [a, b, c, d]
                    valid = True
                    for x_idx in range(4):
                        for y_idx in range(x_idx + 1, 4):
                            x_val, y_val = terms[x_idx], terms[y_idx]
                            if y_val > x_val and y_val % x_val == 0:
                                valid = False
                                break
                            if x_val > y_val and x_val % y_val == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        answers[n] = [a, b, c, d]
                        found = True
                        break
                if found:
                    break
            if answers[n] is not None:
                break
    
    # Try count = 5 for remaining targets
    for n in range(1, max_target + 1):
        if answers[n] is not None:
            continue
        if time.time() > deadline:
            break
        candidates = [s for s in smooth if s <= n]
        # For 5 terms, use combinations but with pruning
        for combo in combinations(candidates, 5):
            if sum(combo) == n:
                # Check antichain
                valid = True
                for i in range(5):
                    for j in range(i + 1, 5):
                        a, b = combo[i], combo[j]
                        if b > a and b % a == 0:
                            valid = False
                            break
                        if a > b and a % b == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    answers[n] = list(combo)
                    break
    
    # Try count = 6 for remaining targets
    for n in range(1, max_target + 1):
        if answers[n] is not None:
            continue
        if time.time() > deadline:
            break
        candidates = [s for s in smooth if s <= n]
        for combo in combinations(candidates, 6):
            if sum(combo) == n:
                valid = True
                for i in range(6):
                    for j in range(i + 1, 6):
                        a, b = combo[i], combo[j]
                        if b > a and b % a == 0:
                            valid = False
                            break
                        if a > b and a % b == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    answers[n] = list(combo)
                    break
    
    # Try higher counts if needed
    for count in range(7, 31):
        any_left = False
        for n in range(1, max_target + 1):
            if answers[n] is None:
                any_left = True
                break
        if not any_left:
            break
        if time.time() > deadline:
            break
        for n in range(1, max_target + 1):
            if answers[n] is not None:
                continue
            if time.time() > deadline:
                break
            candidates = [s for s in smooth if s <= n]
            for combo in combinations(candidates, count):
                if sum(combo) == n:
                    valid = True
                    for i in range(count):
                        for j in range(i + 1, count):
                            a, b = combo[i], combo[j]
                            if b > a and b % a == 0:
                                valid = False
                                break
                            if a > b and a % b == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        answers[n] = list(combo)
                        break
    
    # Verify all answers are found
    for n in range(1, max_target + 1):
        if answers[n] is None:
            # Fallback: should not happen, but provide a valid answer
            # Use a greedy approach as last resort
            answers[n] = [n]  # This is valid since n is smooth if n <= 128... wait, not all n are smooth
            # Actually, we need to find a valid representation
            # Let's try again with a more exhaustive search
            candidates = [s for s in smooth if s <= n]
            for cnt in range(1, 31):
                for combo in combinations(candidates, cnt):
                    if sum(combo) == n:
                        valid = True
                        for i in range(cnt):
                            for j in range(i + 1, cnt):
                                a, b = combo[i], combo[j]
                                if b > a and b % a == 0:
                                    valid = False
                                    break
                                if a > b and a % b == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            answers[n] = list(combo)
                            break
                if answers[n] is not None:
                    break
    
    return answers[1:]
# EVOLVE-BLOCK-END
