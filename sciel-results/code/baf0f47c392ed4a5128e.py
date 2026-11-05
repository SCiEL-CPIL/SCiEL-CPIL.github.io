# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 75.0  # Safety margin before the 90 s kill

    # Precompute 2,3,5-smooth numbers up to 128.
    MAX_N = 128
    smooth = []
    smooth_set = set()
    for x in range(1, MAX_N + 1):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
            smooth_set.add(x)
    
    smooth.sort()
    
    # Precompute valid pairs for 2-term sums
    # sum -> list of (a, b) where a < b, a+b=sum, antichain
    pair_sums = defaultdict(list)
    for i in range(len(smooth)):
        a = smooth[i]
        for j in range(i + 1, len(smooth)):
            b = smooth[j]
            s = a + b
            if s > MAX_N:
                break
            # Check antichain: a does not divide b (since a < b)
            if b % a != 0:
                pair_sums[s].append((a, b))
                
    # Precompute valid triples for 3-term sums?
    # Might be too many, but let's see. 38 numbers. C(38,3) is ~8436.
    # We can just search on the fly for 3 terms, it's fast enough.
    
    answers = []
    
    for n in range(1, MAX_N + 1):
        if time.time() - t0 > DEADLINE:
            # Fallback: should not happen
            # Try to find any valid representation quickly
            # If we run out of time, we might have to return invalid or crash.
            # But the search is very fast, so this is unlikely.
            answers.append([n] if n in smooth_set else [1, n-1])
            continue

        # 1. Check 1 term
        if n in smooth_set:
            answers.append([n])
            continue

        # 2. Check 2 terms
        if n in pair_sums:
            # Pick the first valid pair
            a, b = pair_sums[n][0]
            answers.append([a, b])
            continue

        # 3. Check 3 terms
        found = False
        # Iterate a, b such that a < b and a+b < n
        # c = n - a - b must be > b and in smooth_set
        for i, a in enumerate(smooth):
            if a >= n:
                break
            # b must be > a
            for j in range(i + 1, len(smooth)):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    break
                if c in smooth_set:
                    # Check antichain: a < b < c
                    # Pairs: (a,b), (a,c), (b,c)
                    if b % a != 0 and c % a != 0 and c % b != 0:
                        answers.append([a, b, c])
                        found = True
                        break
            if found:
                break
        
        if found:
            continue

        # 4. Check 4 terms
        found = False
        for i, a in enumerate(smooth):
            if a >= n:
                break
            for j in range(i + 1, len(smooth)):
                b = smooth[j]
                if a + b >= n:
                    break
                for k in range(j + 1, len(smooth)):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        break
                    if d in smooth_set:
                        # Check antichain: a < b < c < d
                        # Pairs: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                        if (b % a != 0 and c % a != 0 and d % a != 0 and
                            c % b != 0 and d % b != 0 and d % c != 0):
                            answers.append([a, b, c, d])
                            found = True
                            break
                    if found: break
                if found: break
            if found: break
        
        if found:
            continue

        # 5. Fallback for 5+ terms (unlikely for n<=128)
        # Use a simple DFS
        def dfs(remaining, last_idx, current_terms, max_terms):
            if remaining == 0:
                return current_terms
            if len(current_terms) == max_terms:
                return None
            
            # Pruning
            if current_terms:
                min_next = current_terms[-1] + 1
            else:
                min_next = 1
            
            terms_left = max_terms - len(current_terms)
            if remaining < terms_left * min_next:
                return None
            
            for idx in range(last_idx + 1, len(smooth)):
                val = smooth[idx]
                if val > remaining:
                    break
                # Antichain check
                valid = True
                for t in current_terms:
                    if val % t == 0:
                        valid = False
                        break
                if not valid:
                    continue
                res = dfs(remaining - val, idx, current_terms + [val], max_terms)
                if res:
                    return res
            return None

        for k in range(5, 31):
            if time.time() - t0 > DEADLINE:
                break
            res = dfs(n, -1, [], k)
            if res:
                answers.append(res)
                found = True
                break
        
        if not found:
            # Last resort
            answers.append([n])

    return answers
# EVOLVE-BLOCK-END
