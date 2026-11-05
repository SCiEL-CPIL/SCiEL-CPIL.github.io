"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth_set = set()
    a = 1
    while a <= 128:
        b = a
        while b <= 128:
            c = b
            while c <= 128:
                smooth_set.add(c)
                c *= 5
            b *= 3
        a *= 2
    smooth = sorted(smooth_set)
    smooth_set = set(smooth)
    
    # Precompute: for each smooth number, which other smooth numbers does it divide?
    # div_map[x] = set of smooth numbers that x divides
    div_map = {}
    for x in smooth:
        div_map[x] = set()
        for y in smooth:
            if y > x and y % x == 0:
                div_map[x].add(y)
    
    def is_antichain(terms):
        for i in range(len(terms)):
            for j in range(i+1, len(terms)):
                a, b = terms[i], terms[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    return False
        return True
    
    answers = []
    total_terms = 0
    
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: use 3-term or whatever we can quickly find
            answer = None
            for count in (3, 2, 1):
                candidates = [x for x in smooth if x <= n]
                from itertools import combinations
                for terms in combinations(candidates, count):
                    if sum(terms) == n and is_antichain(terms):
                        answer = list(terms)
                        break
                if answer is not None:
                    break
            if answer is None:
                answer = [n] if n in smooth_set else [1, n-1]
            answers.append(answer)
            total_terms += len(answer)
            continue
        
        found = None
        
        # Try 1 term
        if n in smooth_set:
            found = [n]
        
        # Try 2 terms
        if found is None:
            for a in smooth:
                if a >= n:
                    break
                b = n - a
                if b in smooth_set and b > a:
                    if b % a != 0 and a % b != 0:
                        found = [a, b]
                        break
        
        # Try 3 terms
        if found is None:
            candidates = [x for x in smooth if x < n]
            for i in range(len(candidates)):
                a = candidates[i]
                for j in range(i+1, len(candidates)):
                    b = candidates[j]
                    c = n - a - b
                    if c <= b or c not in smooth_set:
                        continue
                    # Check antichain
                    if c % a == 0 or c % b == 0 or b % a == 0:
                        continue
                    found = [a, b, c]
                    break
                if found is not None:
                    break
        
        # Try 4 terms
        if found is None:
            candidates = [x for x in smooth if x < n]
            for i in range(len(candidates)):
                a = candidates[i]
                for j in range(i+1, len(candidates)):
                    b = candidates[j]
                    for k in range(j+1, len(candidates)):
                        c = candidates[k]
                        d = n - a - b - c
                        if d <= c or d not in smooth_set:
                            continue
                        # Check antichain for all pairs
                        terms = [a, b, c, d]
                        ok = True
                        for p in range(4):
                            for q in range(p+1, 4):
                                x, y = terms[p], terms[q]
                                if x > y:
                                    x, y = y, x
                                if y % x == 0:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            found = terms
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
        
        # Fallback: use brute force with higher counts
        if found is None:
            from itertools import combinations
            candidates = [x for x in smooth if x <= n]
            for count in (5, 6, 7, 8, 10, 15, 20, 30):
                for terms in combinations(candidates, count):
                    if sum(terms) == n and is_antichain(terms):
                        found = list(terms)
                        break
                if found is not None:
                    break
        
        if found is None:
            # Last resort
            found = [n] if n in smooth_set else [1, n-1]
        
        answers.append(found)
        total_terms += len(found)
    
    return answers
# EVOLVE-BLOCK-END
