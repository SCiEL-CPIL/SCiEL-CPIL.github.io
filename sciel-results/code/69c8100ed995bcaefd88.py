"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    # Generate 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    answers = []
    
    # Pre-compute minimal antichain representations for 1..128
    # We want to minimize the number of terms for each target.
    # The maximum number of terms needed is small.
    
    for n in range(1, 129):
        found = None
        
        # Try k=1
        if n in smooth_set:
            found = [n]
        else:
            # Try k=2
            # We need a + b = n, a < b, a, b in smooth, a not divide b
            # Since a < b, b cannot divide a. We only need to check if a divides b.
            # Actually, if a < b, b % a == 0 implies a divides b.
            # So we just need a % b != 0 (always true if a<b) and b % a != 0.
            candidates = [x for x in smooth if x < n]
            for a in candidates:
                b = n - a
                if b in smooth_set and b > a:
                    # Check antichain: a does not divide b
                    if b % a != 0:
                        found = [a, b]
                        break
            
            if found is None:
                # Try k=3
                # a + b + c = n, a < b < c
                # Check all pairs for divisibility
                candidates = [x for x in smooth if x < n]
                for a in candidates:
                    for b in candidates:
                        if b <= a: continue
                        c = n - a - b
                        if c in smooth_set and c > b:
                            # Check antichain for (a,b,c)
                            # a<b<c so c cannot divide a or b.
                            # b cannot divide a.
                            # Need to check: a|b, a|c, b|c
                            if b % a != 0 and c % a != 0 and c % b != 0:
                                found = [a, b, c]
                                break
                    if found is not None: break
            
            if found is None:
                # Try k=4
                candidates = [x for x in smooth if x < n]
                for a in candidates:
                    for b in candidates:
                        if b <= a: continue
                        for c in candidates:
                            if c <= b: continue
                            d = n - a - b - c
                            if d in smooth_set and d > c:
                                # Check antichain for (a,b,c,d)
                                # a<b<c<d
                                # Check pairs: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                                # Since sorted, only smaller can divide larger.
                                if (b % a != 0 and c % a != 0 and d % a != 0 and
                                    c % b != 0 and d % b != 0 and
                                    d % c != 0):
                                    found = [a, b, c, d]
                                    break
                        if found is not None: break
                    if found is not None: break
            
            if found is None:
                # Fallback: Try k=5, though unlikely needed for n<=128
                # For safety, if not found, we might need a more general search
                # But based on smooth number density, k=4 should suffice.
                # Let's add a general search for k=5 just in case.
                candidates = [x for x in smooth if x < n]
                for a in candidates:
                    for b in candidates:
                        if b <= a: continue
                        for c in candidates:
                            if c <= b: continue
                            for d in candidates:
                                if d <= c: continue
                                e = n - a - b - c - d
                                if e in smooth_set and e > d:
                                    # Check antichain
                                    terms = [a,b,c,d,e]
                                    is_anti = True
                                    for i in range(len(terms)):
                                        for j in range(i+1, len(terms)):
                                            if terms[j] % terms[i] == 0:
                                                is_anti = False
                                                break
                                        if not is_anti: break
                                    if is_anti:
                                        found = terms
                                        break
                            if found is not None: break
                        if found is not None: break
                    if found is not None: break
                
                if found is None:
                    # If still none, something is wrong or k>5 needed.
                    # Given constraints, this should not happen.
                    # We return an empty list or raise error? 
                    # The task says "All targets require valid representations".
                    # So we assume it exists.
                    found = [] # Placeholder

        answers.append(found)
        
    return answers
# EVOLVE-BLOCK-END
