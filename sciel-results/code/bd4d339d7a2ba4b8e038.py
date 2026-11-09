"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_smooth(x):
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _build_smooth(max_val=128):
    return [x for x in range(1, max_val + 1) if _is_smooth(x)]

def _valid_antichain(terms):
    # No term divides another
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            if b % a == 0 or a % b == 0:
                return False
    return True

def construct():
    deadline = time.time() + 85.0  # 85s deadline, 5s safety margin
    
    smooth = _build_smooth(128)
    
    # Precompute for each target n, the list of smooth numbers <= n
    # and try to find minimal antichain representations
    
    answers = [None] * 129  # index 0 unused
    
    # For each target, try 1 term, then 2, then 3, etc.
    # We want to minimize total terms across all 128 targets.
    # Strategy: for each target, find the minimum number of terms needed.
    
    # Precompute which smooth numbers are <= n for each n
    smooth_by_target = {}
    for n in range(1, 129):
        smooth_by_target[n] = [x for x in smooth if x <= n]
    
    # For each target, find the best (minimum terms) antichain representation
    for n in range(1, 129):
        if time.time() > deadline:
            break
        candidates = smooth_by_target[n]
        answer = None
        
        # Try 1 term
        for x in candidates:
            if x == n:
                answer = [x]
                break
            if time.time() > deadline:
                break
        
        if answer is None:
            # Try 2 terms
            for i in range(len(candidates)):
                if time.time() > deadline:
                    break
                a = candidates[i]
                for j in range(i + 1, len(candidates)):
                    b = candidates[j]
                    if a + b == n:
                        # Check antichain: a doesn't divide b and b doesn't divide a
                        if b % a != 0 and a % b != 0:
                            answer = [a, b]
                            break
                if answer is not None:
                    break
        
        if answer is None:
            # Try 3 terms
            found = False
            for i in range(len(candidates)):
                if time.time() > deadline or found:
                    break
                a = candidates[i]
                for j in range(i + 1, len(candidates)):
                    if time.time() > deadline or found:
                        break
                    b = candidates[j]
                    rem = n - a - b
                    if rem <= 0:
                        continue
                    # Need a third term c > b (to keep sorted) such that c = rem
                    # But c must be in candidates and c > b
                    # Actually we just need c to be a smooth number <= n, distinct from a and b
                    # and the three form an antichain
                    if rem in smooth_by_target[n] and rem > b and rem != a and rem != b:
                        # Check antichain among a, b, rem
                        if (b % a != 0 and a % b != 0 and
                            rem % a != 0 and a % rem != 0 and
                            rem % b != 0 and b % rem != 0):
                            answer = [a, b, rem]
                            found = True
                            break
        
        if answer is None:
            # Try 4 terms - use combinations but limit search
            found = False
            cands = candidates
            L = len(cands)
            for i in range(L):
                if time.time() > deadline or found:
                    break
                a = cands[i]
                for j in range(i + 1, L):
                    if time.time() > deadline or found:
                        break
                    b = cands[j]
                    for k in range(j + 1, L):
                        if time.time() > deadline or found:
                            break
                        c = cands[k]
                        rem = n - a - b - c
                        if rem <= c:
                            continue
                        if rem in smooth_by_target[n]:
                            # Check all 4 form antichain
                            terms = [a, b, c, rem]
                            if _valid_antichain(terms):
                                answer = terms
                                found = True
                                break
        
        if answer is None:
            # Fallback: use the seed's approach (try up to 30 terms via combinations)
            # This should rarely be needed
            for count in range(4, 10):
                if time.time() > deadline:
                    break
                for terms in combinations(cands, count):
                    if sum(terms) == n and _valid_antichain(terms):
                        answer = list(terms)
                        break
                if answer is not None:
                    break
        
        answers[n] = answer
    
    # Fill in any missing answers with a fallback
    for n in range(1, 129):
        if answers[n] is None:
            # Last resort: try all combinations up to 30
            cands = smooth_by_target[n]
            answer = None
            for count in range(1, min(31, len(cands) + 1)):
                if time.time() > deadline:
                    break
                for terms in combinations(cands, count):
                    if sum(terms) == n and _valid_antichain(terms):
                        answer = list(terms)
                        break
                if answer is not None:
                    break
            answers[n] = answer
    
    return answers[1:129]
# EVOLVE-BLOCK-END
