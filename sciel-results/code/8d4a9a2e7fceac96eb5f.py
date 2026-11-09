# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _conflict_pairs(x, S):
    """Return pairs (a, b) in S such that {x, a, b} is a bad triple."""
    conflicts = []
    for i, a in enumerate(S):
        for j in range(i + 1, len(S)):
            b = S[j]
            if _is_bad_triple(x, a, b):
                conflicts.append((a, b))
    return conflicts

def _greedy_construct(ordering):
    """Greedy construction following given ordering."""
    selected = []
    for x in ordering:
        can_add = True
        for i, a in enumerate(selected):
            if not can_add:
                break
            for j in range(i + 1, len(selected)):
                b = selected[j]
                if _is_bad_triple(x, a, b):
                    can_add = False
                    break
        if can_add:
            selected.append(x)
    return selected

def _local_search_add(S):
    """Try to add elements not in S to S."""
    S = list(S)
    n = len(S)
    changed = True
    while changed and n < 128:
        changed = False
        for x in range(1, 129):
            if x in S:
                continue
            if not _conflict_pairs(x, S):
                S.append(x)
                n += 1
                changed = True
                break
    return S

def _local_search_1swap(S):
    """Try 1-for-1 swaps to increase size."""
    S = list(S)
    n = len(S)
    improved = True
    while improved and n < 128:
        improved = False
        for i in range(len(S)):
            if improved:
                break
            for x in range(1, 129):
                if x in S:
                    continue
                trial = S[:i] + S[i+1:] + [x]
                if not _conflict_pairs(x, trial):
                    # Check that removing S[i] and adding x doesn't create conflicts
                    # among the remaining elements
                    valid = True
                    for j in range(len(trial)):
                        for k in range(j + 1, len(trial)):
                            for m in range(k + 1, len(trial)):
                                if _is_bad_triple(trial[j], trial[k], trial[m]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        S = trial
                        improved = True
                        break
    return S

def _local_search_2swap(S):
    """Try 2-for-1 swaps: remove 2 elements, add 1."""
    S = list(S)
    n = len(S)
    if n < 2:
        return S
    improved = True
    while improved and n < 128:
        improved = False
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                for x in range(1, 129):
                    if x in S:
                        continue
                    trial = S[:i] + S[i+1:j] + S[j+1:] + [x]
                    # Check if adding x to S minus {S[i], S[j]} creates conflicts
                    base = S[:i] + S[i+1:j] + S[j+1:]
                    if _conflict_pairs(x, base):
                        continue
                    # Check full validity
                    valid = True
                    L = len(trial)
                    for a in range(L):
                        for b in range(a + 1, L):
                            for c in range(b + 1, L):
                                if _is_bad_triple(trial[a], trial[b], trial[c]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        S = trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return S

def _validate(S):
    """Check if S contains no bad triples."""
    for a, b, c in combinations(S, 3):
        if _is_bad_triple(a, b, c):
            return False
    return True

def construct():
    deadline = time.time() + 85.0
    
    # Start with a simple greedy
    best_result = _greedy_construct(list(range(1, 129)))
    best_size = len(best_result)
    
    # Try different orderings
    orderings = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        list(range(1, 129, 2)),  # odds
        [x for x in range(1, 129) if x % 2 == 0],  # evens
    ]
    
    # Add some random orderings
    random.seed(42)
    for _ in range(10):
        perm = list(range(1, 129))
        random.shuffle(perm)
        orderings.append(perm)
    
    for ordering in orderings:
        if time.time() > deadline - 5.0:
            break
        S = _greedy_construct(ordering)
        S = _local_search_add(S)
        if len(S) > best_size:
            best_size = len(S)
            best_result = S
    
    # Local search on best result
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            S = _local_search_add(list(best_result))
            if len(S) > best_size:
                best_size = len(S)
                best_result = S
                improved = True
            S2 = _local_search_1swap(list(best_result))
            if len(S2) > best_size:
                best_size = len(S2)
                best_result = S2
                improved = True
            S3 = _local_search_2swap(list(best_result))
            if len(S3) > best_size:
                best_size = len(S3)
                best_result = S3
                improved = True
    
    # More random restarts
    if time.time() < deadline - 5.0:
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            perm = list(range(1, 129))
            random.shuffle(perm)
            S = _greedy_construct(perm)
            S = _local_search_add(S)
            if len(S) > best_size:
                best_size = len(S)
                best_result = S
    
    # Final validation
    if best_result and not _validate(best_result):
        # Rebuild safely
        best_result = _greedy_construct(list(range(1, 129)))
        best_result = _local_search_add(best_result)
    
    if not best_result:
        best_result = [1]
    
    return best_result
# EVOLVE-BLOCK-END
