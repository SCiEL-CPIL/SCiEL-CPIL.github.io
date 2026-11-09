import random
import time

N = 256

def _is_sidon(S):
    S = sorted(S)
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return sums

def _is_maximal(S, sums):
    S_set = set(S)
    for x in range(1, N+1):
        if x in S_set:
            continue
        # Check if x can be added
        # x can be added if: 2x not in sums, and for all s in S, x+s not in sums
        if 2*x in sums:
            continue
        blocked = False
        for s in S:
            if x + s in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_maximal():
    selected = []
    sums = set()
    for x in range(1, N+1):
        added = [x + a for a in selected] + [2*x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _reduce(S):
    """Try to remove elements while maintaining maximality."""
    S = sorted(S)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if not candidate:
                continue
            if not _is_sidon(candidate):
                continue
            sums = _get_sums(candidate)
            if _is_maximal(candidate, sums):
                S = candidate
                changed = True
                break
    return S

def construct():
    deadline = time.time() + 25  # safety margin
    
    best = _greedy_maximal()
    best_size = len(best)
    
    # Try to reduce the greedy result
    reduced = _reduce(best)
    if len(reduced) < best_size:
        best = reduced
        best_size = len(best)
    
    # Try random approaches
    random.seed(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random Sidon set construction
        S = set()
        sums = set()
        # Try to build a random Sidon set
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            added = [x + a for a in S] + [2*x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                S.add(x)
                sums.update(added)
        
        if len(S) >= best_size:
            continue
        
        # Check maximality
        if _is_maximal(sorted(S), sums):
            best = sorted(S)
            best_size = len(best)
            # Try to reduce further
            reduced = _reduce(best)
            if len(reduced) < best_size:
                best = reduced
                best_size = len(best)
    
    return best
