import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 82.0
    n = 128

    # Known good construction: U={65..128} (sum-free upper block), L={17..32} (sum-free, pair sums < 65)
    U = list(range(65, 129))
    L = list(range(17, 33))
    best = U + L
    best_size = len(best)  # 80

    # Precompute conflict hyperedges: for each triple (a,b,c) distinct, if a+b, a+c, b+c all in 1..128
    # then having all three of a,b,c in the set is forbidden.
    # We'll represent conflicts as a set of frozensets of size 3.
    # For efficiency in checking, we'll build for each pair (a,b) the set of c's that would create a conflict.
    # Actually, let's use a different approach: for a candidate set S, check validity by iterating triples.
    
    def _is_valid(S):
        S_set = set(S)
        for a, b, c in combinations(S, 3):
            if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
                return False
        return True

    def _can_add(S_list, S_set, x):
        """Check if adding x to S creates a forbidden triple."""
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                ab = a + b
                ax = a + x
                bx = b + x
                if ab not in S_set and ab != x:
                    continue
                if ax not in S_set and ax != x:
                    continue
                if bx not in S_set and bx != x:
                    continue
                return False
        return True

    # Phase 1: Greedy addition from 1..64
    S_list = list(best)
    S_set = set(S_list)
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(S_list, S_set, x):
            S_list.append(x)
            S_set.add(x)

    if len(S_list) > best_size:
        best = S_list
        best_size = len(best)

    # Phase 2: Local search - try removing one element and adding multiple
    for remove_idx in range(len(best)):
        if time.time() > deadline:
            break
        candidate = [x for i, x in enumerate(best) if i != remove_idx]
        cand_set = set(candidate)
        added = 0
        for x in range(1, 65):
            if time.time() > deadline:
                break
            if x in cand_set:
                continue
            if _can_add(candidate, cand_set, x):
                candidate.append(x)
                cand_set.add(x)
                added += 1
        if added > 1 and len(candidate) > best_size:
            best = candidate
            best_size = len(best)

    # Phase 3: Try removing two elements and adding more
    for i in range(len(best)):
        if time.time() > deadline:
            break
        for j in range(i + 1, len(best)):
            if time.time() > deadline:
                break
            candidate = [x for k, x in enumerate(best) if k != i and k != j]
            cand_set = set(candidate)
            added = 0
            for x in range(1, 65):
                if time.time() > deadline:
                    break
                if x in cand_set:
                    continue
                if _can_add(candidate, cand_set, x):
                    candidate.append(x)
                    cand_set.add(x)
                    added += 1
            if added > 2 and len(candidate) > best_size:
                best = candidate
                best_size = len(best)

    # Phase 4: Random restarts with different initial blocks
    random.seed(42)
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Try different upper block sizes
        upper_start = 60 + restart * 2
        upper = list(range(upper_start, 129))
        
        # Find largest lower block that works
        lower = []
        for x in range(1, upper_start):
            if time.time() > deadline:
                break
            trial = upper + lower + [x]
            trial_set = set(trial)
            valid = True
            for a, b, c in combinations(trial, 3):
                if (a + b) in trial_set and (a + c) in trial_set and (b + c) in trial_set:
                    valid = False
                    break
            if valid:
                lower.append(x)
        
        candidate = upper + lower
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)

    # Final verification
    if not _is_valid(best):
        return list(range(65, 129)) + list(range(17, 33))

    return [int(x) for x in best]
