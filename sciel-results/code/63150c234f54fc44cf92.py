import time
import random

L = 5354228880  # lcm(1..24)
N = 24
ALL = list(range(1, 25))
RECIP = [0] + [L // i for i in range(1, 25)]  # RECIP[i] = L//i


def has_subsum(weights, target):
    """
    Check if target can be formed by a non-empty subset of weights (positive ints).
    Uses meet-in-the-middle for efficiency.
    """
    if not weights or target <= 0:
        return False
    
    # Filter to relevant weights
    weights = [w for w in weights if 0 < w <= target]
    if not weights:
        return False
    
    # Deduplicate
    weights = sorted(set(weights))
    n = len(weights)
    if n == 0:
        return False
    
    if sum(weights) < target:
        return False
    
    if target in weights:
        return True
    
    # Meet-in-the-middle
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    def gen_sums(arr):
        sums = {0}
        for w in arr:
            new = set()
            for s in sums:
                ns = s + w
                if ns <= target:
                    new.add(ns)
            sums |= new
        return sums
    
    left_sums = gen_sums(left)
    right_sums = gen_sums(right)
    
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False


def find_violation(S):
    """
    S: list of ints.
    Returns index of violator if found, else None.
    A violation: for some x in S, L//x == sum of L//y for a non-empty subset of S\{x}.
    """
    if not S:
        return None
    m = len(S)
    recip_list = [L // x for x in S]
    for i in range(m):
        target = recip_list[i]
        others = [recip_list[j] for j in range(m) if j != i]
        if target > sum(others):
            continue
        if has_subsum(others, target):
            return i
    return None


def is_valid(S):
    """
    S: list of ints. Return True if S is a valid reciprocal-subsum-free set.
    """
    if not S:
        return True
    return find_violation(S) is None


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    def now():
        return time.time()

    best_set = None
    best_size = 0

    # Seed: upper half
    seed = list(range(13, 25))
    if is_valid(seed):
        best_set = seed[:]
        best_size = len(seed)
    else:
        best_set = [24]
        best_size = 1

    # Greedy addition with multiple orderings
    def greedy_add(order, start_set=None):
        S = set(start_set) if start_set is not None else set()
        S_list = list(S)
        for x in order:
            if now() > deadline:
                break
            if x in S:
                continue
            trial = S_list + [x]
            if is_valid(trial):
                S.add(x)
                S_list = list(S)
        return S_list

    orderings = []
    orderings.append(ALL[:])
    orderings.append(ALL[::-1])
    orderings.append(sorted(ALL, key=lambda x: L // x))
    orderings.append(sorted(ALL, key=lambda x: L // x, reverse=True))
    orderings.append(sorted(ALL, key=lambda x: x % 7))
    orderings.append(sorted(ALL, key=lambda x: x % 13))

    rng = random.Random(42)
    for _ in range(40):
        o = ALL[:]
        rng.shuffle(o)
        orderings.append(o)

    for o in orderings:
        if now() > deadline:
            break
        res = greedy_add(o, start_set=None)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res[:]

    # Repair: start from full set, remove violators
    def repair(S):
        S = set(S)
        while now() < deadline and S:
            S_list = list(S)
            viol_idx = find_violation(S_list)
            if viol_idx is None:
                return S_list
            S.remove(S_list[viol_idx])
        return list(S)

    S_full = repair(ALL[:])
    if len(S_full) > best_size:
        best_size = len(S_full)
        best_set = S_full[:]

    # Add back missing elements
    def add_back(S):
        S = set(S)
        missing = [x for x in ALL if x not in S]
        for x in missing:
            if now() > deadline:
                break
            trial = list(S) + [x]
            if is_valid(trial):
                S.add(x)
        return list(S)

    if best_set is not None:
        improved = add_back(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]

    # Local search with swaps and additions
    def local_search(S, max_iters=300):
        S = set(S)
        S_list = list(S)
        iters = 0
        while iters < max_iters and now() < deadline:
            iters += 1
            improved = False

            # Try additions
            missing = [x for x in ALL if x not in S]
            if missing:
                rng2 = random.Random(iters + len(S) * 1000)
                rng2.shuffle(missing)
                for x in missing:
                    if now() > deadline:
                        break
                    trial = list(S) + [x]
                    if is_valid(trial):
                        S.add(x)
                        S_list = list(S)
                        improved = True
                        break
            if improved:
                continue

            # Try swaps: remove one, add another
            if len(S) > 0:
                a = rng.choice(list(S))
                missing = [x for x in ALL if x not in S]
                if missing:
                    rng.shuffle(missing)
                    for b in missing:
                        if now() > deadline:
                            break
                        new_set = (S - {a}) | {b}
                        if is_valid(list(new_set)):
                            S = new_set
                            S_list = list(S)
                            improved = True
                            break

            if not improved:
                break

        return S_list

    if best_set is not None:
        ls = local_search(best_set, max_iters=300)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls[:]

    # Random restarts with repair + add back + local search
    for _ in range(80):
        if now() > deadline:
            break
        target_size = random.randint(max(2, best_size), 24)
        rand_set = random.sample(ALL, min(target_size, 24))
        repaired = repair(rand_set)
        added = add_back(repaired)
        if len(added) > best_size:
            best_size = len(added)
            best_set = added[:]
            ls = local_search(added, max_iters=150)
            if len(ls) > best_size:
                best_size = len(ls)
                best_set = ls[:]

    # Final polish: try all single-element swaps exhaustively
    if best_set is not None and now() < deadline:
        S = set(best_set)
        S_list = list(S)
        changed = True
        while changed and now() < deadline:
            changed = False
            for a in list(S):
                for b in ALL:
                    if b in S:
                        continue
                    new_set = (S - {a}) | {b}
                    if is_valid(list(new_set)):
                        # Check if this swap is beneficial (it's neutral for size, but may enable further additions)
                        S = new_set
                        changed = True
                        break
                if changed:
                    break
        # Try additions again after swaps
        missing = [x for x in ALL if x not in S]
        for x in missing:
            if now() > deadline:
                break
            trial = list(S) + [x]
            if is_valid(trial):
                S.add(x)
        S_list = list(S)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list

    # Final validation
    if best_set is None or not is_valid(best_set):
        best_set = [24]
        best_size = 1
    else:
        best_set = list(best_set)
        best_size = len(best_set)

    return best_set


if __name__ == "__main__":
    print(construct())
