"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _conflict(a, b):
    """Return True if (a*b) % (a+b) == 0 for distinct a,b."""
    if a == b:
        return False
    return (a * b) % (a + b) == 0

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin within 60s

    N = 256
    odds = list(range(1, N + 1, 2))
    evens = list(range(2, N + 1, 2))

    best_set = set(odds)
    best_size = len(best_set)

    def can_add(x, s):
        for y in s:
            if _conflict(x, y):
                return False
        return True

    # Precompute conflict matrix for speed? 256x256 = 65536 booleans, cheap.
    # Build a bitmask or set of conflicts per element.
    # conflicts[i] = set of j != i such that _conflict(i+1, j+1)
    # Use 0-indexed: element value v = idx+1
    conflicts = [set() for _ in range(N)]
    for i in range(N):
        vi = i + 1
        for j in range(i + 1, N):
            vj = j + 1
            if _conflict(vi, vj):
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Represent set as a boolean list of size N (index = value-1)
    def valid(mask):
        # mask: list of bools, index = value-1
        for i in range(N):
            if not mask[i]:
                continue
            for j in conflicts[i]:
                if j > i and mask[j]:
                    return False
        return True

    # Start with all odds
    mask = [False] * N
    for v in odds:
        mask[v - 1] = True

    cur_size = sum(mask)
    best_size = cur_size
    best_mask = mask[:]

    # Greedy: try to add evens in an order that maximizes additions
    # Try multiple random orders
    def greedy_add(mask):
        # try to add evens that are currently not in mask
        added = True
        while added:
            added = False
            for v in evens:
                idx = v - 1
                if mask[idx]:
                    continue
                ok = True
                for j in conflicts[idx]:
                    if mask[j]:
                        ok = False
                        break
                if ok:
                    mask[idx] = True
                    added = True
        return mask

    # Phase 1: greedy add evens to odds
    greedy_mask = mask[:]
    greedy_mask = greedy_add(greedy_mask)
    gsize = sum(greedy_mask)
    if gsize > best_size:
        best_size = gsize
        best_mask = greedy_mask[:]
    cur_mask = greedy_mask[:]
    cur_size = gsize

    # Phase 2: local search with 1-swap and add/remove moves
    # A move: remove one element, add another (or just add, or just remove)
    # We want to maximize size while staying valid.
    # Use a "score" = size. Try moves that keep validity.

    def try_moves(mask, size, budget):
        # 1-swap: pick i in set, j not in set, swap if valid
        # add: pick j not in set, add if valid
        # remove: not useful for maximizing, skip unless to enable add
        # We'll do: random add attempts, random swap attempts
        in_set = [i for i in range(N) if mask[i]]
        not_in_set = [i for i in range(N) if not mask[i]]
        # Rebuild lists periodically
        while time.time() < budget:
            # Try a random add
            if not_in_set:
                j = random.choice(not_in_set)
                ok = True
                for k in conflicts[j]:
                    if mask[k]:
                        ok = False
                        break
                if ok:
                    mask[j] = True
                    size += 1
                    in_set.append(j)
                    not_in_set.remove(j)
                    if size > best_size_global[0]:
                        best_size_global[0] = size
                        best_mask_global[0] = mask[:]
                    continue
            # Try a random swap: remove i, add j
            if in_set and not_in_set:
                i = random.choice(in_set)
                # temporarily remove i
                mask[i] = False
                size -= 1
                in_set.remove(i)
                not_in_set.append(i)
                # try to add some j
                improved = False
                for _ in range(3):
                    j = random.choice(not_in_set)
                    if j == i:
                        continue
                    ok = True
                    for k in conflicts[j]:
                        if mask[k]:
                            ok = False
                            break
                    if ok:
                        mask[j] = True
                        size += 1
                        in_set.append(j)
                        not_in_set.remove(j)
                        improved = True
                        break
                if not improved:
                    # revert
                    mask[i] = True
                    size += 1
                    in_set.append(i)
                    not_in_set.remove(i)
                else:
                    if size > best_size_global[0]:
                        best_size_global[0] = size
                        best_mask_global[0] = mask[:]
        return mask, size

    best_size_global = [best_size]
    best_mask_global = [best_mask]

    # Multiple restarts with different greedy orders
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        # Different random order of evens for greedy
        evens_shuffled = evens[:]
        random.shuffle(evens_shuffled)
        m = [False] * N
        for v in odds:
            m[v - 1] = True
        # greedy add in shuffled order
        for v in evens_shuffled:
            idx = v - 1
            if m[idx]:
                continue
            ok = True
            for k in conflicts[idx]:
                if m[k]:
                    ok = False
                    break
            if ok:
                m[idx] = True
        size = sum(m)
        if size > best_size_global[0]:
            best_size_global[0] = size
            best_mask_global[0] = m[:]
        # local search from here
        budget = min(deadline - 2.0, time.time() + 3.0)
        m, size = try_moves(m, size, budget)

    # Final: return best
    result = [i + 1 for i in range(N) if best_mask_global[0][i]]
    return result

# EVOLVE-BLOCK-END
