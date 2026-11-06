import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def conflicts(a, b):
        return (a * b) % (a + b) == 0

    # Start with all odds (mutually compatible)
    odds = list(range(1, 257, 2))
    selected = set(odds)

    # Add evens that are compatible with all odds
    for e in range(2, 257, 2):
        ok = True
        for o in odds:
            if conflicts(e, o):
                ok = False
                break
        if ok:
            selected.add(e)

    # Resolve conflicts among added evens
    evens_in_set = sorted(x for x in selected if x % 2 == 0)
    changed = True
    while changed:
        changed = False
        for i in range(len(evens_in_set)):
            if time.time() > deadline:
                break
            a = evens_in_set[i]
            if a not in selected:
                continue
            for j in range(i + 1, len(evens_in_set)):
                b = evens_in_set[j]
                if b not in selected:
                    continue
                if conflicts(a, b):
                    # Remove one with more total conflicts
                    ca = sum(1 for x in selected if x != a and conflicts(a, x))
                    cb = sum(1 for x in selected if x != b and conflicts(b, x))
                    if ca <= cb:
                        selected.discard(b)
                    else:
                        selected.discard(a)
                    changed = True
                    break

    def count_conf(x, s):
        c = 0
        for y in s:
            if y != x and conflicts(x, y):
                c += 1
        return c

    # Greedy improvement: add evens with 0 conflicts, or swap 1-conflict
    for _ in range(500):
        if time.time() > deadline:
            break
        improved = False
        for e in range(2, 257, 2):
            if e in selected:
                continue
            conf = [y for y in selected if conflicts(e, y)]
            if len(conf) == 0:
                selected.add(e)
                improved = True
            elif len(conf) == 1:
                p = conf[0]
                if count_conf(p, selected) == 1:
                    selected.discard(p)
                    selected.add(e)
                    improved = True
        if not improved:
            break

    # Randomized local search
    random.seed(42)
    best_size = len(selected)
    best_set = set(selected)

    for it in range(2000000):
        if time.time() > deadline:
            break

        r = random.random()
        if r < 0.5:
            # Try to add an unselected element
            unselected = [x for x in range(1, 257) if x not in selected]
            if not unselected:
                break
            e = random.choice(unselected)
            conf = [y for y in selected if conflicts(e, y)]
            if len(conf) == 0:
                selected.add(e)
            elif len(conf) == 1:
                p = conf[0]
                if count_conf(p, selected) == 1:
                    selected.discard(p)
                    selected.add(e)
        else:
            # Swap: remove one, add one
            a = random.choice(list(selected))
            selected.discard(a)
            unselected = [x for x in range(1, 257) if x not in selected]
            if unselected:
                added = False
                # Try a few candidates
                for c in random.sample(unselected, min(len(unselected), 20)):
                    if count_conf(c, selected) == 0:
                        selected.add(c)
                        added = True
                        break
                if not added:
                    selected.add(a)  # revert

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)

    if len(best_set) > len(selected):
        selected = best_set

    return list(selected)
