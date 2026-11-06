import time

def _subset_sums(arr):
    """Return set of all subset sums (including empty subset = 0)."""
    sums = {0}
    for x in arr:
        sums = sums | {s + x for s in sums}
    return sums

def _distinct(arr):
    """True iff all 2^n subset sums are distinct."""
    sums = {0}
    for x in arr:
        sums = sums | {s + x for s in sums}
    return len(sums) == 1 << len(arr)

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before the 90s kill

    # Greedy construction: pick the smallest valid next element each step.
    # This is fast because we only need to check one candidate at a time.
    arr = []
    sums = {0}
    for i in range(12):
        candidate = 1
        while True:
            new_sums = {s + candidate for s in sums}
            if new_sums.isdisjoint(sums):
                sums |= new_sums
                arr.append(candidate)
                break
            candidate += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if len(arr) < 12:
        # Fallback: powers of two are always valid
        return [1 << i for i in range(12)]

    # Local search: repeatedly try to reduce the largest element to a smaller
    # unused value that keeps all subset sums distinct.
    while time.time() < deadline:
        # Work on the largest element first
        idx = max(range(12), key=lambda i: arr[i])
        old = arr[idx]
        others = {arr[j] for j in range(12) if j != idx}
        found = False
        # Try smaller values, descending, skipping values already in use
        for v in range(old - 1, 0, -1):
            if v in others:
                continue
            arr[idx] = v
            if _distinct(arr):
                found = True
                break
            arr[idx] = old
            if time.time() > deadline:
                break
        if not found:
            arr[idx] = old
            # If the largest element can't be reduced, try a broader pass
            # over all elements to see if any single reduction works.
            any_improve = False
            for j in range(12):
                if time.time() > deadline:
                    break
                oj = arr[j]
                oj_set = {arr[k] for k in range(12) if k != j}
                for v in range(oj - 1, 0, -1):
                    if v in oj_set:
                        continue
                    arr[j] = v
                    if _distinct(arr):
                        any_improve = True
                        break
                    arr[j] = oj
                    if time.time() > deadline:
                        break
                if any_improve:
                    break
            if not any_improve:
                break  # local minimum

    # Final safety check
    if not _distinct(arr):
        return [1 << i for i in range(12)]

    return arr
