import time

def construct():
    start = time.time()
    deadline = start + 55.0
    n = 12
    m = 40
    total = 1 << n

    # Precompute comparability: comp[i] is a set of masks j where i and j are comparable
    # Use list of lists for faster iteration
    comp = [None] * total
    for i in range(total):
        ci = []
        for j in range(total):
            if i != j and ((i & j) == i or (i & j) == j):
                ci.append(j)
        comp[i] = ci

    # Build a strong initial solution: use a chain structure
    # A chain of length 13 (empty to full) gives C(13,2)=78 comparable pairs
    # We want 40 sets. Use multiple overlapping chains.
    # Strategy: take all subsets of {0,1,2,3,4,5} that form a "grid"
    # Actually, let's use: for each k=0..5, all subsets of {0,...,k}
    # That gives 1+2+4+8+16+32 = 63, too many
    # Better: use a specific structured set

    # Initial: empty, all singletons, all pairs with 0, all triples with {0,1}, etc.
    initial = []
    initial.append(0)  # empty
    for i in range(n):
        initial.append(1 << i)  # 12 singletons
    for i in range(1, n):
        initial.append((1 << 0) | (1 << i))  # 11 pairs with 0
    for i in range(2, n):
        initial.append((1 << 0) | (1 << 1) | (1 << i))  # 10 triples with {0,1}
    # That's 1+12+11+10 = 34
    # Add 6 more carefully chosen sets
    initial.append((1 << 0) | (1 << 1) | (1 << 2) | (1 << 3))  # {0,1,2,3}
    initial.append((1 << 0) | (1 << 1) | (1 << 2) | (1 << 4))  # {0,1,2,4}
    initial.append((1 << 0) | (1 << 1) | (1 << 3) | (1 << 4))  # {0,1,3,4}
    initial.append((1 << 0) | (1 << 2) | (1 << 3) | (1 << 4))  # {0,2,3,4}
    initial.append((1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4))  # {0,1,2,3,4}
    initial.append((1 << n) - 1)  # full set

    # Ensure distinct and exactly 40
    seen = set()
    selected = []
    for mask in initial:
        if mask not in seen:
            seen.add(mask)
            selected.append(mask)
    # If less than 40, add more structured sets
    if len(selected) < m:
        # Add more pairs and triples
        for i in range(1, n):
            for j in range(i + 1, n):
                if (1 << i) | (1 << j) not in seen:
                    seen.add((1 << i) | (1 << j))
                    selected.append((1 << i) | (1 << j))
                    if len(selected) == m:
                        break
            if len(selected) == m:
                break

    selected = selected[:m]

    # Precompute membership as a set for O(1) lookup
    sel_set = set(selected)

    def score_of(sel):
        s = 0
        for x in sel:
            for y in comp[x]:
                if y in sel:
                    s += 1
        return s // 2

    current = score_of(selected)
    best = current
    best_sel = list(selected)

    # Local search: 1-swap and 2-swap
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 1-swap
        for idx in range(m):
            if time.time() >= deadline:
                break
            old = selected[idx]
            loss = 0
            for y in comp[old]:
                if y in sel_set and y != old:
                    loss += 1
            best_gain = -1
            best_new = -1
            for new in range(total):
                if new in sel_set:
                    continue
                gain = 0
                for y in comp[new]:
                    if y in sel_set and y != old:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_new = new
            if best_new >= 0 and best_gain > loss:
                sel_set.remove(old)
                sel_set.add(best_new)
                selected[idx] = best_new
                current += best_gain - loss
                if current > best:
                    best = current
                    best_sel = list(selected)
                improved = True

        # 2-swap: try swapping two elements
        if time.time() < deadline:
            for i1 in range(m):
                if time.time() >= deadline:
                    break
                for i2 in range(i1 + 1, m):
                    if time.time() >= deadline:
                        break
                    old1, old2 = selected[i1], selected[i2]
                    # Compute loss of removing both
                    loss = 0
                    for y in comp[old1]:
                        if y in sel_set and y != old1 and y != old2:
                            loss += 1
                    for y in comp[old2]:
                        if y in sel_set and y != old1 and y != old2:
                            loss += 1
                    # Also check if old1 and old2 are comparable to each other
                    if (old1 & old2) == old1 or (old1 & old2) == old2:
                        loss += 1

                    best_gain = -1
                    best_new1, best_new2 = -1, -1
                    # Try all pairs of new sets (this is expensive, so limit)
                    # Use a heuristic: try top candidates
                    candidates = []
                    for new in range(total):
                        if new in sel_set:
                            continue
                        gain = 0
                        for y in comp[new]:
                            if y in sel_set and y != old1 and y != old2:
                                gain += 1
                        candidates.append((gain, new))
                    candidates.sort(reverse=True)
                    candidates = candidates[:50]  # Limit to top 50

                    for c1 in candidates:
                        if time.time() >= deadline:
                            break
                        g1, n1 = c1
                        for c2 in candidates:
                            if c2[1] == c1[1]:
                                continue
                            g2, n2 = c2
                            # Check if n1 and n2 are comparable
                            extra = 0
                            if (n1 & n2) == n1 or (n1 & n2) == n2:
                                extra = 1
                            total_gain = g1 + g2 + extra
                            if total_gain > best_gain:
                                best_gain = total_gain
                                best_new1, best_new2 = n1, n2
                    if best_new1 >= 0 and best_gain > loss:
                        sel_set.remove(old1)
                        sel_set.remove(old2)
                        sel_set.add(best_new1)
                        sel_set.add(best_new2)
                        selected[i1] = best_new1
                        selected[i2] = best_new2
                        current += best_gain - loss
                        if current > best:
                            best = current
                            best_sel = list(selected)
                        improved = True

    # Convert to lists
    result = []
    for mask in best_sel:
        result.append([int(b) for b in range(n) if (mask >> b) & 1])
    return result
