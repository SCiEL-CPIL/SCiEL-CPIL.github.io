import time

def construct():
    deadline = time.time() + 85
    
    n = 128
    max_sum = 2 * n
    S = set(range(1, n + 1))
    mem = [False] * (max_sum + 1)
    for x in S:
        mem[x] = True
    
    def find_bad_triples():
        bad = []
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i + 1, m):
                b = elems[j]
                ab = a + b
                if ab > max_sum or not mem[ab]:
                    continue
                for k in range(j + 1, m):
                    c = elems[k]
                    ac = a + c
                    if ac > max_sum or not mem[ac]:
                        continue
                    bc = b + c
                    if bc <= max_sum and mem[bc]:
                        bad.append((a, b, c))
        return bad
    
    def count_involvement(bad_triples):
        counts = {}
        for (a, b, c) in bad_triples:
            for x in (a, b, c, a + b, a + c, b + c):
                if x in S:
                    counts[x] = counts.get(x, 0) + 1
        return counts
    
    bad = find_bad_triples()
    
    while bad and time.time() < deadline:
        counts = count_involvement(bad)
        if not counts:
            break
        # Remove top 3 most involved elements per batch
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        for x, _ in sorted_counts[:3]:
            if x in S:
                S.discard(x)
                mem[x] = False
        bad = find_bad_triples()
    
    # Final verification pass - fix any remaining violations
    changed = True
    while changed and time.time() < deadline:
        changed = False
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i + 1, m):
                b = elems[j]
                ab = a + b
                if ab > max_sum or not mem[ab]:
                    continue
                for k in range(j + 1, m):
                    c = elems[k]
                    ac = a + c
                    if ac > max_sum or not mem[ac]:
                        continue
                    bc = b + c
                    if bc <= max_sum and mem[bc]:
                        # Remove one of the six elements
                        # Prefer removing sums (they may be involved in fewer other constraints)
                        for x in (ab, ac, bc, a, b, c):
                            if x in S:
                                S.discard(x)
                                mem[x] = False
                                changed = True
                                break
                        break
                if changed:
                    break
            if changed:
                break
    
    return list(sorted(S))
