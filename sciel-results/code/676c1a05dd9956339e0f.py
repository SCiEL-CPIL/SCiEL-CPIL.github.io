import time
import random

_start_time = time.time()
DEADLINE = _start_time + 80

N = 12
M = 40
TOTAL = 1 << N  # 4096

def construct():
    # Precompute comparability matrix
    # comp[i][j] = 1 if i and j are comparable (one is subset of the other)
    # We'll store as a list of bytearrays for speed
    comp = [bytearray(TOTAL) for _ in range(TOTAL)]
    for i in range(TOTAL):
        for j in range(i, TOTAL):
            if (i | j) == i or (i | j) == j:
                comp[i][j] = 1
                if i != j:
                    comp[j][i] = 1

    def count_pairs(family):
        cnt = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                cnt += comp[family[i]][family[j]]
        return cnt

    def get_degrees(family):
        deg = [0] * len(family)
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                c = comp[family[i]][family[j]]
                deg[i] += c
                deg[j] += c
        return deg

    # Greedy construction: build a dense chain-based family
    # Start with empty set and full set, then greedily add most comparable sets
    family = [0, TOTAL - 1]  # empty set and full set
    seen = set(family)

    # Precompute size of each mask for ordering
    sizes = [bin(m).count('1') for m in range(TOTAL)]

    # Greedy: repeatedly add the set that maximizes comparable pairs with current family
    for _ in range(M - 2):
        best_mask = -1
        best_score = -1
        for m in range(TOTAL):
            if m in seen:
                continue
            s = 0
            for f in family:
                s += comp[m][f]
            # Tie-break: prefer middle sizes for more flexibility
            if s > best_score or (s == best_score and abs(sizes[m] - N // 2) < abs(sizes[best_mask] - N // 2)):
                best_score = s
                best_mask = m
        family.append(best_mask)
        seen.add(best_mask)

    current_score = count_pairs(family)
    best_family = list(family)
    best_score = current_score

    # Local search: try swapping each set with a random one
    improved = True
    while improved:
        if time.time() > DEADLINE - 3:
            break
        improved = False
        deg = get_degrees(family)

        for i in range(M):
            if time.time() > DEADLINE - 3:
                break
            old = family[i]
            old_deg = deg[i]

            best_new = -1
            best_gain = 0

            # Sample candidates
            for _ in range(200):
                new_m = random.randrange(TOTAL)
                if new_m == old or new_m in seen:
                    continue
                # Degree of new_m in family without old
                new_deg = 0
                for j in range(M):
                    if j == i:
                        continue
                    new_deg += comp[new_m][family[j]]
                gain = new_deg - old_deg
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_m

            if best_new != -1:
                seen.remove(old)
                seen.add(best_new)
                family[i] = best_new
                current_score += best_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
                break

    # Additional multi-restart with different initializations if time allows
    if time.time() < DEADLINE - 10:
        for restart in range(3):
            if time.time() > DEADLINE - 5:
                break
            # Different greedy initialization
            fam = [0, TOTAL - 1]
            seenset = set(fam)
            for _ in range(M - 2):
                best_m = -1
                best_s = -1
                for m in range(TOTAL):
                    if m in seenset:
                        continue
                    s = 0
                    for f in fam:
                        s += comp[m][f]
                    if s > best_s or (s == best_s and random.random() < 0.1):
                        best_s = s
                        best_m = m
                fam.append(best_m)
                seenset.add(best_m)

            sc = count_pairs(fam)
            # Local search on this restart
            imp = True
            while imp:
                if time.time() > DEADLINE - 5:
                    break
                imp = False
                dg = get_degrees(fam)
                for i in range(M):
                    if time.time() > DEADLINE - 5:
                        break
                    old = fam[i]
                    old_d = dg[i]
                    bn = -1
                    bg = 0
                    for _ in range(150):
                        nm = random.randrange(TOTAL)
                        if nm == old or nm in seenset:
                            continue
                        nd = 0
                        for j in range(M):
                            if j == i:
                                continue
                            nd += comp[nm][fam[j]]
                        g = nd - old_d
                        if g > bg:
                            bg = g
                            bn = nm
                    if bn != -1:
                        seenset.remove(old)
                        seenset.add(bn)
                        fam[i] = bn
                        sc += bg
                        imp = True
                        if sc > best_score:
                            best_score = sc
                            best_family = list(fam)
                        break

    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(N) if (mask >> i) & 1]
        result.append(lst)

    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i + 1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1 - t0:.2f}s, Comparables: {cnt}")
