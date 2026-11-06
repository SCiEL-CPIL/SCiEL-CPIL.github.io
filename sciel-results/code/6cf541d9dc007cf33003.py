import itertools
import random
import time

N = 16
NUM_POINTS = N * N
POINTS = list(itertools.product(range(N), repeat=2))
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# For each pair (p, a), precompute the bitmask of q's such that (q, p, a) is isosceles.
# Then a set S covers q iff there exists a pair {p,a} ⊆ S whose mask includes q.
# Represent masks as Python ints (bit i set iff point i is covered).

FULL_MASK = (1 << NUM_POINTS) - 1

# pair_mask[p][a] = bitmask of q such that (q,p,a) is isosceles, p != a
pair_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

def _dist2(p, q):
    return (p[0]-q[0])**2 + (p[1]-q[1])**2

# Build pair masks: for each q, find isosceles pairs.
# For each q, for each p, for each a: if isosceles, set bit q in pair_mask[p][a] and pair_mask[a][p].
start_t = time.time()
for q_idx, q in enumerate(POINTS):
    for p_idx, p in enumerate(POINTS):
        if p_idx == q_idx:
            continue
        for a_idx, a in enumerate(POINTS):
            if a_idx == q_idx or a_idx == p_idx:
                continue
            d_qp = _dist2(q, p)
            d_qa = _dist2(q, a)
            d_pa = _dist2(p, a)
            if d_qp == d_qa or d_qp == d_pa or d_qa == d_pa:
                pair_mask[p_idx][a_idx] |= (1 << q_idx)
                pair_mask[a_idx][p_idx] |= (1 << q_idx)
# Symmetrize safety
for i in range(NUM_POINTS):
    for j in range(NUM_POINTS):
        if i != j:
            pair_mask[i][j] = pair_mask[i][j] | pair_mask[j][i]
# Note: pair_mask[i][i] unused.

# For fast "can we add p to S" we need: for all q, is q covered by S ∪ {p}?
# q is covered by S∪{p} iff (q covered by S) OR (exists a in S with pair_mask[p][a] has bit q).
# We maintain cover_mask = union over pairs in S of pair_mask[p][a].
# When adding p: new_cover = cover_mask | (union over a in S of pair_mask[p][a])
# Precompute for each p: add_mask[p][a] = pair_mask[p][a] (same thing).

def _is_maximal(S, cover_mask):
    # S is maximal iff cover_mask == FULL_MASK (every q covered by some pair in S)
    return (cover_mask & FULL_MASK) == FULL_MASK

def _covers_all(cover_mask):
    return (cover_mask & FULL_MASK) == FULL_MASK

def _pair_union(S, p):
    # union of pair_mask[p][a] for a in S
    u = 0
    for a in S:
        u |= pair_mask[p][a]
    return u

def construct():
    deadline = time.time() + 320.0  # safety margin under 330s
    best_set = None
    best_size = NUM_POINTS + 1

    def _time_ok():
        return time.time() < deadline

    # Greedy set-cover construction:
    # Start with empty S, cover_mask=0.
    # Repeatedly add the point p that maximizes newly-covered q's (|cover_mask_new \ cover_mask|),
    # until all q are covered. This gives a maximal set (since all covered).
    # Then local search to minimize size.

    def greedy_cover(seed_val):
        random.seed(seed_val)
        S = set()
        cover_mask = 0
        # Precompute for speed: for each p, the set of q it can cover with each a.
        # We'll do lazy: each iteration compute gain for each candidate p not in S.
        remaining = list(range(NUM_POINTS))
        random.shuffle(remaining)
        while not _covers_all(cover_mask):
            if not _time_ok():
                break
            best_p = None
            best_gain = -1
            best_new_mask = 0
            for p in remaining:
                if p in S:
                    continue
                # new coverage from adding p
                add_u = 0
                for a in S:
                    add_u |= pair_mask[p][a]
                new_mask = cover_mask | add_u
                gain = bin(new_mask & ~cover_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_new_mask = new_mask
            if best_p is None:
                break
            S.add(best_p)
            cover_mask = best_new_mask
        return S, cover_mask

    def local_search_minimize(S, cover_mask, seed_val):
        random.seed(seed_val)
        S = set(S)
        cover_mask = cover_mask
        improved = True
        while improved and _time_ok():
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            for p in s_list:
                if p not in S:
                    continue
                # Try removing p: new_cover = union over pairs {a,b} in S\{p}
                # = cover_mask minus contributions only from pairs involving p.
                # But pairs not involving p still contribute. So:
                # new_cover = (cover_mask with pairs not involving p) 
                # We recompute: for each a in S\{p}, for each b in S\{p}, a<b: pair_mask[a][b]
                # That's O(k^2) where k=|S|. For k~20, that's 400 ops. Fine.
                new_cover = 0
                Sp = S - {p}
                sp_list = list(Sp)
                for i in range(len(sp_list)):
                    a = sp_list[i]
                    for j in range(i+1, len(sp_list)):
                        b = sp_list[j]
                        new_cover |= pair_mask[a][b]
                if _covers_all(new_cover):
                    S = Sp
                    cover_mask = new_cover
                    improved = True
                    break
        return S, cover_mask

    def swap_search(S, cover_mask, seed_val, iters=200):
        # Try swapping p in S with r not in S if it keeps coverage and reduces... size same.
        # Only useful if it enables later removals. We do: for each p in S, r not in S,
        # if S\{p}∪{r} still covers all, do it (random order), then try to remove.
        random.seed(seed_val)
        S = set(S)
        cover_mask = cover_mask
        for _ in range(iters):
            if not _time_ok():
                break
            s_list = list(S)
            random.shuffle(s_list)
            notS = [i for i in range(NUM_POINTS) if i not in S]
            random.shuffle(notS)
            moved = False
            for p in s_list:
                if p not in S:
                    continue
                for r in notS:
                    if r in S:
                        continue
                    # test S' = S - {p} + {r}
                    Sp = S - {p}
                    Snew = Sp | {r}
                    # compute cover of Snew
                    new_cover = 0
                    sl = list(Snew)
                    for i in range(len(sl)):
                        a = sl[i]
                        for j in range(i+1, len(sl)):
                            b = sl[j]
                            new_cover |= pair_mask[a][b]
                    if _covers_all(new_cover):
                        S = Snew
                        cover_mask = new_cover
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break
        return S, cover_mask

    # Try multiple strategies
    restart = 0
    while _time_ok():
        # Strategy 1: greedy cover then minimize
        S, cm = greedy_cover(restart)
        if _covers_all(cm):
            S2, cm2 = local_search_minimize(S, cm, restart + 1000)
            if len(S2) < best_size:
                best_size = len(S2)
                best_set = set(S2)
        restart += 1
        if not _time_ok():
            break

        # Strategy 2: random initial maximal set (pick points until maximal), then minimize
        random.seed(restart + 5000)
        S = set()
        cm = 0
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        for p in order:
            if p in S:
                continue
            add_u = 0
            for a in S:
                add_u |= pair_mask[p][a]
            new_cm = cm | add_u
            if new_cm != cm or not _covers_all(cm):
                # adding p changes coverage; always add if it helps, else skip
                if new_cm != cm:
                    S.add(p)
                    cm = new_cm
        if not _covers_all(cm):
            # force maximal: add arbitrary points until covered
            for p in order:
                if p in S:
                    continue
                add_u = 0
                for a in S:
                    add_u |= pair_mask[p][a]
                new_cm = cm | add_u
                S.add(p)
                cm = new_cm
                if _covers_all(cm):
                    break
        if _covers_all(cm):
            S2, cm2 = local_search_minimize(S, cm, restart + 2000)
            if len(S2) < best_size:
                best_size = len(S2)
                best_set = set(S2)
        restart += 1

    if best_set is None:
        # Fallback: return all points (always maximal and no-isosceles? No, all points has isosceles triples!)
        # Fallback: greedy cover only
        S, cm = greedy_cover(0)
        best_set = S

    return [POINTS[i] for i in sorted(best_set)]


if __name__ == "__main__":
    s = construct()
    print(len(s))
