import random
import time

N = 16
FULL_MASK = (1 << N) - 1


def _triple_mask(i, j, k):
    return (1 << i) | (1 << j) | (1 << k)


def _is_linear_with(triple, existing_triples):
    a, b, c = triple
    for (x, y, z) in existing_triples:
        common = 0
        if a == x or a == y or a == z:
            common += 1
        if b == x or b == y or b == z:
            common += 1
        if c == x or c == y or c == z:
            common += 1
        if common > 1:
            return False
    return True


def _build_covered(triples_masks):
    covered = bytearray(1 << N)
    for t_mask in triples_masks:
        rest = FULL_MASK ^ t_mask
        sub = rest
        while True:
            covered[t_mask | sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & rest
    return covered


def _find_max_independent_set(covered):
    best_set = 0
    best_size = 0
    for S in range(1 << N):
        if not covered[S]:
            c = S.bit_count()
            if c > best_size:
                best_size = c
                best_set = S
    return best_set, best_size


def _build_uncovered(triples_masks):
    """uncovered[S]=1 iff S contains NO triple from triples_masks."""
    uncovered = bytearray(1 << N)
    uncovered[0] = 1
    for t_mask in triples_masks:
        rest = FULL_MASK ^ t_mask
        sub = rest
        while True:
            uncovered[t_mask | sub] = 0
            if sub == 0:
                break
            sub = (sub - 1) & rest
    return uncovered


def _is_cover5(triples_masks):
    """True iff no 5-subset is uncovered (i.e. alpha <= 5)."""
    for t in triples_masks:
        rest = FULL_MASK ^ t
        sub = rest
        while True:
            s = t | sub
            if s.bit_count() == 5:
                return False
            if sub == 0:
                break
            sub = (sub - 1) & rest
    return True


def _find_max_uncovered(uncovered):
    best = 0
    best_c = 0
    for S in range(1 << N):
        if uncovered[S]:
            c = S.bit_count()
            if c > best_c:
                best_c = c
                best = S
    return best, best_c


def _greedy_init(all_triples, pair_to_triples, rng):
    triples = []
    shuffled = all_triples[:]
    rng.shuffle(shuffled)
    for t in shuffled:
        if _is_linear_with(t, triples):
            triples.append(t)
    return triples


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150 s

    rng = random.Random(123)

    all_triples = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                all_triples.append((i, j, k))

    pair_to_triples = {}
    for idx, (i, j, k) in enumerate(all_triples):
        for p1, p2 in [(i, j), (i, k), (j, k)]:
            key = (min(p1, p2), max(p1, p2))
            pair_to_triples.setdefault(key, []).append(idx)

    cur_triples = _greedy_init(all_triples, pair_to_triples, rng)
    cur_masks = [_triple_mask(*t) for t in cur_triples]

    best_alpha = 1 << 30
    best_triples = list(cur_triples)

    # Precompute the list of 5-subsets for the alpha<=5 test.
    five_sets = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    for e in range(d + 1, N):
                        five_sets.append((1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e))

    while time.time() < deadline:
        # Exact independence number via superset marking (robust ground truth).
        covered = _build_covered(cur_masks)
        max_ind_set, cur_alpha = _find_max_independent_set(covered)

        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best_triples = list(cur_triples)
            if best_alpha <= 5:
                break  # optimal for this family

        # Cheap incremental maintenance of the uncovered set for fast moves.
        uncovered = _build_uncovered(cur_masks)

        ind_points = [p for p in range(N) if (max_ind_set >> p) & 1]

        added = False
        if len(ind_points) >= 2:
            ind_pairs = []
            for i in range(len(ind_points)):
                for j in range(i + 1, len(ind_points)):
                    p1, p2 = ind_points[i], ind_points[j]
                    ind_pairs.append(((min(p1, p2), max(p1, p2)), p1, p2))
            rng.shuffle(ind_pairs)
            for key, p1, p2 in ind_pairs:
                candidates = pair_to_triples.get(key, [])
                rng.shuffle(candidates)
                for cand_idx in candidates:
                    t = all_triples[cand_idx]
                    if t in cur_triples:
                        continue
                    if _is_linear_with(t, cur_triples):
                        cur_triples.append(t)
                        cur_masks.append(_triple_mask(*t))
                        added = True
                        break
                if added:
                    break

        if not added and cur_triples:
            rem_idx = rng.randrange(len(cur_triples))
            rem_t = cur_triples[rem_idx]
            found_add = False
            for _ in range(30):
                add_idx = rng.randrange(len(all_triples))
                add_t = all_triples[add_idx]
                if add_t == rem_t:
                    continue
                ok = True
                for j, e in enumerate(cur_triples):
                    if j == rem_idx:
                        continue
                    if not _is_linear_with(add_t, [e]):
                        ok = False
                        break
                if ok:
                    cur_triples[rem_idx] = add_t
                    cur_masks[rem_idx] = _triple_mask(*add_t)
                    found_add = True
                    break
            if not found_add:
                cur_triples[rem_idx] = rem_t
                cur_masks[rem_idx] = _triple_mask(*rem_t)

        if not added:
            r = rng.random()
            if r < 0.3 and cur_triples:
                rem_idx = rng.randrange(len(cur_triples))
                cur_triples.pop(rem_idx)
                cur_masks.pop(rem_idx)
            elif r < 0.6:
                for _ in range(15):
                    add_idx = rng.randrange(len(all_triples))
                    add_t = all_triples[add_idx]
                    if add_t not in cur_triples and _is_linear_with(add_t, cur_triples):
                        cur_triples.append(add_t)
                        cur_masks.append(_triple_mask(*add_t))
                        break

    return [list(t) for t in best_triples]
