import time
import random

N_POINTS = 16
N_MASKS = 1 << N_POINTS
ALL_MASKS = range(N_MASKS)

POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

TRIPLES = []
for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            TRIPLES.append((1 << a) | (1 << b) | (1 << c))

SUPERSETS = [None] * len(TRIPLES)
for i, t in enumerate(TRIPLES):
    comp = ((1 << N_POINTS) - 1) ^ t
    sups = []
    sub = comp
    while True:
        sups.append(t | sub)
        if sub == 0:
            break
        sub = (sub - 1) & comp
    SUPERSETS[i] = sups

MASK_TO_INDEX = {t: i for i, t in enumerate(TRIPLES)}
SORTED_MASKS = sorted(ALL_MASKS, key=lambda x: POP_COUNT[x], reverse=True)


def construct():
    deadline = time.time() + 110.0

    def calc_alpha_and_set(triple_masks):
        valid = [True] * N_MASKS
        for t in triple_masks:
            idx = MASK_TO_INDEX[t]
            for s in SUPERSETS[idx]:
                valid[s] = False
        for s in SORTED_MASKS:
            if valid[s]:
                return POP_COUNT[s], s
        return 0, 0

    def can_add(triple_set, new_t):
        for t in triple_set:
            if (t & new_t).bit_count() >= 2:
                return False
        return True

    _random = random.Random(42)

    # Initialize with greedy maximal system
    triples_list = []
    triples_set = set()
    available = TRIPLES[:]
    _random.shuffle(available)
    for t in available:
        if can_add(triples_set, t):
            triples_list.append(t)
            triples_set.add(t)

    current_alpha, current_set = calc_alpha_and_set(triples_list)
    best_alpha = current_alpha
    best_triples = triples_list[:]
    best_set = set(triples_list)

    # Phase 1: Directed local search (break current independent set)
    while time.time() < deadline - 30.0:
        # Find triples that are subsets of the current independent set
        candidates_in_set = []
        for t in TRIPLES:
            if (t & ~current_set) == 0:
                if t not in triples_set:
                    candidates_in_set.append(t)

        moves_made = False

        # Try adding candidates directly
        _random.shuffle(candidates_in_set)
        for t in candidates_in_set:
            if can_add(triples_set, t):
                new_list = triples_list + [t]
                new_set = triples_set | {t}
                new_alpha, new_ind_set = calc_alpha_and_set(new_list)
                if new_alpha < current_alpha:
                    triples_list = new_list
                    triples_set = new_set
                    current_alpha = new_alpha
                    current_set = new_ind_set
                    moves_made = True
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = triples_list[:]
                        best_set = triples_set.copy()
                    break

        if moves_made:
            continue

        # Try swaps: remove a conflicting triple, add a candidate
        if candidates_in_set:
            t_in = _random.choice(candidates_in_set)
        else:
            # No triple fully inside independent set; try random exploration
            t_in = _random.choice(TRIPLES)
            if t_in in triples_set:
                # Try a random swap instead
                if triples_list:
                    t_out = _random.choice(triples_list)
                    new_list = [t for t in triples_list if t != t_out]
                    new_set = triples_set - {t_out}
                    t_add = _random.choice(TRIPLES)
                    if t_add not in new_set and can_add(new_set, t_add):
                        new_list.append(t_add)
                        new_set.add(t_add)
                        new_alpha, new_ind_set = calc_alpha_and_set(new_list)
                        if new_alpha <= current_alpha:
                            triples_list = new_list
                            triples_set = new_set
                            current_alpha = new_alpha
                            current_set = new_ind_set
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_triples = triples_list[:]
                                best_set = triples_set.copy()
                continue

        # Find triples conflicting with t_in
        conflicting = [t for t in triples_list if (t & t_in).bit_count() >= 2]

        if conflicting:
            # Try removing each conflicting triple and adding t_in
            _random.shuffle(conflicting)
            for t_out in conflicting:
                new_list = [t for t in triples_list if t != t_out]
                new_set = triples_set - {t_out}
                if can_add(new_set, t_in):
                    new_list.append(t_in)
                    new_set.add(t_in)
                    new_alpha, new_ind_set = calc_alpha_and_set(new_list)
                    if new_alpha <= current_alpha:
                        triples_list = new_list
                        triples_set = new_set
                        current_alpha = new_alpha
                        current_set = new_ind_set
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = triples_list[:]
                            best_set = triples_set.copy()
                        moves_made = True
                        break

        if not moves_made:
            # Random exploration: try a random swap to escape local optimum
            if triples_list and _random.random() < 0.5:
                t_out = _random.choice(triples_list)
                new_list = [t for t in triples_list if t != t_out]
                new_set = triples_set - {t_out}
                t_add = _random.choice(TRIPLES)
                if t_add not in new_set and can_add(new_set, t_add):
                    new_list.append(t_add)
                    new_set.add(t_add)
                    new_alpha, new_ind_set = calc_alpha_and_set(new_list)
                    if new_alpha <= current_alpha:
                        triples_list = new_list
                        triples_set = new_set
                        current_alpha = new_alpha
                        current_set = new_ind_set
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = triples_list[:]
                            best_set = triples_set.copy()

    # Phase 2: Final polish - try to squeeze out one more improvement
    while time.time() < deadline:
        candidates_in_set = []
        for t in TRIPLES:
            if (t & ~current_set) == 0:
                if t not in triples_set:
                    candidates_in_set.append(t)

        _random.shuffle(candidates_in_set)
        improved = False
        for t in candidates_in_set:
            if can_add(triples_set, t):
                new_list = triples_list + [t]
                new_set = triples_set | {t}
                new_alpha, new_ind_set = calc_alpha_and_set(new_list)
                if new_alpha < current_alpha:
                    triples_list = new_list
                    triples_set = new_set
                    current_alpha = new_alpha
                    current_set = new_ind_set
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = triples_list[:]
                        best_set = triples_set.copy()
                    improved = True
                    break

        if improved:
            continue

        # Try swaps in polish phase
        if candidates_in_set:
            t_in = _random.choice(candidates_in_set)
            conflicting = [t for t in triples_list if (t & t_in).bit_count() >= 2]
            _random.shuffle(conflicting)
            for t_out in conflicting:
                new_list = [t for t in triples_list if t != t_out]
                new_set = triples_set - {t_out}
                if can_add(new_set, t_in):
                    new_list.append(t_in)
                    new_set.add(t_in)
                    new_alpha, new_ind_set = calc_alpha_and_set(new_list)
                    if new_alpha < current_alpha:
                        triples_list = new_list
                        triples_set = new_set
                        current_alpha = new_alpha
                        current_set = new_ind_set
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = triples_list[:]
                            best_set = triples_set.copy()
                        break

    result = []
    for mask in best_triples:
        pts = []
        for i in range(N_POINTS):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)

    return result
