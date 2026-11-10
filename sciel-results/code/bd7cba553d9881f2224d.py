"""Linear triple systems on 16 points with small independence number."""
import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    N = 16
    FULL_MASK = (1 << N) - 1
    ALL_TRIPLES = list(itertools.combinations(range(N), 3))
    num_triples = len(ALL_TRIPLES)

    # Precompute triple bitmasks
    triple_masks = {}
    for t in ALL_TRIPLES:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[t] = mask

    # Precompute pair-to-triple-index mapping for conflict detection
    # pair_to_indices[pair] = list of triple indices containing that pair
    pair_to_indices = {}
    for i, t in enumerate(ALL_TRIPLES):
        for k in range(3):
            for l in range(k + 1, 3):
                p = (min(t[k], t[l]), max(t[k], t[l]))
                if p not in pair_to_indices:
                    pair_to_indices[p] = []
                pair_to_indices[p].append(i)

    # Build conflict sets: conflict[i] = set of triple indices j that share >= 2 points with i
    conflict = [set() for _ in range(num_triples)]
    for p, tlist in pair_to_indices.items():
        for idx_a in range(len(tlist)):
            for idx_b in range(idx_a + 1, len(tlist)):
                a = tlist[idx_a]
                b = tlist[idx_b]
                conflict[a].add(b)
                conflict[b].add(a)

    # Precompute for each triple index, the set of triple indices it conflicts with
    # Also precompute a quick check: for a set of triple indices, can we add triple j?
    # We'll use a bitmask approach: for each triple i, store the conflict set as a frozenset for fast lookup

    def independence_number_fast(triple_indices):
        """
        Find the largest subset of {0..15} containing no triple from triple_indices.
        triple_indices: list of indices into ALL_TRIPLES
        Uses superset marking with bytearray.
        """
        if not triple_indices:
            return N

        bad = bytearray(1 << N)
        for ti in triple_indices:
            m = (1 << ALL_TRIPLES[ti][0]) | (1 << ALL_TRIPLES[ti][1]) | (1 << ALL_TRIPLES[ti][2])
            free = FULL_MASK ^ m
            sub = free
            while True:
                bad[m | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & free

        best = 0
        for S in range(1 << N):
            if not bad[S]:
                pc = S.bit_count()
                if pc > best:
                    best = pc
        return best

    def independence_number_with_set(triple_indices):
        """Same as above but also returns the actual independent set."""
        if not triple_indices:
            return N, set(range(N))

        bad = bytearray(1 << N)
        for ti in triple_indices:
            m = (1 << ALL_TRIPLES[ti][0]) | (1 << ALL_TRIPLES[ti][1]) | (1 << ALL_TRIPLES[ti][2])
            free = FULL_MASK ^ m
            sub = free
            while True:
                bad[m | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & free

        best = 0
        best_set = set()
        for S in range(1 << N):
            if not bad[S]:
                pc = S.bit_count()
                if pc > best:
                    best = pc
                    best_set = set(i for i in range(N) if S & (1 << i))
        return best, best_set

    def can_add(triple_idx, current_triple_indices):
        """Check if triple_idx can be added to current_triple_indices without violating linearity."""
        for ci in current_triple_indices:
            if ci in conflict[triple_idx]:
                return False
        return True

    def find_valid_triples_to_add(current_triple_indices):
        """Find all triple indices that can be added to the current set."""
        valid = []
        current_set = set(current_triple_indices)
        for i in range(num_triples):
            if i not in current_set:
                if can_add(i, current_triple_indices):
                    valid.append(i)
        return valid

    def greedy_packing(seed_offset=0):
        """Build a dense linear triple system via randomized greedy packing."""
        rng = random.Random(42 + seed_offset)
        used_pairs = set()
        triples = []

        candidates = list(range(num_triples))
        rng.shuffle(candidates)

        for idx in candidates:
            t = ALL_TRIPLES[idx]
            pairs = [
                (min(t[0], t[1]), max(t[0], t[1])),
                (min(t[0], t[2]), max(t[0], t[2])),
                (min(t[1], t[2]), max(t[1], t[2]))
            ]
            can_add = True
            for p in pairs:
                if p in used_pairs:
                    can_add = False
                    break
            if can_add:
                for p in pairs:
                    used_pairs.add(p)
                triples.append(idx)

        return triples

    def local_search_sa(initial_triple_indices, max_time=60):
        """
        Simulated Annealing local search.
        State: list of triple indices (linear).
        Moves:
          1. Add a random valid triple.
          2. Remove a random triple.
          3. Swap: remove a random triple, add a different random valid triple.
        Energy: independence_number.
        We want to minimize energy.
        """
        rng = random.Random(1000)
        current = initial_triple_indices[:]
        current_score = independence_number_fast(current)

        best = current[:]
        best_score = current_score

        timeout = time.time() + max_time

        T0 = 5.0
        Tend = 0.01

        steps = 0
        no_improve_steps = 0
        max_no_improve = 500

        while time.time() < timeout:
            steps += 1
            if no_improve_steps > max_no_improve:
                break

            T = max(Tend, T0 * (0.9995 ** steps))

            move_type = rng.random()
            new_triples = current[:]
            valid_move = False

            if move_type < 0.4:
                # Add a random triple
                valid_candidates = find_valid_triples_to_add(current)
                if valid_candidates:
                    idx = rng.choice(valid_candidates)
                    new_triples.append(idx)
                    valid_move = True
                else:
                    # Try a few random ones
                    for _ in range(5):
                        idx = rng.randint(0, num_triples - 1)
                        if idx not in set(current) and can_add(idx, current):
                            new_triples.append(idx)
                            valid_move = True
                            break

            elif move_type < 0.7:
                # Remove a random triple
                if len(current) > 0:
                    idx = rng.randint(0, len(current) - 1)
                    new_triples.pop(idx)
                    valid_move = True
            else:
                # Swap: remove one, add another
                if len(current) > 0:
                    idx = rng.randint(0, len(current) - 1)
                    removed = new_triples.pop(idx)
                    added = False
                    valid_candidates = find_valid_triples_to_add(new_triples)
                    if valid_candidates:
                        # Prefer adding a triple that helps
                        nidx = rng.choice(valid_candidates)
                        new_triples.append(nidx)
                        added = True
                        valid_move = True
                    if not added:
                        new_triples.append(removed)
                        valid_move = False

            if not valid_move:
                continue

            new_score = independence_number_fast(new_triples)
            delta = new_score - current_score

            if delta < 0 or (T > 0 and rng.random() < 2.71828 ** (-delta / T)):
                current = new_triples
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1

        return best, best_score

    def targeted_search(initial_triple_indices, max_time=60):
        """
        Targeted search: find large independent sets and add triples to break them.
        """
        rng = random.Random(2000)
        best = initial_triple_indices[:]
        best_score = independence_number_fast(best)

        current = initial_triple_indices[:]
        current_score = best_score

        timeout = time.time() + max_time

        for iteration in range(10000):
            if time.time() > timeout:
                break

            alpha, iset = independence_number_with_set(current)
            if alpha < 4:
                break

            # Try to add a triple that is a subset of the independent set
            iset_list = list(iset)
            candidates = list(itertools.combinations(iset_list, 3))
            rng.shuffle(candidates)

            added = False
            for new_triple in candidates:
                # Find the index of this triple
                ti = ALL_TRIPLES.index(new_triple) if new_triple in triple_masks else None
                if ti is None:
                    continue
                if can_add(ti, current):
                    new_current = current + [ti]
                    new_score = independence_number_fast(new_current)
                    if new_score < current_score:
                        current = new_current
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best = current[:]
                    added = True
                    break

            if not added:
                # Try removing a random triple and adding a better one
                if len(current) > 3:
                    idx = rng.randint(0, len(current) - 1)
                    removed = current.pop(idx)

                    for new_triple in candidates:
                        ti = ALL_TRIPLES.index(new_triple) if new_triple in triple_masks else None
                        if ti is None:
                            continue
                        if can_add(ti, current):
                            new_current = current + [ti]
                            new_score = independence_number_fast(new_current)
                            if new_score < current_score:
                                current = new_current
                                current_score = new_score
                                if new_score < best_score:
                                    best_score = new_score
                                    best = current[:]
                            break
                    else:
                        current.append(removed)

        return best, best_score

    # Phase 1: Generate initial systems with different seeds
    best_triples = None
    best_score = N + 1

    for seed_offset in range(30):
        if time.time() > deadline - 30:
            break
        initial = greedy_packing(seed_offset)
        if not initial:
            continue
        score = independence_number_fast(initial)
        if score < best_score:
            best_score = score
            best_triples = initial[:]

    if best_triples is None:
        # Fallback: 5 disjoint triples
        best_triples = [ALL_TRIPLES.index((0, 1, 2)),
                        ALL_TRIPLES.index((3, 4, 5)),
                        ALL_TRIPLES.index((6, 7, 8)),
                        ALL_TRIPLES.index((9, 10, 11)),
                        ALL_TRIPLES.index((12, 13, 14))]
        best_score = independence_number_fast(best_triples)

    # Phase 2: Local search (Simulated Annealing)
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        best_triples, best_score = local_search_sa(best_triples, max_time=min(remaining_time, 60))

    # Phase 3: Targeted local search
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        best_triples, best_score = targeted_search(best_triples, max_time=remaining_time)

    # Phase 4: Another round of SA with the best found so far
    remaining_time = deadline - time.time() - 3
    if remaining_time > 10:
        best_triples, best_score = local_search_sa(best_triples, max_time=min(remaining_time, 30))

    # Convert triple indices back to triples
    result = [list(ALL_TRIPLES[i]) for i in best_triples]

    # Verify linearity
    def is_linear(triples):
        for i in range(len(triples)):
            ti = set(triples[i])
            for j in range(i + 1, len(triples)):
                tj = set(triples[j])
                if len(ti & tj) > 1:
                    return False
        return True

    if not is_linear(result):
        # Fallback to a known valid system
        result = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]

    return result

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} triples")
    for t in result:
        print(t)
