import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    N = 16
    ALL = (1 << N) - 1

    # Precompute all C(16,3)=560 triples as masks
    all_triple_masks = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                all_triple_masks.append((1 << a) | (1 << b) | (1 << c))
    T = len(all_triple_masks)  # 560

    # Point -> bitmask of triple indices containing it
    point_hits = [0] * N
    for i, m in enumerate(all_triple_masks):
        x = m
        while x:
            low = x & (-x)
            p = low.bit_length() - 1
            point_hits[p] |= (1 << i)
            x ^= low

    # Upper bound on number of triples in a linear system (packing bound):
    # each triple uses 3 pairs, each pair in at most 1 triple, C(16,2)=120 pairs
    max_triples = 120 // 3  # 40

    def compute_alpha(triple_set):
        """Independence number = max subset S with no triple fully in S.
        Equivalent to 16 - (minimum vertex cover of the hypergraph).
        Fast bitmask B&B with memoization + upper bound."""
        n = len(triple_set)
        if n == 0:
            return 16

        # Precompute for each point which triples it hits (restricted to this set)
        # Represent each triple by a 3-bit index into local array
        local_hits = [0] * N
        for i, m in enumerate(triple_set):
            x = m
            while x:
                low = x & (-x)
                p = low.bit_length() - 1
                local_hits[p] |= (1 << i)
                x ^= low

        all_mask = (1 << n) - 1

        # Sort points by degree descending for better pruning
        order = sorted(range(N), key=lambda p: bin(local_hits[p]).count('1'), reverse=True)
        ordered_hits = [local_hits[p] for p in order]

        # Precompute suffix OR: suffix_or[i] = OR of ordered_hits[i:]
        suffix_or = [0] * (N + 1)
        for i in range(N - 1, -1, -1):
            suffix_or[i] = suffix_or[i + 1] | ordered_hits[i]

        best = [N]  # min vertex cover size, worst = N
        memo = {}

        def popcount(x):
            return bin(x).count('1')

        def bb(covered, idx):
            # covered: bitmask of triples already covered
            if covered == all_mask:
                used = popcount(covered)  # not accurate; track used separately
                return used
            if idx == N:
                return N + 1  # infeasible
            # Upper bound: if we add all remaining points, what's min possible?
            # Actually we want to know if we can beat best[0]
            # Current used count:
            used = popcount(covered)  # WRONG - covered is triple mask not point mask
            return N + 1

        # Redo: track used points count explicitly
        memo2 = {}

        def bb2(covered, idx, used):
            if used >= best[0]:
                return
            if covered == all_mask:
                best[0] = used
                return
            if idx == N:
                return
            key = (covered, idx)
            if key in memo2:
                if memo2[key] <= used:
                    return
            else:
                memo2[key] = used

            h = ordered_hits[idx]
            # Option 1: include point idx
            bb2(covered | h, idx + 1, used + 1)
            # Option 2: exclude point idx
            uncovered = all_mask & ~covered
            if uncovered:
                rem = suffix_or[idx + 1]
                if (uncovered & ~rem) != 0:
                    return
            bb2(covered, idx + 1, used)

        bb2(0, 0, 0)
        return N - best[0]

    def is_valid_addition(mask, existing_masks):
        for m in existing_masks:
            inter = mask & m
            if inter and (inter & (inter - 1)):
                return False
        return True

    # --- Phase 1: Greedy construction maximizing triple count ---
    best_alpha = 17
    best_triples = []

    def greedy_build(seed):
        rng = random.Random(seed)
        chosen = []
        chosen_masks = []
        order = list(range(T))
        rng.shuffle(order)
        for idx in order:
            m = all_triple_masks[idx]
            if is_valid_addition(m, chosen_masks):
                chosen.append(m)
                chosen_masks.append(m)
        return chosen

    # Build several greedy solutions, keep the one with max triples / min alpha
    for seed in range(30):
        if time.time() > deadline - 30:
            break
        chosen = greedy_build(seed)
        if not chosen:
            continue
        a = compute_alpha(chosen)
        # Prefer more triples, then lower alpha
        if a < best_alpha or (a == best_alpha and len(chosen) > len(best_triples)):
            best_alpha = a
            best_triples = chosen[:]

    # --- Phase 2: Local search (swap-based) to reduce alpha ---
    current = best_triples[:]
    current_alpha = best_alpha

    # Build a fast validity structure: for each pair of points, which chosen triple
    # contains them (0 = none, else index+1). When removing a triple, we can quickly
    # find candidate new triples that become valid.
    def build_pair_map(triples):
        pair_map = {}  # (a,b) -> index
        for i, m in enumerate(triples):
            pts = []
            x = m
            while x:
                low = x & (-x)
                p = low.bit_length() - 1
                pts.append(p)
                x ^= low
            for j in range(3):
                for k in range(j + 1, 3):
                    pair_map[(pts[j], pts[k])] = i
        return pair_map

    def local_search(triples, alpha, time_limit):
        triples = triples[:]
        alpha = alpha
        pair_map = build_pair_map(triples)
        t_end = time.time() + time_limit
        steps = 0
        while time.time() < t_end:
            steps += 1
            if not triples:
                break
            # Pick a random triple to remove
            i_remove = random.randint(0, len(triples) - 1)
            rm_mask = triples[i_remove]
            rm_pts = []
            x = rm_mask
            while x:
                low = x & (-x)
                p = low.bit_length() - 1
                rm_pts.append(p)
                x ^= low

            # Remove from pair_map
            for j in range(3):
                for k in range(j + 1, 3):
                    del pair_map[(rm_pts[j], rm_pts[k])]
            triples.pop(i_remove)

            # Find candidate triples that are now valid: any triple whose all 3 pairs
            # are free (not in pair_map). We can find them by checking triples that
            # share a point with removed triple (more likely to fit) plus random.
            candidates = []
            # Triples sharing at least one point with removed triple
            for ci, cm in enumerate(all_triple_masks):
                if ci == i_remove:
                    continue
                if (cm & rm_mask) == 0:
                    continue  # doesn't share, less likely to be useful but still possible
                # Check validity
                pts_c = []
                x = cm
                while x:
                    low = x & (-x)
                    p = low.bit_length() - 1
                    pts_c.append(p)
                    x ^= low
                ok = True
                for j in range(3):
                    for k in range(j + 1, 3):
                        if (pts_c[j], pts_c[k]) in pair_map:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append(ci)

            # Also try some random candidates
            for _ in range(10):
                ci = random.randint(0, T - 1)
                if ci == i_remove:
                    continue
                cm = all_triple_masks[ci]
                pts_c = []
                x = cm
                while x:
                    low = x & (-x)
                    p = low.bit_length() - 1
                    pts_c.append(p)
                    x ^= low
                ok = True
                for j in range(3):
                    for k in range(j + 1, 3):
                        if (pts_c[j], pts_c[k]) in pair_map:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append(ci)

            if not candidates:
                # Revert
                triples.insert(i_remove, rm_mask)
                for j in range(3):
                    for k in range(j + 1, 3):
                        pair_map[(rm_pts[j], rm_pts[k])] = i_remove
                continue

            # Try each candidate, pick the best alpha
            best_new_alpha = alpha
            best_choice = -1
            for ci in candidates:
                cm = all_triple_masks[ci]
                # Tentatively add
                triples.append(cm)
                pts_c = []
                x = cm
                while x:
                    low = x & (-x)
                    p = low.bit_length() - 1
                    pts_c.append(p)
                    x ^= low
                for j in range(3):
                    for k in range(j + 1, 3):
                        pair_map[(pts_c[j], pts_c[k])] = len(triples) - 1
                na = compute_alpha(triples)
                # Undo
                triples.pop()
                for j in range(3):
                    for k in range(j + 1, 3):
                        del pair_map[(pts_c[j], pts_c[k])]
                if na < best_new_alpha:
                    best_new_alpha = na
                    best_choice = ci

            if best_choice < 0:
                # Revert removal
                triples.insert(i_remove, rm_mask)
                for j in range(3):
                    for k in range(j + 1, 3):
                        pair_map[(rm_pts[j], rm_pts[k])] = i_remove
                continue

            # Apply best choice
            cm = all_triple_masks[best_choice]
            triples.append(cm)
            pts_c = []
            x = cm
            while x:
                low = x & (-x)
                p = low.bit_length() - 1
                pts_c.append(p)
                x ^= low
            for j in range(3):
                for k in range(j + 1, 3):
                    pair_map[(pts_c[j], pts_c[k])] = len(triples) - 1
            alpha = best_new_alpha

        return triples, alpha

    # Run local search with remaining time
    remaining = deadline - time.time() - 5
    if remaining > 5 and current:
        current, current_alpha = local_search(current, current_alpha, remaining)
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = current[:]

    # Convert masks back to [a,b,c] lists
    result = []
    for m in best_triples:
        pts = []
        x = m
        while x:
            low = x & (-x)
            p = low.bit_length() - 1
            pts.append(p)
            x ^= low
        pts.sort()
        result.append(pts)

    return result


if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time() - t0:.1f}s, Triples: {len(res)}")
    # Verify validity
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            inter = set(res[i]) & set(res[j])
            if len(inter) > 1:
                print(f"Invalid pair: {res[i]} and {res[j]}")
    # Compute alpha
    print(f"Alpha: {compute_alpha([ (1<<a)|(1<<b)|(1<<c) for a,b,c in res ])}")
