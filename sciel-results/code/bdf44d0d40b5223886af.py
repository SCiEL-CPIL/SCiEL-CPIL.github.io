import itertools
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Generate STS(15) using Fano plane / affine geometry construction
    # Points 0..14, triples from XOR structure
    sts_triples = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            k = i ^ j
            if k > j:
                t = tuple(sorted([i - 1, j - 1, k - 1]))
                sts_triples.append(t)

    n_sts = len(sts_triples)  # 35 triples

    # Precompute: for each STS triple, which pairs it contains
    triple_pairs = []
    for t in sts_triples:
        triple_pairs.append([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])

    # For each pair, which STS triples contain it
    pair_to_triples = defaultdict(list)
    for idx, t in enumerate(sts_triples):
        for p in triple_pairs[idx]:
            pair_to_triples[p].append(idx)

    # All 8-subsets of 16 points
    all_8_subsets = list(itertools.combinations(range(16), 8))
    n_subsets = len(all_8_subsets)

    # Precompute for each 8-subset:
    # - which STS triples are fully contained
    # - whether 15 is in the subset
    subset_sts_contained = []  # list of lists of STS triple indices
    subset_has_15 = []
    subset_points_set = []
    for s in all_8_subsets:
        s_set = frozenset(s)
        contained = []
        for idx, t in enumerate(sts_triples):
            if t[0] in s_set and t[1] in s_set and t[2] in s_set:
                contained.append(idx)
        subset_sts_contained.append(contained)
        subset_has_15.append(15 in s_set)
        subset_points_set.append(s_set)

    # Precompute for each 8-subset, the set of pairs within the subset (excluding pairs with 15)
    # For checking if an M-pair is in the subset
    subset_pairs_15 = []  # for subsets containing 15: pairs among the 7 non-15 points
    for s in all_8_subsets:
        if 15 in s:
            others = [x for x in s if x != 15]
            pairs = [(min(a, b), max(a, b)) for a in range(len(others)) for b in range(a + 1, len(others))]
            subset_pairs_15.append(pairs)
        else:
            subset_pairs_15.append(None)

    def evaluate_M(M):
        """Return the number of independent 8-sets. 0 means alpha <= 7."""
        M_set = frozenset(M)
        count = 0
        for i in range(n_subsets):
            s_set = subset_points_set[i]
            contained = subset_sts_contained[i]
            has_15 = subset_has_15[i]

            # Check if any kept STS triple is in this subset
            has_kept_triple = False
            for idx in contained:
                # Check if any pair of this triple is in M
                covered = False
                for p in triple_pairs[idx]:
                    if p in M_set:
                        covered = True
                        break
                if not covered:
                    has_kept_triple = True
                    break

            if has_kept_triple:
                continue  # This subset contains a triple, not independent

            # Check if 15 is in subset and no M-pair is within subset
            if has_15:
                pairs_in_sub = subset_pairs_15[i]
                m_pair_in_sub = False
                for p in pairs_in_sub:
                    if p in M_set:
                        m_pair_in_sub = True
                        break
                if not m_pair_in_sub:
                    # No M-pair in subset, so {15,u,v} for any M-pair (u,v) is NOT in subset
                    # Wait: the new triples are {15, u, v} for (u,v) in M.
                    # If 15 is in subset, and (u,v) is an M-pair with u,v in subset,
                    # then {15,u,v} is a triple in the system that's contained in subset.
                    # So subset is NOT independent.
                    # If no M-pair has both endpoints in subset, then no new triple is contained.
                    # So subset IS independent.
                    count += 1
            else:
                # 15 not in subset, no new triples possible
                # All STS triples in subset are covered (we checked above)
                count += 1

        return count

    def get_final_triples(M):
        M_set = set(M)
        final = []
        for u, v in M:
            final.append([15, u, v])
        for idx, t in enumerate(sts_triples):
            keep = True
            for p in triple_pairs[idx]:
                if p in M_set:
                    keep = False
                    break
            if keep:
                final.append(list(t))
        return final

    # Local search: minimize number of independent 8-sets
    best_M = []
    best_count = float('inf')

    for restart in range(300):
        if time.time() > deadline - 5:
            break

        # Random initial matching on points 0..14
        points = list(range(15))
        random.shuffle(points)
        M = []
        for i in range(0, 14, 2):
            p = tuple(sorted((points[i], points[i + 1])))
            M.append(p)

        # Local search
        stuck = 0
        while time.time() < deadline - 3:
            cur_count = evaluate_M(M)

            if cur_count < best_count:
                best_count = cur_count
                best_M = M[:]

            if best_count == 0:
                break

            # Find a violating subset
            M_set = frozenset(M)
            viol_idx = -1
            for i in range(n_subsets):
                s_set = subset_points_set[i]
                contained = subset_sts_contained[i]
                has_15 = subset_has_15[i]

                has_kept = False
                for idx in contained:
                    covered = False
                    for p in triple_pairs[idx]:
                        if p in M_set:
                            covered = True
                            break
                    if not covered:
                        has_kept = True
                        break
                if has_kept:
                    continue
                if has_15:
                    pairs_in_sub = subset_pairs_15[i]
                    m_pair_in_sub = False
                    for p in pairs_in_sub:
                        if p in M_set:
                            m_pair_in_sub = True
                            break
                    if not m_pair_in_sub:
                        viol_idx = i
                        break
                else:
                    viol_idx = i
                    break

            if viol_idx == -1:
                break

            # Generate candidates
            S_set = subset_points_set[viol_idx]
            has_15 = subset_has_15[viol_idx]
            candidates = []

            if has_15:
                # Add a pair from S\{15}
                others = [x for x in all_8_subsets[viol_idx] if x != 15]
                for a in range(len(others)):
                    for b in range(a + 1, len(others)):
                        p = tuple(sorted((others[a], others[b])))
                        # Check compatibility with current M
                        compat = True
                        for mp in M:
                            if p[0] in mp or p[1] in mp:
                                compat = False
                                break
                        if compat:
                            candidates.append(M + [p])

            # Remove a pair that covers an STS triple in S
            remove_cands = set()
            for idx in subset_sts_contained[viol_idx]:
                for p in triple_pairs[idx]:
                    if p in M:
                        remove_cands.add(p)

            for p in remove_cands:
                candidates.append([mp for mp in M if mp != p])

            # Swap: remove one pair, add another
            if len(candidates) > 20:
                # Limit candidates for speed
                random.shuffle(candidates)
                candidates = candidates[:20]

            improved = False
            for new_M in candidates:
                if time.time() > deadline - 3:
                    break
                new_count = evaluate_M(new_M)
                if new_count < cur_count:
                    M = new_M
                    improved = True
                    break

            if not improved:
                stuck += 1
                if stuck > 15:
                    # Perturbation
                    if M and random.random() < 0.5:
                        M.pop(random.randint(0, len(M) - 1))
                    else:
                        p1, p2 = random.sample(range(15), 2)
                        p = tuple(sorted((p1, p2)))
                        compat = True
                        for mp in M:
                            if p[0] in mp or p[1] in mp:
                                compat = False
                                break
                        if compat:
                            M.append(p)
                    stuck = 0

        if best_count == 0:
            break

    if best_count == 0:
        return get_final_triples(best_M)

    # If we didn't find alpha=7, try a different strategy:
    # Use a more structured initial M and more aggressive search
    if time.time() < deadline - 10:
        # Try all matchings of size 7 on 15 points (leave one point unmatched)
        # There are 15 * (13!/(2^6 * 6!)) = 15 * 10395 = 155925 matchings
        # Too many. Instead, try greedy + local search from better starts.

        # Greedy: pick pairs to minimize independent sets
        for attempt in range(100):
            if time.time() > deadline - 5:
                break

            # Start with empty M, greedily add pairs
            M = []
            used_points = set()
            for _ in range(7):
                best_pair = None
                best_score = float('inf')
                for u in range(15):
                    if u in used_points:
                        continue
                    for v in range(u + 1, 15):
                        if v in used_points:
                            continue
                        p = (u, v)
                        new_M = M + [p]
                        score = evaluate_M(new_M)
                        if score < best_score:
                            best_score = score
                            best_pair = p
                if best_pair is None:
                    break
                M.append(best_pair)
                used_points.add(best_pair[0])
                used_points.add(best_pair[1])

            # Local search from this start
            stuck = 0
            while time.time() < deadline - 3:
                cur_count = evaluate_M(M)
                if cur_count < best_count:
                    best_count = cur_count
                    best_M = M[:]
                    if best_count == 0:
                        break

                if best_count == 0:
                    break

                M_set = frozenset(M)
                viol_idx = -1
                for i in range(n_subsets):
                    s_set = subset_points_set[i]
                    contained = subset_sts_contained[i]
                    has_15 = subset_has_15[i]

                    has_kept = False
                    for idx in contained:
                        covered = False
                        for p in triple_pairs[idx]:
                            if p in M_set:
                                covered = True
                                break
                        if not covered:
                            has_kept = True
                            break
                    if has_kept:
                        continue
                    if has_15:
                        pairs_in_sub = subset_pairs_15[i]
                        m_pair_in_sub = False
                        for p in pairs_in_sub:
                            if p in M_set:
                                m_pair_in_sub = True
                                break
                        if not m_pair_in_sub:
                            viol_idx = i
                            break
                    else:
                        viol_idx = i
                        break

                if viol_idx == -1:
                    break

                S_set = subset_points_set[viol_idx]
                has_15 = subset_has_15[viol_idx]
                candidates = []

                if has_15:
                    others = [x for x in all_8_subsets[viol_idx] if x != 15]
                    for a in range(len(others)):
                        for b in range(a + 1, len(others)):
                            p = tuple(sorted((others[a], others[b])))
                            compat = True
                            for mp in M:
                                if p[0] in mp or p[1] in mp:
                                    compat = False
                                    break
                            if compat:
                                candidates.append(M + [p])

                remove_cands = set()
                for idx in subset_sts_contained[viol_idx]:
                    for p in triple_pairs[idx]:
                        if p in M:
                            remove_cands.add(p)

                for p in remove_cands:
                    candidates.append([mp for mp in M if mp != p])

                if len(candidates) > 15:
                    random.shuffle(candidates)
                    candidates = candidates[:15]

                improved = False
                for new_M in candidates:
                    if time.time() > deadline - 3:
                        break
                    new_count = evaluate_M(new_M)
                    if new_count < cur_count:
                        M = new_M
                        improved = True
                        break

                if not improved:
                    stuck += 1
                    if stuck > 10:
                        if M and random.random() < 0.5:
                            M.pop(random.randint(0, len(M) - 1))
                        else:
                            p1, p2 = random.sample(range(15), 2)
                            p = tuple(sorted((p1, p2)))
                            compat = True
                            for mp in M:
                                if p[0] in mp or p[1] in mp:
                                    compat = False
                                    break
                            if compat:
                                M.append(p)
                        stuck = 0

            if best_count == 0:
                break

    # Fallback: if best_M is empty, use a default
    if not best_M:
        points = list(range(15))
        random.shuffle(points)
        best_M = []
        for i in range(0, 14, 2):
            best_M.append(tuple(sorted((points[i], points[i + 1]))))

    return get_final_triples(best_M)
