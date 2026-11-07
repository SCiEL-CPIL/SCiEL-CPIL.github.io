import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute comparability matrix as bitmasks for speed
    # comp_mask[i] = bitmask of all j such that i and j are comparable
    comp_mask = [0] * total_masks
    for a in range(total_masks):
        mask_a = 0
        for b in range(a + 1, total_masks):
            if (a | b) == a or (a | b) == b:
                mask_a |= (1 << b)
        comp_mask[a] = mask_a
        comp_mask[b] |= (1 << a)

    # Precompute popcount for speed
    popcount = [0] * total_masks
    for i in range(1, total_masks):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def count_comparables(family):
        cnt = 0
        for i in range(len(family)):
            fi = family[i]
            for j in range(i + 1, len(family)):
                if comp_mask[fi] & (1 << family[j]):
                    cnt += 1
        return cnt

    # Greedy chain-based initialization
    family = []
    seen = set()
    for sz in range(n + 1):
        mask = (1 << sz) - 1
        family.append(mask)
        seen.add(mask)

    remaining = [mask for mask in range(total_masks) if mask not in seen]
    for _ in range(m - len(family)):
        best_mask = -1
        best_deg = -1
        for c in remaining:
            deg = 0
            for s in family:
                if comp_mask[c] & (1 << s):
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_mask = c
        if best_mask == -1:
            break
        family.append(best_mask)
        seen.add(best_mask)
        remaining.remove(best_mask)

    while len(family) < m:
        for mask in range(total_masks):
            if mask not in seen:
                family.append(mask)
                seen.add(mask)
                break
    if len(family) > m:
        family = family[:m]
        seen = set(family)

    # Hill climbing to get a good starting point
    current_score = count_comparables(family)
    best_family = list(family)
    best_score = current_score

    improved = True
    while improved:
        if time.time() > deadline - 30:
            break
        improved = False
        degrees = [0] * len(family)
        for i in range(len(family)):
            fi = family[i]
            for j in range(i + 1, len(family)):
                if comp_mask[fi] & (1 << family[j]):
                    degrees[i] += 1
                    degrees[j] += 1

        for i in range(len(family)):
            if time.time() > deadline - 30:
                break
            old_set = family[i]
            old_deg = degrees[i]
            best_new = None
            best_gain = 0
            for new_set in range(total_masks):
                if new_set == old_set or new_set in seen:
                    continue
                new_deg = 0
                for j in range(len(family)):
                    if j == i:
                        continue
                    if comp_mask[new_set] & (1 << family[j]):
                        new_deg += 1
                gain = new_deg - old_deg
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_set
            if best_new is not None:
                family[i] = best_new
                seen.remove(old_set)
                seen.add(best_new)
                current_score += best_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
                break

    # Branch and bound maximum clique search
    # We want to find a 40-clique in the comparability graph
    # Upper bound: for a partial clique P, the max additional vertices we can add
    # is bounded by the number of vertices in the intersection of neighbors of all vertices in P
    # that are not already in P, but a tighter bound uses the sum of top (m-|P|) degrees
    # within the candidate set.

    def max_clique_bnb(initial_clique, candidates, time_limit):
        """
        Find a maximum clique containing initial_clique, where candidates are
        vertices that can be added (comparable to all in initial_clique).
        Returns the best clique found (as a set of vertex indices into 0..4095).
        """
        best_clique = list(initial_clique)
        best_size = len(initial_clique)

        def bnb(clique, cand_list, depth):
            nonlocal best_clique, best_size

            if time.time() > time_limit:
                return

            # If we already have a clique of size m, we're done
            if best_size >= m:
                return

            if not cand_list:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = list(clique)
                return

            # Upper bound: len(clique) + len(cand_list)
            # But we can do better: only vertices in cand_list that are mutually
            # comparable can all be added. A simple bound is len(clique) + len(cand_list).
            # A tighter bound: sort cand_list by degree within cand_list, and the
            # max additional is at most the number of vertices that can form a clique.
            # For simplicity, use len(clique) + len(cand_list) as upper bound.
            if len(clique) + len(cand_list) <= best_size:
                return

            # We need to add (m - len(clique)) more vertices from cand_list
            needed = m - len(clique)
            if len(cand_list) < needed:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = list(clique)
                return

            # Branch: pick a vertex from cand_list
            # Order by degree within cand_list (descending) for better pruning
            # Compute degrees within cand_list
            cand_set = set(cand_list)
            degs = []
            for v in cand_list:
                d = 0
                for w in cand_list:
                    if w != v and (comp_mask[v] & (1 << w)):
                        d += 1
                degs.append((d, v))
            degs.sort(reverse=True)

            # If the top (needed) degrees sum suggests we can't beat best_size, prune
            # Actually, a simpler bound: if needed > len(cand_list), prune (already checked)

            # Try adding vertices in order of degree
            for idx, (d, v) in enumerate(degs):
                if time.time() > time_limit:
                    return
                if len(clique) + len(degs) - idx <= best_size:
                    break  # Can't improve

                # Add v to clique
                new_clique = clique + [v]
                # New candidates: those in cand_list after v that are comparable to v
                new_cand = []
                for _, w in degs[idx+1:]:
                    if comp_mask[v] & (1 << w):
                        new_cand.append(w)

                bnb(new_clique, new_cand, depth + 1)

        bnb(list(initial_clique), list(candidates), 0)
        return best_clique

    # Use branch and bound to find a 40-clique
    # Start from the current best family
    current_clique = list(best_family)
    # Candidates: all vertices comparable to all in current_clique
    # For BnB, we want to find if we can get a 40-clique

    # Actually, let's use a different approach: 
    # Start with an empty clique and use BnB to find a 40-clique
    # But that's too slow. Instead, use the current family as a seed.

    # Let's try: take the current best_family, and try to improve it using BnB
    # by considering swaps.

    # Alternative: use BnB directly on the full graph with a good initial bound
    # The current best_score gives us a lower bound.

    # Let me try a simpler BnB: find a clique of size 40
    # Start with a greedy clique, then use BnB to extend/improve

    # Greedy clique construction for BnB seed
    def greedy_clique():
        clique = []
        remaining = set(range(total_masks))
        while remaining and len(clique) < m:
            # Pick vertex with max degree in remaining subgraph
            best_v = None
            best_d = -1
            for v in list(remaining):
                d = 0
                for w in remaining:
                    if w != v and (comp_mask[v] & (1 << w)):
                        d += 1
                if d > best_d:
                    best_d = d
                    best_v = v
            if best_v is None:
                break
            clique.append(best_v)
            remaining.discard(best_v)
            # Remove vertices not comparable to best_v
            remaining = {w for w in remaining if comp_mask[best_v] & (1 << w)}
        return clique

    if time.time() < deadline - 15:
        greedy_c = greedy_clique()
        if len(greedy_c) >= 20:
            # Use BnB to try to extend this to size 40
            # Candidates: vertices comparable to all in greedy_c
            candidates = []
            for v in range(total_masks):
                if v in greedy_c:
                    continue
                can_add = True
                for g in greedy_c:
                    if not (comp_mask[v] & (1 << g)):
                        can_add = False
                        break
                if can_add:
                    candidates.append(v)

            if len(candidates) >= m - len(greedy_c):
                result_clique = max_clique_bnb(greedy_c, candidates, deadline - 10)
                if len(result_clique) > len(best_family):
                    best_family = result_clique
                    best_score = count_comparables(best_family)

    # Final hill climbing polish
    if time.time() < deadline - 5:
        family = list(best_family)
        seen = set(family)
        current_score = best_score

        for _ in range(5):
            if time.time() > deadline - 5:
                break
            improved = False
            degrees = [0] * len(family)
            for i in range(len(family)):
                fi = family[i]
                for j in range(i + 1, len(family)):
                    if comp_mask[fi] & (1 << family[j]):
                        degrees[i] += 1
                        degrees[j] += 1

            for i in range(len(family)):
                if time.time() > deadline - 5:
                    break
                old_set = family[i]
                old_deg = degrees[i]
                best_new = None
                best_gain = 0
                for new_set in range(total_masks):
                    if new_set == old_set or new_set in seen:
                        continue
                    new_deg = 0
                    for j in range(len(family)):
                        if j == i:
                            continue
                        if comp_mask[new_set] & (1 << family[j]):
                            new_deg += 1
                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_set
                if best_new is not None and best_gain > 0:
                    family[i] = best_new
                    seen.remove(old_set)
                    seen.add(best_new)
                    current_score += best_gain
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_family = list(family)
                    break
            if not improved:
                break

    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
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
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
