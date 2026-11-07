import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute comparability as bitsets for fast intersection
    # comp_bits[a] is a Python int with bit b set iff a and b are comparable
    comp_bits = [0] * total_masks
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            if (a | b) == a or (a | b) == b:
                comp_bits[a] |= (1 << b)
                comp_bits[b] |= (1 << a)

    # Precompute degree of each vertex in the full comparability graph
    full_degree = [bin(comp_bits[a]).count('1') for a in range(total_masks)]

    def count_comparables(family):
        """Count comparable pairs in a list of masks."""
        cnt = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if (comp_bits[family[i]] >> family[j]) & 1:
                    cnt += 1
        return cnt

    def greedy_initialization():
        """Build an initial family of size m using greedy strategy."""
        family = []
        used = set()

        # Start with a maximal chain: {}, {0}, {0,1}, ..., {0,...,11}
        for sz in range(n + 1):
            mask = (1 << sz) - 1
            family.append(mask)
            used.add(mask)

        # Greedily add remaining sets by maximizing comparability with existing
        remaining = [mask for mask in range(total_masks) if mask not in used]

        for _ in range(m - len(family)):
            best_mask = -1
            best_deg = -1
            for c in remaining:
                deg = 0
                for s in family:
                    if (comp_bits[c] >> s) & 1:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_mask = c
            if best_mask == -1:
                break
            family.append(best_mask)
            used.add(best_mask)
            remaining.remove(best_mask)

        # Ensure exactly m sets
        while len(family) < m:
            for mask in range(total_masks):
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
                    break

        return family

    def hill_climb(family, time_budget):
        """Local search: try replacing each element with any unused mask."""
        current_score = count_comparables(family)
        best_family = list(family)
        best_score = current_score
        used = set(family)

        improved = True
        while improved:
            if time.time() > time_budget:
                break
            improved = False

            for i in range(len(family)):
                if time.time() > time_budget:
                    break
                old_set = family[i]

                # Compute current degree of old_set within family
                old_deg = 0
                for j in range(len(family)):
                    if j != i and (comp_bits[old_set] >> family[j]) & 1:
                        old_deg += 1

                best_new = None
                best_gain = 0

                # Try all possible replacements
                for new_set in range(total_masks):
                    if new_set == old_set or new_set in used:
                        continue

                    new_deg = 0
                    for j in range(len(family)):
                        if j != i and (comp_bits[new_set] >> family[j]) & 1:
                            new_deg += 1

                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_set

                if best_new is not None and best_gain > 0:
                    family[i] = best_new
                    used.remove(old_set)
                    used.add(best_new)
                    current_score += best_gain
                    improved = True

                    if current_score > best_score:
                        best_score = current_score
                        best_family = list(family)

                    break  # Restart scan from beginning after a change

        return best_family, best_score

    def branch_and_bound_clique(candidate_pool, current_clique, time_budget):
        """
        Branch-and-bound to find a large clique in the comparability graph.
        candidate_pool: list of vertex indices that can be added
        current_clique: list of vertices in the current clique
        Returns the best clique found (list of masks).
        """
        best_clique = list(current_clique)
        best_size = len(best_clique)

        # Sort candidates by degree within the candidate pool (descending)
        def sort_key(v):
            # Count neighbors of v within candidate_pool
            deg = 0
            for u in candidate_pool:
                if u != v and (comp_bits[v] >> u) & 1:
                    deg += 1
            return -deg

        sorted_pool = sorted(candidate_pool, key=sort_key)

        def bbb(candidates, clique, depth):
            nonlocal best_clique, best_size

            if time.time() > time_budget:
                return

            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)

            if not candidates:
                return

            # Upper bound: current clique size + number of candidates
            # that are connected to all current clique members
            # Tighter bound: use sum of top (m - len(clique)) degrees within candidates
            if len(clique) + len(candidates) <= best_size:
                return

            # Filter candidates that are adjacent to all clique members
            valid_candidates = []
            for v in candidates:
                if all((comp_bits[v] >> u) & 1 for u in clique):
                    valid_candidates.append(v)

            if not valid_candidates:
                return

            # Compute upper bound using degrees within valid_candidates
            # For each candidate, count its neighbors within valid_candidates
            degs = []
            for v in valid_candidates:
                d = 0
                for u in valid_candidates:
                    if u != v and (comp_bits[v] >> u) & 1:
                        d += 1
                degs.append(d)

            # Sort by degree descending
            sorted_degs = sorted(degs, reverse=True)
            # Upper bound: len(clique) + sum of top (m - len(clique)) degrees / 2
            # But simpler: len(clique) + len(valid_candidates)
            if len(clique) + len(valid_candidates) <= best_size:
                return

            # Branch on the first candidate
            v = valid_candidates[0]
            remaining = valid_candidates[1:]

            # Include v
            bbb([u for u in remaining if (comp_bits[v] >> u) & 1], clique + [v], depth + 1)

            # Exclude v
            bbb(remaining, clique, depth + 1)

        bbb(sorted_pool, current_clique, 0)
        return best_clique

    # Phase 1: Greedy initialization
    family = greedy_initialization()
    score = count_comparables(family)
    best_family = list(family)
    best_score = score

    # Phase 2: Hill climbing
    if time.time() < deadline - 20:
        best_family, best_score = hill_climb(family, deadline - 20)

    # Phase 3: Branch-and-bound refinement on promising subsets
    if time.time() < deadline - 10:
        # Try B&B on the current best family to see if we can find a better clique
        # within the neighborhood of the current solution
        current_masks = set(best_family)
        # Candidate pool: all masks comparable to at least half the current family
        candidate_pool = []
        for mask in range(total_masks):
            if mask in current_masks:
                continue
            comp_count = 0
            for s in best_family:
                if (comp_bits[mask] >> s) & 1:
                    comp_count += 1
            if comp_count >= len(best_family) // 2:
                candidate_pool.append(mask)

        if candidate_pool:
            # Try to find a better clique using B&B
            # Start with the current family as initial clique
            initial_clique = list(best_family)
            # Only consider candidates from pool
            bb_result = branch_and_bound_clique(candidate_pool, initial_clique, deadline - 10)
            if len(bb_result) > len(best_family):
                # Found a larger clique, but we need exactly m sets
                # Use it as seed for hill climbing
                new_family = list(bb_result)
                while len(new_family) < m:
                    for mask in range(total_masks):
                        if mask not in set(new_family):
                            new_family.append(mask)
                            break
                new_score = count_comparables(new_family)
                if new_score > best_score:
                    best_family = new_family
                    best_score = new_score

    # Phase 4: Final hill climbing polish
    if time.time() < deadline - 5:
        best_family, best_score = hill_climb(best_family, deadline - 5)

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
