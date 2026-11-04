import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    D = 11

    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # 330

    # Precompute support sets for intersection checks
    sup_sets = [set(s) for s in supports]

    # Build incompatibility graph: two supports are incompatible if |intersection| >= 3
    # Two weight-4 supports sharing 3+ coordinates are incompatible
    # (their vectors can have distance < 2 when signs align)
    # Actually: distance^2 = sum of (v_i - w_i)^2 over all coords.
    # For supports A, B with |A ∩ B| = 3: on the 3 shared coords, values are ±1,
    # so differences are 0 or 2, contributing 0 or 4 each.
    # On the 1 coord in A\B: v=±1, w=0, diff^2=1
    # On the 1 coord in B\A: v=0, w=±1, diff^2=1
    # Min distance^2 = 0+0+0+1+1 = 2 (when signs match on all 3 shared)
    # Max norm^2 = 4, so we need min_dist >= max_norm, i.e., min_dist^2 >= 4.
    # But min_dist^2 = 2 < 4, so they ARE incompatible.
    # For |A ∩ B| = 2: min_dist^2 = 1+1+1+1 = 4 >= 4. Compatible!
    # For |A ∩ B| = 1: min_dist^2 = 2+2+2+2 = 8. Compatible.
    # For |A ∩ B| = 0: min_dist^2 = 4+4+4+4 = 16. Compatible.
    # So incompatible iff |A ∩ B| >= 3.

    # Use bitmasks for fast intersection checks
    sup_masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        sup_masks.append(m)

    # Build adjacency as bitmasks (Python int for arbitrary precision)
    # adj[i] is a bitmask of nodes incompatible with i
    adj = [0] * N_sup
    for i in range(N_sup):
        for j in range(i + 1, N_sup):
            inter = sup_masks[i] & sup_masks[j]
            if bin(inter).count('1') >= 3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Convert to numpy arrays for vectorized operations where possible
    # But for the B&B, we'll work with Python sets/bitmasks

    # Precompute neighbor lists as sets for the coloring routine
    neighbors = [set() for _ in range(N_sup)]
    for i in range(N_sup):
        mask = adj[i]
        for j in range(N_sup):
            if mask & (1 << j):
                neighbors[i].add(j)

    def color_bound(candidates):
        """
        Compute a coloring upper bound on the maximum independent set size
        within the candidate set. Uses greedy coloring (Tomita-style).
        Returns the number of colors used, which is an upper bound on MIS size.
        """
        if not candidates:
            return 0

        # Order candidates by degree within candidates (descending) for tighter coloring
        cand_list = list(candidates)
        cand_set = set(cand_list)

        # Sort by degree in the induced subgraph (descending)
        deg = {v: len(neighbors[v] & cand_set) for v in cand_list}
        cand_list.sort(key=lambda v: -deg[v])

        # Greedy coloring
        color = {}
        num_colors = 0
        for v in cand_list:
            # Find the smallest color not used by neighbors
            used = set()
            for nb in neighbors[v]:
                if nb in color:
                    used.add(color[nb])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            num_colors = max(num_colors, c + 1)

        return num_colors

    def branch_and_bound(candidates, current_size, best_size, best_set, deadline):
        """
        Recursive branch-and-bound for Maximum Independent Set.
        candidates: set of nodes still available
        current_size: size of current partial solution
        best_size: best solution size found so far
        best_set: best solution set found so far
        Returns (best_size, best_set)
        """
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = set()
            return

        # Upper bound: current_size + coloring bound on candidates
        ub = current_size + color_bound(candidates)
        if ub <= best_size:
            return

        # Select pivot vertex (one with max degree in candidates)
        cand_list = list(candidates)
        pivot = max(cand_list, key=lambda v: len(neighbors[v] & candidates))

        # Branch on each vertex in candidates (ordered by degree)
        # For each vertex v, branch: include v (remove v and its neighbors) or exclude v
        # We iterate over candidates, for each: include it, then the remaining candidates
        # must be from candidates \ {v} \ N(v)

        # Order by degree descending for better pruning
        cand_list.sort(key=lambda v: -len(neighbors[v] & candidates))

        for v in cand_list:
            if time.time() > deadline:
                return

            # Include v in the independent set
            new_candidates = candidates - {v} - neighbors[v]
            new_size = current_size + 1

            if new_size + color_bound(new_candidates) <= best_size:
                continue

            if new_size > best_size:
                best_size = new_size
                # We'll track the actual set at the end

            branch_and_bound(new_candidates, new_size, best_size, best_set, deadline)

            # Exclude v: remove v from candidates
            candidates = candidates - {v}

            # Re-check upper bound
            if current_size + color_bound(candidates) <= best_size:
                return

    def branch_and_bound_with_set(candidates, current_set, best_size, best_set, deadline):
        """
        Branch-and-bound that tracks the actual set for reconstruction.
        """
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return

        # Upper bound
        ub = len(current_set) + color_bound(candidates)
        if ub <= best_size:
            return

        # Order candidates by degree descending
        cand_list = list(candidates)
        cand_list.sort(key=lambda v: -len(neighbors[v] & candidates))

        for v in cand_list:
            if time.time() > deadline:
                return

            # Include v
            new_candidates = candidates - {v} - neighbors[v]
            current_set.add(v)

            if len(current_set) + color_bound(new_candidates) > best_size:
                branch_and_bound_with_set(new_candidates, current_set, best_size, best_set, deadline)

            current_set.discard(v)

            # Exclude v
            candidates = candidates - {v}

            if len(current_set) + color_bound(candidates) <= best_size:
                return

    def greedy_mis(candidates, rng):
        """Greedy MIS with minimum degree heuristic."""
        selected = set()
        avail = set(candidates)

        while avail:
            if time.time() > deadline:
                break

            # Find node with minimum degree in induced subgraph
            min_deg = float('inf')
            best_node = None
            for u in avail:
                deg = len(neighbors[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    best_node = u

            selected.add(best_node)
            to_remove = {best_node} | (neighbors[best_node] & avail)
            avail -= to_remove

        return selected

    def local_search_1_2_swap(selected, deadline):
        """Improve MIS using 1-2 swaps."""
        selected = set(selected)
        improved = True

        while improved and time.time() < deadline:
            improved = False

            for v in list(selected):
                if time.time() > deadline:
                    break

                # Find all nodes compatible with (selected - {v})
                sel_minus_v = selected - {v}
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    if not (neighbors[x] & sel_minus_v):
                        potential_adds.append(x)

                # Find a compatible pair
                found = False
                n_pot = len(potential_adds)
                for i in range(n_pot):
                    u1 = potential_adds[i]
                    for j in range(i + 1, n_pot):
                        u2 = potential_adds[j]
                        if u2 not in neighbors[u1]:
                            selected.discard(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break

                if improved:
                    break

        return selected

    def local_search_add(selected, deadline):
        """Try to add vertices compatible with all current selected."""
        selected = set(selected)
        for x in range(N_sup):
            if time.time() > deadline:
                break
            if x in selected:
                continue
            if not (neighbors[x] & selected):
                selected.add(x)
        return selected

    best_iset = set()
    best_size = 0

    # Phase 1: Greedy + local search to get a good initial solution
    rng = np.random.default_rng(42)
    n_greedy_restarts = 20
    for r in range(n_greedy_restarts):
        if time.time() > deadline - 30.0:
            break

        if r == 0:
            iset = greedy_mis(set(range(N_sup)), rng)
        else:
            avail = set(range(N_sup))
            n_remove = rng.integers(5, 40)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis(avail, rng)

        iset = local_search_add(iset, deadline)
        iset = local_search_1_2_swap(iset, deadline)

        if len(iset) > best_size:
            best_size = len(iset)
            best_iset = set(iset)

    # Phase 2: Branch-and-bound to find exact/optimal MIS
    # Use the greedy solution as initial lower bound
    if time.time() < deadline - 5.0:
        # Run B&B from the full candidate set
        # To make it tractable, we can use the greedy solution as a warm start
        # and try to beat it

        # For B&B to work well, we need a good initial bound.
        # Let's try a few different orderings.
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break

            # Randomize the order of candidates for different search paths
            all_nodes = list(range(N_sup))
            rng.shuffle(all_nodes)

            # Use a modified B&B that starts with a random subset excluded
            if attempt == 0:
                candidates = set(range(N_sup))
            else:
                # Exclude a random fraction to break symmetry
                n_excl = rng.integers(20, 80)
                excl = set(rng.choice(N_sup, size=n_excl, replace=False))
                candidates = set(range(N_sup)) - excl

            # Run B&B with a time limit
            bnb_deadline = min(deadline - 5.0, time.time() + 20.0)

            # Use the set-tracking version
            current_set = set()
            bnb_best_size = best_size
            bnb_best_set = set(best_iset)

            def bnb(candidates, current_set, deadline):
                nonlocal bnb_best_size, bnb_best_set

                if time.time() > deadline:
                    return

                if not candidates:
                    if len(current_set) > bnb_best_size:
                        bnb_best_size = len(current_set)
                        bnb_best_set = set(current_set)
                    return

                ub = len(current_set) + color_bound(candidates)
                if ub <= bnb_best_size:
                    return

                cand_list = list(candidates)
                cand_list.sort(key=lambda v: -len(neighbors[v] & candidates))

                for v in cand_list:
                    if time.time() > deadline:
                        return

                    # Include v
                    new_cands = candidates - {v} - neighbors[v]
                    current_set.add(v)

                    if len(current_set) + color_bound(new_cands) > bnb_best_size:
                        bnb(new_cands, current_set, deadline)

                    current_set.discard(v)

                    # Exclude v
                    candidates = candidates - {v}

                    if len(current_set) + color_bound(candidates) <= bnb_best_size:
                        return

            bnb(candidates, current_set, bnb_deadline)

            if bnb_best_size > best_size:
                best_size = bnb_best_size
                best_iset = set(bnb_best_set)

    # Phase 3: Final local search polish on the best solution
    if time.time() < deadline - 3.0:
        best_iset = local_search_add(best_iset, deadline)
        best_iset = local_search_1_2_swap(best_iset, deadline)

    # Generate vectors: for each support in the MIS, include all 2^4 = 16 sign patterns
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    if not vectors:
        # Fallback: return a small valid set
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)

    return np.array(vectors, dtype=np.int8)
