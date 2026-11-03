import numpy as np
import itertools
import time
import random
from scipy.optimize import linprog

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    # Generate all 2-subsets of range(11)
    supports = list(itertools.combinations(range(D), 2))
    n_sup = len(supports)

    # Build conflict graph: edge if intersection size >= 2 (i.e., 3 or 4)
    # Two supports are COMPATIBLE if |intersection| <= 1
    # Because: if |A ∩ B| <= 1, we can assign signs so that
    # sum_{k in A∩B} a_k * b_k = 0 (if |A∩B|=1, set one to 0 or opposite sign)
    # Actually for |A∩B|=1: the dot product on the intersection is ±1.
    # We need |dot| <= 2 for all pairs. With weight-4 vectors, |dot| <= 4.
    # If |A∩B|=0: dot=0, fine.
    # If |A∩B|=1: dot=±1, fine (|dot|<=2).
    # If |A∩B|=2: dot can be 0, ±2. Need to check sign assignment.
    # If |A∩B|=3: dot can be -3,-1,1,3. Can be 3 > 2, problematic.
    # If |A∩B|=4: dot can be -4,-2,0,2,4. Can be 4 > 2, problematic.
    #
    # So the safe condition is |A∩B| <= 1. For |A∩B|=2, we need careful sign assignment.
    # Let's be conservative: only allow |A∩B| <= 1.

    # Conflict: |intersection| >= 2
    conflict = np.zeros((n_sup, n_sup), dtype=bool)
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i + 1, n_sup):
            s_j = set(supports[j])
            if len(s_i & s_j) >= 2:
                conflict[i, j] = True
                conflict[j, i] = True

    # Precompute adjacency for each node
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(n_sup):
            if i != j and conflict[i, j]:
                adj[i].add(j)

    # Precompute compatible neighbors (non-conflicting)
    compat = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(n_sup):
            if i != j and not conflict[i, j]:
                compat[i].add(j)

    def run_backtracking(seed, time_limit=10.0):
        """CSP backtracking with forward checking to find a large independent set."""
        rng = random.Random(seed)
        bt_start = time.time()
        bt_deadline = bt_start + time_limit

        # Order supports by degree (fewest compatible = most constrained first)
        order = sorted(range(n_sup), key=lambda x: len(compat[x]))

        # State: selected set, and for each unselected node, remaining compatible count
        selected = []
        selected_set = set()
        # For forward checking: track which nodes are still available
        available = [True] * n_sup

        def backtrack(depth, best_so_far):
            nonlocal available
            if time.time() > bt_deadline:
                return False

            # Prune: even if we select all remaining, can't beat best
            remaining = sum(available)
            if len(selected) + remaining <= len(best_so_far):
                return False

            if not any(available):
                if len(selected) > len(best_so_far):
                    best_so_far.clear()
                    best_so_far.extend(selected)
                return True

            # Select next variable: pick from available with fewest remaining compatible options
            avail_indices = [i for i in range(n_sup) if available[i]]
            if not avail_indices:
                return False

            # Heuristic: pick node with fewest compatible available neighbors
            # (most constrained)
            def heuristic_score(i):
                return sum(1 for j in compat[i] if available[j])

            # Pick a random tie-break among the most constrained
            min_score = min(heuristic_score(i) for i in avail_indices)
            candidates = [i for i in avail_indices if heuristic_score(i) == min_score]
            idx = rng.choice(candidates)

            # Try selecting idx
            saved_available = available[:]
            selected.append(idx)
            selected_set.add(idx)

            # Forward checking: remove idx and all its conflicts
            for j in adj[idx]:
                if available[j]:
                    available[j] = False
            available[idx] = False

            if backtrack(depth + 1, best_so_far):
                pass  # best_so_far updated

            # Backtrack
            selected.pop()
            selected_set.discard(idx)
            available = saved_available

            # Try NOT selecting idx (skip it)
            saved_available2 = available[:]
            available[idx] = False

            if backtrack(depth + 1, best_so_far):
                pass

            # Restore
            available = saved_available2

            return True

        best = []
        backtrack(0, best)
        return best

    def run_greedy(seed):
        if time.time() > deadline:
            return []
        rng = np.random.default_rng(seed)
        active = np.ones(n_sup, dtype=bool)
        deg = (~conflict).sum(axis=1).astype(np.int64)  # number of compatible nodes
        selected = []

        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            # Pick node with fewest remaining compatible active neighbors (most constrained)
            active_compat = (~conflict[active_indices, :]).sum(axis=1).astype(np.int64)
            # Count only active ones
            for idx in active_indices:
                pass
            # Recompute: for each active node, how many active compatible neighbors
            compat_counts = np.zeros(n_sup, dtype=np.int64)
            for idx in active_indices:
                compat_counts[idx] = sum(1 for j in compat[idx] if active[j])

            min_count = np.min(compat_counts[active_indices])
            candidates_for_pick = active_indices[compat_counts[active_indices] == min_count]
            idx = rng.choice(candidates_for_pick)
            selected.append(idx)

            # Remove idx and its conflicting neighbors
            removed_mask = np.zeros(n_sup, dtype=bool)
            removed_mask[idx] = True
            removed_mask[conflict[idx].astype(bool) & active] = True
            removed_indices = np.where(removed_mask)[0]

            active[removed_mask] = False

        return selected

    def local_search(selected_indices, max_iters=100):
        if time.time() > deadline:
            return list(selected_indices)

        selected_set = set(selected_indices)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            indices_list = list(selected_set)
            rng_ls = random.Random(iters * 12345)
            rng_ls.shuffle(indices_list)

            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)

                # Find j not in selected_set that conflicts ONLY with i
                sel_mask = np.zeros(n_sup, dtype=bool)
                sel_mask[list(selected_set)] = True

                non_sel = np.where(~sel_mask)[0]
                if len(non_sel) == 0:
                    continue

                # For each non-selected, count conflicts with selected
                conf_with_sel = np.array([sum(1 for k in selected_set if conflict[j, k]) for j in non_sel])
                single_conf = non_sel[conf_with_sel == 1]

                valid_j = []
                for j in single_conf:
                    conflicting_sel = [k for k in selected_set if conflict[j, k]]
                    if len(conflicting_sel) == 1 and conflicting_sel[0] == i:
                        valid_j.append(j)

                if len(valid_j) >= 2:
                    found = False
                    for a_idx in range(len(valid_j)):
                        if found:
                            break
                        for b_idx in range(a_idx + 1, len(valid_j)):
                            ja, jb = valid_j[a_idx], valid_j[b_idx]
                            if not conflict[ja, jb]:
                                selected_set.discard(i)
                                selected_set.add(ja)
                                selected_set.add(jb)
                                improved = True
                                found = True
                                break

                if improved:
                    break

        return list(selected_set)

    best_set = []
    best_size = 0

    # Phase 1: Greedy with multiple restarts
    n_restarts = 300
    for seed in range(n_restarts):
        if time.time() > deadline:
            break
        result = run_greedy(seed)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result

    # Phase 2: Local search on best greedy result
    if time.time() < deadline - 10:
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set

    # Phase 3: Backtracking with forward checking (CSP approach)
    if time.time() < deadline - 20:
        # Run backtracking with time limit
        bt_time = min(20.0, deadline - time.time() - 5)
        bt_result = run_backtracking(42, time_limit=bt_time)
        if len(bt_result) > best_size:
            best_size = len(bt_result)
            best_set = bt_result
            # Apply local search to backtracking result
            if time.time() < deadline - 10:
                improved_bt = local_search(best_set)
                if len(improved_bt) > best_size:
                    best_size = len(improved_bt)
                    best_set = improved_bt

    # Phase 4: More greedy restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for seed in range(n_restarts, n_restarts + 300):
            if time.time() > deadline - 5:
                break
            result = run_greedy(seed)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
                if time.time() < deadline - 10:
                    improved_set = local_search(best_set)
                    if len(improved_set) > best_size:
                        best_size = len(improved_set)
                        best_set = improved_set

    # Phase 5: Additional local search passes
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved_set = local_search(best_set)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set

    # Construct vectors from best independent set of supports
    # For supports with |intersection| <= 1, we can always assign signs to make dot product valid
    # For |A∩B| = 0: dot = 0
    # For |A∩B| = 1: dot = ±1 (always <= 2)
    # So any sign assignment works! We just need nonzero vectors.

    vectors = []
    for idx in best_set:
        sup = supports[idx]
        # Use all 16 sign combinations? No, that's too many.
        # Actually, we want to maximize k. Each support gives up to 16 vectors.
        # But we need min pairwise distance >= max norm.
        # All vectors have norm 2 (four ±1 entries).
        # Two vectors from same support: distance squared = sum of squared diffs.
        # If they differ in 2 positions: dist^2 = 4, dist = 2. OK.
        # If they differ in 1 position: dist^2 = 4, dist = 2. OK.
        # If they differ in 3 positions: dist^2 = 12, dist = 2√3. OK.
        # If they differ in 4 positions: dist^2 = 16, dist = 4. OK.
        # All distances >= 2 = max norm. Good.
        #
        # Two vectors from different supports with |A∩B| <= 1:
        # dot product is in {-1, 0, 1} (since |A∩B| <= 1)
        # dist^2 = 4 + 4 - 2*dot = 8 - 2*dot >= 8 - 2 = 6 > 4
        # So dist >= √6 > 2. Good.
        #
        # So we can use ALL 16 sign combinations for each support!

        for signs in itertools.product([1, -1], repeat=len(sup)):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)

    if not vectors:
        # Fallback: return seed solution
        seed_vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    seed_vectors.append(v)
        return np.array(seed_vectors, dtype=np.int64)

    return np.array(vectors, dtype=np.int64)
