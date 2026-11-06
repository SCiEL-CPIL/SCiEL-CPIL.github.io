import itertools
import random
import time
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560

    # Build transversals as lists of triple indices (faster iteration than bitmasks for small sets)
    transversals = []
    seen_masks = set()
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        matchings = set()
        for perm in itertools.permutations(range(6)):
            p1 = tuple(sorted((sp[perm[0]], sp[perm[1]])))
            p2 = tuple(sorted((sp[perm[2]], sp[perm[3]])))
            p3 = tuple(sorted((sp[perm[4]], sp[perm[5]])))
            matchings.add(tuple(sorted((p1, p2, p3))))
        for m in matchings:
            (p1, p2, p3) = m
            a, b = p1
            c, d = p2
            e, f = p3
            idxs = []
            for (x, y, z) in ((a, c, e), (a, c, f), (a, d, e), (a, d, f),
                              (b, c, e), (b, c, f), (b, d, e), (b, d, f)):
                idxs.append(triple_to_idx[tuple(sorted((x, y, z)))])
            key = tuple(sorted(idxs))
            if key not in seen_masks:
                seen_masks.add(key)
                transversals.append(idxs)
    num_trans = len(transversals)

    # For each triple, list of transversals it belongs to
    triple_trans = [[] for _ in range(num_triples)]
    for ti, tm in enumerate(transversals):
        for idx in tm:
            triple_trans[idx].append(ti)

    # ---- Phase 1: Greedy construction from empty set ----
    # Add triples in order of lowest conflict potential
    # Score each triple by how many transversals it participates in (lower = safer)
    triple_score = [len(triple_trans[i]) for i in range(num_triples)]
    order = sorted(range(num_triples), key=lambda i: (triple_score[i], random.random()))

    state = set()  # set of triple indices present
    # Track which transversals are "satisfied" (all 8 triples present)
    trans_counts = [0] * num_trans  # how many of the 8 triples are in state
    trans_full = [False] * num_trans

    def add_triple(idx):
        state.add(idx)
        for ti in triple_trans[idx]:
            trans_counts[ti] += 1
            if trans_counts[ti] == 8:
                trans_full[ti] = True

    def remove_triple(idx):
        state.discard(idx)
        for ti in triple_trans[idx]:
            if trans_counts[ti] > 0:
                trans_counts[ti] -= 1
                if trans_counts[ti] < 8:
                    trans_full[ti] = False

    # Greedy: try to add each triple; if it creates a conflict, skip it
    # But first, try to add all "safe" triples (those not in any full transversal yet)
    for idx in order:
        if idx in state:
            continue
        # Check if adding idx would complete any transversal
        would_complete = False
        for ti in triple_trans[idx]:
            if trans_counts[ti] == 7:
                would_complete = True
                break
        if not would_complete:
            add_triple(idx)

    # Now try to add remaining triples one by one, removing a conflicting triple if needed
    remaining = [i for i in order if i not in state]
    random.shuffle(remaining)
    for idx in remaining:
        if idx in state:
            continue
        # Check conflicts
        conflict_trans = [ti for ti in triple_trans[idx] if trans_counts[ti] == 7]
        if not conflict_trans:
            add_triple(idx)
        else:
            # Try to remove one triple from each conflict to make room
            # Only proceed if we can fix all conflicts by removing at most 1 triple
            # Find a triple whose removal fixes all conflicts
            candidates_to_remove = []
            for ti in conflict_trans:
                # The 7 present triples in this transversal
                for j in transversals[ti]:
                    if j != idx and j in state:
                        candidates_to_remove.append(j)
            # Try removing each candidate and check if all conflicts are resolved
            for rem in candidates_to_remove:
                # Check if removing rem resolves all conflict_trans
                ok = True
                for ti in conflict_trans:
                    if rem not in transversals[ti]:
                        ok = False
                        break
                if ok:
                    remove_triple(rem)
                    add_triple(idx)
                    break

    best_state = set(state)
    best_count = len(best_state)
    num_conflicts = sum(1 for tf in trans_full if tf)

    # ---- Phase 2: Local search to add more triples by resolving conflicts ----
    def try_local_search(state_set, trans_counts_arr, trans_full_arr, max_iters, time_limit):
        """Local search: try to add absent triples by removing conflicting ones."""
        local_state = set(state_set)
        local_counts = list(trans_counts_arr)
        local_full = list(trans_full_arr)
        local_best = set(local_state)
        local_best_count = len(local_best)
        local_best_conf = sum(1 for tf in local_full if tf)

        # Precompute for speed: for each triple, which transversals it's in
        # and the indices in those transversals

        def l_add(idx):
            local_state.add(idx)
            for ti in triple_trans[idx]:
                local_counts[ti] += 1
                if local_counts[ti] == 8:
                    local_full[ti] = True

        def l_remove(idx):
            local_state.discard(idx)
            for ti in triple_trans[idx]:
                if local_counts[ti] > 0:
                    local_counts[ti] -= 1
                    if local_counts[ti] < 8:
                        local_full[ti] = False

        # Build list of absent triples sorted by "desirability"
        # A triple is desirable if it participates in few transversals
        absent = [i for i in range(num_triples) if i not in local_state]
        random.shuffle(absent)

        for _ in range(max_iters):
            if time.time() > time_limit:
                break

            # Find current conflicts
            conflicts = [ti for ti in range(num_trans) if local_full[ti]]
            if not conflicts:
                # No conflicts: try to add more triples
                added_any = False
                for idx in absent[:]:
                    if idx in local_state:
                        continue
                    # Check if adding would create conflict
                    would_conflict = False
                    for ti in triple_trans[idx]:
                        if local_counts[ti] == 7:
                            would_conflict = True
                            break
                    if not would_conflict:
                        l_add(idx)
                        if idx in absent:
                            absent.remove(idx)
                        added_any = True
                        if len(local_state) > local_best_count:
                            local_best = set(local_state)
                            local_best_count = len(local_state)
                            local_best_conf = sum(1 for tf in local_full if tf)
                if not added_any:
                    break
                continue

            # Try to resolve conflicts by adding absent triples that help
            # or by removing present triples to free up capacity
            # Strategy: for each conflict, find an absent triple that, if added,
            # requires removing exactly one present triple
            resolved = False
            for ti in conflicts:
                # The 8 triples in this transversal
                t8 = transversals[ti]
                present_in_t8 = [j for j in t8 if j in local_state]
                absent_in_t8 = [j for j in t8 if j not in local_state]
                if len(present_in_t8) == 8:
                    # Full conflict: need to remove at least one
                    # Remove the one that appears in the fewest other conflicts
                    best_rem = -1
                    best_score = float('inf')
                    for j in present_in_t8:
                        # Count how many other conflicts j participates in
                        sc = 0
                        for ti2 in triple_trans[j]:
                            if ti2 != ti and local_full[ti2]:
                                sc += 1
                        if sc < best_score:
                            best_score = sc
                            best_rem = j
                    if best_rem >= 0:
                        l_remove(best_rem)
                        resolved = True
                        break
            if resolved:
                # Update best if we made progress (fewer conflicts or same conflicts but more triples)
                cur_conf = sum(1 for tf in local_full if tf)
                if cur_conf < local_best_conf or (cur_conf == local_best_conf and len(local_state) > local_best_count):
                    local_best = set(local_state)
                    local_best_count = len(local_state)
                    local_best_conf = cur_conf
                continue

            # Try: find an absent triple that, if we remove one present triple from each of its
            # near-full transversals, allows it to be added
            found_improvement = False
            for idx in absent[:30]:  # limit candidates for speed
                if idx in local_state:
                    continue
                near_full = [ti for ti in triple_trans[idx] if local_counts[ti] == 7]
                if not near_full:
                    # Can add directly
                    l_add(idx)
                    if idx in absent:
                        absent.remove(idx)
                    cur_conf = sum(1 for tf in local_full if tf)
                    if len(local_state) > local_best_count:
                        local_best = set(local_state)
                        local_best_count = len(local_state)
                        local_best_conf = cur_conf
                    found_improvement = True
                    break
                elif len(near_full) == 1:
                    # Need to remove one triple from that one transversal
                    ti = near_full[0]
                    t8 = transversals[ti]
                    # Find a present triple in t8 (not idx) that's in fewest other conflicts
                    best_rem = -1
                    best_score = float('inf')
                    for j in t8:
                        if j != idx and j in local_state:
                            sc = 0
                            for ti2 in triple_trans[j]:
                                if ti2 != ti and local_full[ti2]:
                                    sc += 1
                            if sc < best_score:
                                best_score = sc
                                best_rem = j
                    if best_rem >= 0 and best_score <= 1:
                        # Removing best_rem affects at most 1 other conflict
                        l_remove(best_rem)
                        l_add(idx)
                        if idx in absent:
                            absent.remove(idx)
                        cur_conf = sum(1 for tf in local_full if tf)
                        if len(local_state) > local_best_count or (
                                len(local_state) == local_best_count and cur_conf < local_best_conf):
                            local_best = set(local_state)
                            local_best_count = len(local_state)
                            local_best_conf = cur_conf
                        found_improvement = True
                        break
                elif len(near_full) == 2:
                    # Try to find two removals (one from each) that are disjoint or same
                    ti1, ti2 = near_full
                    t8_1 = transversals[ti1]
                    t8_2 = transversals[ti2]
                    # Find best pair of removals
                    best_pair = None
                    best_pair_score = float('inf')
                    for j1 in t8_1:
                        if j1 != idx and j1 in local_state:
                            sc1 = 0
                            for ti3 in triple_trans[j1]:
                                if ti3 != ti1 and local_full[ti3]:
                                    sc1 += 1
                            for j2 in t8_2:
                                if j2 != idx and j2 in local_state and j2 != j1:
                                    sc2 = 0
                                    for ti3 in triple_trans[j2]:
                                        if ti3 != ti2 and local_full[ti3]:
                                            sc2 += 1
                                    total = sc1 + sc2
                                    if total < best_pair_score:
                                        best_pair_score = total
                                        best_pair = (j1, j2)
                    if best_pair and best_pair_score <= 2:
                        j1, j2 = best_pair
                        l_remove(j1)
                        l_remove(j2)
                        l_add(idx)
                        if idx in absent:
                            absent.remove(idx)
                        cur_conf = sum(1 for tf in local_full if tf)
                        if len(local_state) > local_best_count or (
                                len(local_state) == local_best_count and cur_conf < local_best_conf):
                            local_best = set(local_state)
                            local_best_count = len(local_state)
                            local_best_conf = cur_conf
                        found_improvement = True
                        break
            if not found_improvement:
                # Try 2-for-2 swap: remove two present triples, add two absent ones
                if len(local_state) >= 2 and len(absent) >= 2:
                    # Pick two absent triples that together have few conflicts
                    best_swap = None
                    best_swap_score = float('inf')
                    for i1 in range(min(10, len(absent))):
                        idx1 = absent[i1]
                        if idx1 in local_state:
                            continue
                        nf1 = [ti for ti in triple_trans[idx1] if local_counts[ti] == 7]
                        for i2 in range(i1 + 1, min(10, len(absent))):
                            idx2 = absent[i2]
                            if idx2 in local_state:
                                continue
                            nf2 = [ti for ti in triple_trans[idx2] if local_counts[ti] == 7]
                            all_nf = set(nf1) | set(nf2)
                            if len(all_nf) <= 2:
                                # Find removals
                                rem_set = set()
                                feasible = True
                                for ti in all_nf:
                                    t8 = transversals[ti]
                                    cands = [j for j in t8 if j in local_state and j != idx1 and j != idx2]
                                    if not cands:
                                        feasible = False
                                        break
                                    # Pick the one with fewest other conflicts
                                    best_j = min(cands, key=lambda j: sum(
                                        1 for ti3 in triple_trans[j]
                                        if ti3 != ti and local_full[ti3]))
                                    rem_set.add(best_j)
                                if feasible and len(rem_set) <= 2:
                                    score = len(rem_set)
                                    if score < best_swap_score:
                                        best_swap_score = score
                                        best_swap = (idx1, idx2, list(rem_set))
                    if best_swap:
                        idx1, idx2, rems = best_swap
                        for r in rems:
                            l_remove(r)
                        l_add(idx1)
                        l_add(idx2)
                        if idx1 in absent:
                            absent.remove(idx1)
                        if idx2 in absent:
                            absent.remove(idx2)
                        cur_conf = sum(1 for tf in local_full if tf)
                        if len(local_state) > local_best_count or (
                                len(local_state) == local_best_count and cur_conf < local_best_conf):
                            local_best = set(local_state)
                            local_best_count = len(local_state)
                            local_best_conf = cur_conf
                        found_improvement = True

                if not found_improvement and local_state and absent:
                    # Try random swap: remove a random present triple, try to add a random absent one
                    rem = random.choice(list(local_state))
                    l_remove(rem)
                    # Now try to add some absent triples
                    added = False
                    for idx in absent[:10]:
                        if idx in local_state:
                            continue
                        would_conflict = False
                        for ti in triple_trans[idx]:
                            if local_counts[ti] == 7:
                                would_conflict = True
                                break
                        if not would_conflict:
                            l_add(idx)
                            if idx in absent:
                                absent.remove(idx)
                            added = True
                            break
                    if not added:
                        # Put it back
                        l_add(rem)
                    cur_conf = sum(1 for tf in local_full if tf)
                    if len(local_state) > local_best_count:
                        local_best = set(local_state)
                        local_best_count = len(local_state)
                        local_best_conf = cur_conf

        return local_best, local_best_count, local_best_conf

    # Run multiple rounds of local search with increasing aggressiveness
    for rnd in range(15):
        if time.time() > deadline - 2.0:
            break
        max_iters = 3000 + rnd * 500
        cur_state, cur_count, cur_conf = try_local_search(
            best_state, trans_counts, trans_full, max_iters, deadline - 1.0)
        if cur_count > best_count:
            best_state = cur_state
            best_count = cur_count
            # Rebuild trans_counts and trans_full
            trans_counts = [0] * num_trans
            trans_full = [False] * num_trans
            for idx in best_state:
                for ti in triple_trans[idx]:
                    trans_counts[ti] += 1
                    if trans_counts[ti] == 8:
                        trans_full[ti] = True
        elif rnd % 3 == 2:
            # Occasionally perturb: remove a few random triples to escape local optima
            perturb_n = min(3, len(best_state) // 20)
            if perturb_n > 0:
                to_remove = random.sample(list(best_state), perturb_n)
                for idx in to_remove:
                    best_state.discard(idx)
                    for ti in triple_trans[idx]:
                        if trans_counts[ti] > 0:
                            trans_counts[ti] -= 1
                            if trans_counts[ti] < 8:
                                trans_full[ti] = False
                best_count = len(best_state)

    # ---- Phase 3: Ensure validity by removing triples from any remaining full transversals ----
    # Use a smarter removal: pick the triple that breaks the most conflicts
    final_state = set(best_state)
    # Rebuild counts
    fc = [0] * num_trans
    ff = [False] * num_trans
    for idx in final_state:
        for ti in triple_trans[idx]:
            fc[ti] += 1
            if fc[ti] == 8:
                ff[ti] = True

    conflict_count = sum(1 for x in ff if x)
    while conflict_count > 0 and time.time() < deadline - 0.5:
        # Find all conflicting transversals
        conflict_list = [ti for ti in range(num_trans) if ff[ti]]
        if not conflict_list:
            break
        # For each triple in the union of these transversals, count how many conflicts it helps resolve
        # Pick the triple that resolves the most conflicts
        best_rem = -1
        best_resolve = 0
        # Collect all triples in conflicting transversals
        candidate_triples = set()
        for ti in conflict_list:
            for j in transversals[ti]:
                if j in final_state:
                    candidate_triples.add(j)
        for j in candidate_triples:
            resolve = 0
            for ti in triple_trans[j]:
                if ff[ti]:
                    resolve += 1
            if resolve > best_resolve:
                best_resolve = resolve
                best_rem = j
        if best_rem < 0 or best_resolve == 0:
            # Fallback: random removal from first conflict
            ti = conflict_list[0]
            t8 = transversals[ti]
            present = [j for j in t8 if j in final_state]
            if present:
                best_rem = random.choice(present)
            else:
                break
        # Remove best_rem
        final_state.discard(best_rem)
        for ti2 in triple_trans[best_rem]:
            if fc[ti2] > 0:
                fc[ti2] -= 1
                if fc[ti2] < 8:
                    if ff[ti2]:
                        conflict_count -= 1
                    ff[ti2] = False

    result = [list(all_triples[i]) for i in sorted(final_state)]
    return result
