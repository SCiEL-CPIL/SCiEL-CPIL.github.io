# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def _compute_forbidden(m):
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    return forbidden


def _build_conflict_masks(forbidden, m):
    """For each residue r, build a bitmask of residues s such that (r-s)%m is in forbidden.
       conflict_mask[r] has bit s set iff r cannot coexist with s."""
    conflict_mask = [0] * m
    for r in range(m):
        mask = 0
        for s in range(m):
            if (r - s) % m in forbidden:
                mask |= (1 << s)
        conflict_mask[r] = mask
    return conflict_mask


def _greedy_build(order, conflict_mask, m):
    """Build an independent set by scanning `order` and adding candidates that don't conflict
       with already-selected elements."""
    selected_set = set()
    selected_list = []
    selected_mask = 0
    for cand in order:
        if (selected_mask & conflict_mask[cand]) == 0:
            selected_set.add(cand)
            selected_list.append(cand)
            selected_mask |= (1 << cand)
    return selected_set, selected_list, selected_mask


def _simulated_annealing(sset, conflict_mask, all_elements, m, time_budget):
    """Simulated annealing with add/remove/swap moves to escape local optima."""
    import math
    import random as _rnd
    
    sset = set(sset)
    smask = 0
    for s in sset:
        smask |= (1 << s)
    
    best_set = set(sset)
    best_size = len(sset)
    
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.9995
    T = T_start
    
    deadline = time.time() + time_budget
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and T > T_end:
        # Pick a random move type
        move_type = _rnd.random()
        
        if move_type < 0.3:
            # Add a random element
            cand = _rnd.choice(all_elements)
            if cand in sset:
                continue
            if (smask & conflict_mask[cand]) == 0:
                # Always accept valid additions
                sset.add(cand)
                smask |= (1 << cand)
                if len(sset) > best_size:
                    best_set = set(sset)
                    best_size = len(sset)
                    no_improve_count = 0
            else:
                # Reject (would create conflict)
                no_improve_count += 1
        elif move_type < 0.5:
            # Remove a random element
            if len(sset) == 0:
                continue
            s = _rnd.choice(list(sset))
            sset.remove(s)
            smask &= ~(1 << s)
            no_improve_count += 1
        elif move_type < 0.8:
            # Swap: remove one, add one
            if len(sset) == 0:
                continue
            s = _rnd.choice(list(sset))
            cand = _rnd.choice(all_elements)
            if cand in sset or cand == s:
                continue
            
            # Check if swap is valid
            base_mask = smask & ~(1 << s)
            if (base_mask & conflict_mask[cand]) == 0:
                # Valid swap (net 0 change in size)
                sset = (sset - {s}) | {cand}
                smask = base_mask | (1 << cand)
                no_improve_count += 1
            else:
                # Invalid swap, reject
                no_improve_count += 1
        else:
            # 2-swap: remove two, add two
            if len(sset) < 2:
                continue
            s_list = list(sset)
            s1, s2 = _rnd.sample(s_list, 2)
            cand1, cand2 = _rnd.sample(all_elements, 2)
            
            # Check if both candidates conflict with the remaining set
            base_mask = smask & ~(1 << s1) & ~(1 << s2)
            new_set = sset - {s1, s2}
            
            if (base_mask & conflict_mask[cand1]) == 0 and cand1 not in new_set:
                base_mask |= (1 << cand1)
                new_set.add(cand1)
                if (base_mask & conflict_mask[cand2]) == 0 and cand2 not in new_set:
                    # Valid 2-swap
                    sset = new_set
                    smask = base_mask | (1 << cand2)
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        T *= cooling_rate
        
        if no_improve_count > max_no_improve:
            # Restart with higher temperature to escape
            T = max(T, 1.0)
            no_improve_count = 0
    
    # Return the best set found during SA
    if len(best_set) > len(sset):
        return best_set, list(best_set)
    return sset, list(sset)


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 600.0
    forbidden = _compute_forbidden(m)
    conflict_mask = _build_conflict_masks(forbidden, m)

    best_set = set()
    best_list = []
    best_size = 0

    def try_update(sset, slist):
        nonlocal best_set, best_list, best_size
        if len(slist) > best_size:
            best_set = set(sset)
            best_list = list(slist)
            best_size = len(slist)

    # Precompute list of all elements for faster iteration
    all_elements = list(range(m))

    # --- Phase 1: greedy with multiple orderings ---
    sset, slist, smask = _greedy_build(list(range(m)), conflict_mask, m)
    try_update(sset, slist)

    random.seed(12345)
    n_shuffles = 400
    for i in range(n_shuffles):
        if time.time() > deadline - 60.0:
            break
        order = list(range(m))
        random.shuffle(order)
        sset, slist, smask = _greedy_build(order, conflict_mask, m)
        try_update(sset, slist)

    # --- Enhanced local search with 2-swap moves ---
    def local_search(sset, max_iter=50000, time_budget=None):
        """Local search with add, 1-out-2-in, 1-swap, and 2-swap moves."""
        if time_budget is None:
            time_budget = deadline - 30.0
        
        sset = set(sset)
        smask = 0
        for s in sset:
            smask |= (1 << s)

        for it in range(max_iter):
            if time.time() > time_budget:
                break
            improved = False

            # Try pure add
            for cand in all_elements:
                if cand in sset:
                    continue
                if (smask & conflict_mask[cand]) == 0:
                    sset.add(cand)
                    smask |= (1 << cand)
                    improved = True
                    break

            if not improved:
                # Try 1-out-2-in: remove one element, add two
                found = False
                slist_copy = list(sset)
                for s in slist_copy:
                    new_mask = smask & ~(1 << s)
                    new_set = sset - {s}
                    added = 0
                    for cand in all_elements:
                        if added == 2:
                            break
                        if cand in new_set:
                            continue
                        if (new_mask & conflict_mask[cand]) == 0:
                            new_set.add(cand)
                            new_mask |= (1 << cand)
                            added += 1
                    if added == 2:
                        sset = new_set
                        smask = new_mask
                        improved = True
                        found = True
                        break

            if not improved:
                # Try 1-swap: remove one, add one (net 0, enables future adds)
                found_swap = False
                slist_copy = list(sset)
                for s in slist_copy:
                    base_mask = smask & ~(1 << s)
                    for cand in all_elements:
                        if cand in sset:
                            continue
                        if (base_mask & conflict_mask[cand]) != 0:
                            continue
                        # Check if this swap enables at least one more add
                        new_mask = base_mask | (1 << cand)
                        can_add_more = False
                        for extra in all_elements:
                            if extra in sset or extra == cand:
                                continue
                            if (new_mask & conflict_mask[extra]) == 0:
                                can_add_more = True
                                break
                        if can_add_more:
                            sset = (sset - {s}) | {cand}
                            smask = new_mask
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break

            if not improved:
                # Try 2-swap: remove two elements, add two (net 0, may enable larger gains)
                found_2swap = False
                slist_copy = list(sset)
                n_s = len(slist_copy)
                for i in range(n_s):
                    s1 = slist_copy[i]
                    for j in range(i + 1, n_s):
                        s2 = slist_copy[j]
                        # Remove both s1 and s2
                        base_mask = smask & ~(1 << s1) & ~(1 << s2)
                        new_set = sset - {s1, s2}
                        added = 0
                        for cand in all_elements:
                            if added == 2:
                                break
                            if cand in new_set:
                                continue
                            if (base_mask & conflict_mask[cand]) == 0:
                                new_set.add(cand)
                                base_mask |= (1 << cand)
                                added += 1
                        if added == 2:
                            sset = new_set
                            smask = base_mask
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break

            if not improved:
                break

        return sset, list(sset)

    # --- Phase 2: local search on best set ---
    sset, slist = local_search(best_set, max_iter=50000)
    try_update(sset, slist)

    # --- Phase 2.5: Simulated annealing to escape local optima ---
    if time.time() < deadline - 60.0:
        sa_budget = min(120.0, deadline - time.time() - 30.0)
        if sa_budget > 10.0:
            sset, slist = _simulated_annealing(best_set, conflict_mask, all_elements, m, sa_budget)
            try_update(sset, slist)

    # --- Phase 3: multi-start local search from random greedy sets ---
    random.seed(67890)
    n_starts = 80
    for i in range(n_starts):
        if time.time() > deadline - 60.0:
            break
        order = list(range(m))
        random.shuffle(order)
        sset, slist, smask = _greedy_build(order, conflict_mask, m)
        sset, slist = local_search(sset, max_iter=20000)
        try_update(sset, slist)

    # --- Phase 3.5: More simulated annealing starts ---
    if time.time() < deadline - 30.0:
        sa_starts = 5
        for i in range(sa_starts):
            if time.time() > deadline - 20.0:
                break
            # Start SA from a random greedy set
            order = list(range(m))
            random.shuffle(order)
            sset, slist, smask = _greedy_build(order, conflict_mask, m)
            sa_budget = min(30.0, deadline - time.time() - 10.0)
            if sa_budget > 5.0:
                sset, slist = _simulated_annealing(sset, conflict_mask, all_elements, m, sa_budget)
                try_update(sset, slist)

    # --- Phase 4: final polish on the best solution ---
    if best_size > 0 and time.time() < deadline - 10.0:
        sset, slist = local_search(best_set, max_iter=100000, time_budget=deadline - 5.0)
        try_update(sset, slist)

    # --- Phase 4.5: Final SA polish ---
    if best_size > 0 and time.time() < deadline - 5.0:
        sa_budget = min(15.0, deadline - time.time() - 2.0)
        if sa_budget > 3.0:
            sset, slist = _simulated_annealing(best_set, conflict_mask, all_elements, m, sa_budget)
            try_update(sset, slist)

    return np.array(sorted(best_list), dtype=np.int64)


# EVOLVE-BLOCK-END
