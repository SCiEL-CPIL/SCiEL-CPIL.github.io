"""
Triple systems on 16 points: find a large set of triples with no K(2,2,2).

Strategy:
  1. Build the conflict structure: two triples conflict if they appear together
     in some K(2,2,2).  A K(2,2,2) has 8 transversals; any 8 of them complete
     a forbidden configuration.  We track, for each K(2,2,2), how many of its
     8 transversals are in the current solution.
  2. Use a fast repair-and-grow local search:
       - Start from a random seed (small random set of triples).
       - GROW: repeatedly add a random unselected triple if it doesn't complete
         any K(2,2,2).
       - REPAIR: while some K(2,2,2) has all 8 transversals present, remove one.
       - SWAP/REMOVE-ADD: try to remove one triple and add multiple others.
  3. Run many restarts within the time budget.
  4. Return the best valid solution found.

The key insight is that we can check validity of adding a triple in O(1) amortized
time by maintaining per-K(2,2,2) counters.  The total number of K(2,2,2) structures
is ~116k, and each triple belongs to a manageable number of them.
"""

import time
import random
import itertools
import sys

# Global structures
_N_TRIPLES = 560  # C(16,3)
_ALL_TRIPLES = None
_TRIPLE_INDEX = None
# For each K(2,2,2): list of 8 triple indices
_K222S = []
# For each triple index: list of K(2,2,2) indices it belongs to
_TRIPLE_TO_K222 = [[] for _ in range(_N_TRIPLES)]

def _init_global():
    global _ALL_TRIPLES, _TRIPLE_INDEX, _K222S, _TRIPLE_TO_K222
    if _ALL_TRIPLES is not None:
        return

    pts = range(16)
    all_triples = list(itertools.combinations(pts, 3))
    _ALL_TRIPLES = all_triples
    _TRIPLE_INDEX = {t: i for i, t in enumerate(all_triples)}
    _N_TRIPLES = len(all_triples)

    # Enumerate all K(2,2,2) structures (3 disjoint pairs from 16 points)
    # For each, compute the 8 transversals and store as triple indices.
    seen_k222 = set()  # frozenset of the 6 points, to deduplicate
    for pair1 in itertools.combinations(range(16), 2):
        for pair2 in itertools.combinations(range(16), 2):
            if set(pair1) & set(pair2):
                continue
            for pair3 in itertools.combinations(range(16), 2):
                if set(pair1) & set(pair3) or set(pair2) & set(pair3):
                    continue
                six_pts = frozenset(set(pair1) | set(pair2) | set(pair3))
                if six_pts in seen_k222:
                    continue
                seen_k222.add(six_pts)

                # 8 transversals
                transversals = []
                for x in pair1:
                    for y in pair2:
                        for z in pair3:
                            t = tuple(sorted((x, y, z)))
                            transversals.append(_TRIPLE_INDEX[t])

                _K222S.append(transversals)
                for ti in transversals:
                    _TRIPLE_TO_K222[ti].append(len(_K222S) - 1)

def _grow_and_repair(selected, counters, time_limit=2.0):
    """
    Grow the solution by adding triples, then repair any violations.
    selected: set of triple indices
    counters: list of length len(_K222S), counters[k] = how many of the 8
              transversals of K(2,2,2) k are in selected.
    Returns (selected, counters, improved)
    """
    start_time = time.time()
    n_k222 = len(_K222S)
    unselected = set(range(_N_TRIPLES)) - selected

    def _is_addable(t):
        """Check if adding triple t would complete any K(2,2,2)."""
        for k_idx in _TRIPLE_TO_K222[t]:
            if counters[k_idx] == 7:  # adding t would make it 8
                return False
        return True

    # GROW phase: repeatedly add random addable triples
    added_any = True
    while added_any and time.time() - start_time < time_limit:
        added_any = False
        if not unselected:
            break
        # Try a random batch of unselected triples
        candidates = list(unselected)
        random.shuffle(candidates)
        for t in candidates:
            if time.time() - start_time > time_limit:
                break
            if t not in unselected:
                continue
            if _is_addable(t):
                selected.add(t)
                unselected.discard(t)
                for k_idx in _TRIPLE_TO_K222[t]:
                    counters[k_idx] += 1
                added_any = True

    # REPAIR phase: remove triples that complete any K(2,2,2)
    repaired = False
    for k_idx in range(n_k222):
        if counters[k_idx] >= 8:
            # Remove a random triple from this K(2,2,2)
            members = [t for t in _K222S[k_idx] if t in selected]
            if members:
                t_rm = random.choice(members)
                selected.discard(t_rm)
                unselected.add(t_rm)
                for k2 in _TRIPLE_TO_K222[t_rm]:
                    counters[k2] -= 1
                repaired = True

    if repaired:
        # May need to repair again after removals
        for k_idx in range(n_k222):
            if counters[k_idx] >= 8:
                members = [t for t in _K222S[k_idx] if t in selected]
                if members:
                    t_rm = random.choice(members)
                    selected.discard(t_rm)
                    unselected.add(t_rm)
                    for k2 in _TRIPLE_TO_K222[t_rm]:
                        counters[k2] -= 1

    return selected, counters, repaired

def _try_remove_add(selected, counters, time_limit=1.0):
    """
    Try to improve by removing one triple and adding multiple others.
    Returns (selected, counters, improved)
    """
    start_time = time.time()
    n_k222 = len(_K222S)
    unselected = set(range(_N_TRIPLES)) - selected

    if not unselected:
        return selected, counters, False

    # Try removing random triples
    sel_list = list(selected)
    random.shuffle(sel_list)

    for t_out in sel_list[:40]:
        if time.time() - start_time > time_limit:
            break

        # Simulate removal
        trial_counters = counters[:]  # copy
        for k_idx in _TRIPLE_TO_K222[t_out]:
            trial_counters[k_idx] -= 1

        # Try to add as many as possible
        trial_selected = selected.copy()
        trial_selected.discard(t_out)
        trial_unselected = set(unselected)
        trial_unselected.add(t_out)

        added_count = 0
        candidates = list(trial_unselected)
        random.shuffle(candidates)
        for t in candidates:
            if time.time() - start_time > time_limit:
                break
            if t in trial_selected:
                continue
            # Check if addable with trial_counters
            can_add = True
            for k_idx in _TRIPLE_TO_K222[t]:
                if trial_counters[k_idx] == 7:
                    can_add = False
                    break
            if can_add:
                trial_selected.add(t)
                trial_unselected.discard(t)
                for k_idx in _TRIPLE_TO_K222[t]:
                    trial_counters[k_idx] += 1
                added_count += 1

        if added_count > 1:
            # Accept the change
            selected = trial_selected
            counters = trial_counters
            unselected = trial_unselected
            return selected, counters, True

    return selected, counters, False

def _single_restart(seed_size, grow_time=1.5, swap_time=1.0):
    """
    One restart: seed with random triples, grow, repair, then try swaps.
    Returns (selected_set, size)
    """
    selected = set()
    counters = [0] * len(_K222S)

    # Seed: pick random triples
    all_triples = list(range(_N_TRIPLES))
    random.shuffle(all_triples)
    for t in all_triples[:seed_size]:
        selected.add(t)
        for k_idx in _TRIPLE_TO_K222[t]:
            counters[k_idx] += 1

    # Repair initial violations
    for k_idx in range(len(_K222S)):
        if counters[k_idx] >= 8:
            members = [t for t in _K222S[k_idx] if t in selected]
            if members:
                t_rm = random.choice(members)
                selected.discard(t_rm)
                for k2 in _TRIPLE_TO_K222[t_rm]:
                    counters[k2] -= 1

    # Grow
    selected, counters, _ = _grow_and_repair(selected, counters, time_limit=grow_time)

    # Try a few rounds of remove-add
    for _ in range(3):
        selected, counters, improved = _try_remove_add(selected, counters, time_limit=swap_time)
        if not improved:
            break

    return selected, len(selected)

def _validate(selected):
    """Check that no K(2,2,2) has all 8 transversals in selected."""
    for k_idx in range(len(_K222S)):
        count = 0
        for t in _K222S[k_idx]:
            if t in selected:
                count += 1
        if count == 8:
            return False
    return True

def _fallback_greedy(deadline):
    """
    Fallback: greedy construction with conflict checking via counters.
    """
    selected = set()
    counters = [0] * len(_K222S)
    all_triples = list(range(_N_TRIPLES))
    random.shuffle(all_triples)

    for t in all_triples:
        if time.time() > deadline:
            break
        can_add = True
        for k_idx in _TRIPLE_TO_K222[t]:
            if counters[k_idx] == 7:
                can_add = False
                break
        if can_add:
            selected.add(t)
            for k_idx in _TRIPLE_TO_K222[t]:
                counters[k_idx] += 1

    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    _init_global()

    best_set = set()
    best_size = 0

    # Phase 1: Greedy fallback to get a baseline
    baseline = _fallback_greedy(deadline - 30.0)
    best_set = baseline
    best_size = len(baseline)

    # Phase 2: Multiple restarts with local search
    while time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        if remaining < 4.0:
            break

        grow_t = min(1.5, remaining * 0.25)
        swap_t = min(1.0, remaining * 0.15)

        seed_size = random.randint(5, 20)
        selected, size = _single_restart(seed_size, grow_time=grow_t, swap_time=swap_t)

        if size > best_size:
            # Validate
            if _validate(selected):
                best_set = selected
                best_size = size

    # Final validation
    if not _validate(best_set):
        # Repair
        counters = [0] * len(_K222S)
        for t in best_set:
            for k_idx in _TRIPLE_TO_K222[t]:
                counters[k_idx] += 1
        for k_idx in range(len(_K222S)):
            if counters[k_idx] >= 8:
                members = [t for t in _K222S[k_idx] if t in best_set]
                if members:
                    t_rm = random.choice(members)
                    best_set.discard(t_rm)
                    for k2 in _TRIPLE_TO_K222[t_rm]:
                        counters[k2] -= 1

    return [list(_ALL_TRIPLES[v]) for v in sorted(best_set)]
