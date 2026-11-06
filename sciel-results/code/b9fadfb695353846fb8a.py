"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill

    N = 256
    all_nums = list(range(1, N + 1))
    odds = [x for x in all_nums if x % 2 == 1]
    evens = [x for x in all_nums if x % 2 == 0]

    # Precompute conflict: (a*b) % (a+b) == 0
    def conflicts(a, b):
        s = a + b
        p = a * b
        return p % s == 0

    # For each even, precompute conflicts with all odds and all other evens
    even_conf_odds = {}   # e -> set of odds that conflict with e
    even_conf_evens = {}  # e -> set of evens that conflict with e

    for e in evens:
        oc = set()
        for o in odds:
            if conflicts(e, o):
                oc.add(o)
        even_conf_odds[e] = oc

        ec = set()
        for e2 in evens:
            if e2 != e and conflicts(e, e2):
                ec.add(e2)
        even_conf_evens[e] = ec

    # An even is "odd-safe" if it has NO conflict with any odd
    odd_safe_evens = [e for e in evens if len(even_conf_odds[e]) == 0]

    # Among odd-safe evens, build conflict graph and find large independent set
    # Greedy: sort by degree (fewer conflicts first), pick if no conflict with selected
    def greedy_independent_set(even_list, conf_map):
        """even_list: list of evens to consider; conf_map: e -> set of conflicting evens"""
        selected = []
        selected_set = set()
        # Sort by number of conflicts within even_list (ascending)
        def degree_in(e):
            return len(conf_map[e] & set(even_list))
        ordered = sorted(even_list, key=lambda e: (degree_in(e), e))
        for e in ordered:
            if not (conf_map[e] & selected_set):
                selected.append(e)
                selected_set.add(e)
        return selected_set

    def try_greedy_full(order):
        """Given an ordering of all evens, greedily add those that are compatible."""
        selected = set(odds)
        selected_evens = set()
        for e in order:
            # Check conflicts with odds
            if even_conf_odds[e] & selected:
                continue
            # Check conflicts with selected evens
            if even_conf_evens[e] & selected_evens:
                continue
            selected.add(e)
            selected_evens.add(e)
        return selected

    # Strategy 1: Start with all odds, add odd-safe evens via independent set
    best_set = set(odds)
    best_size = len(best_set)

    # Add odd-safe evens using greedy independent set with multiple orderings
    if odd_safe_evens:
        # Multiple greedy passes with different orderings
        orderings = [
            sorted(odd_safe_evens, key=lambda e: len(even_conf_evens[e])),
            sorted(odd_safe_evens, key=lambda e: -len(even_conf_evens[e])),
            sorted(odd_safe_evens, key=lambda e: e),
            sorted(odd_safe_evens, key=lambda e: -e),
            sorted(odd_safe_evens, key=lambda e: e.bit_count()),
            sorted(odd_safe_evens, key=lambda e: -e.bit_count()),
        ]
        for order in orderings:
            if time.time() > DEADLINE:
                break
            indep = greedy_independent_set(order, even_conf_evens)
            candidate = set(odds) | indep
            if len(candidate) > best_size:
                best_size = len(candidate)
                best_set = candidate

    # Strategy 2: Full greedy over all evens with various orderings
    all_orders = [
        sorted(evens, key=lambda e: (len(even_conf_odds[e]), len(even_conf_evens[e]), e)),
        sorted(evens, key=lambda e: (len(even_conf_evens[e]), len(even_conf_odds[e]), e)),
        sorted(evens, key=lambda e: e),
        sorted(evens, key=lambda e: -e),
        sorted(evens, key=lambda e: e.bit_count()),
        sorted(evens, key=lambda e: -e.bit_count()),
    ]
    for order in all_orders:
        if time.time() > DEADLINE:
            break
        s = try_greedy_full(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 3: Local search on best_set
    def local_search(current_set, max_iter=200):
        if time.time() > DEADLINE:
            return current_set, len(current_set)
        cur = set(current_set)
        cur_odds = {x for x in cur if x % 2 == 1}
        cur_evens = {x for x in cur if x % 2 == 0}
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < DEADLINE:
            improved = False
            iterations += 1

            # Try adding each unselected even
            for e in evens:
                if e in cur:
                    continue
                if even_conf_odds[e] & cur_odds:
                    continue
                if even_conf_evens[e] & cur_evens:
                    continue
                cur.add(e)
                cur_evens.add(e)
                improved = True
                break

            # Try 1-for-k swaps: remove one even, add multiple others
            if not improved and time.time() < DEADLINE:
                for rem in list(cur_evens):
                    trial_evens = cur_evens - {rem}
                    added = []
                    for e in evens:
                        if e in trial_evens or e in cur_odds:
                            continue
                        if even_conf_odds[e] & cur_odds:
                            continue
                        if even_conf_evens[e] & trial_evens:
                            continue
                        # Also check against already-added in this trial
                        if any(conflicts(e, ae) for ae in added):
                            continue
                        trial_evens.add(e)
                        added.append(e)
                    if len(added) > 1:
                        cur_evens = trial_evens
                        cur = cur_odds | cur_evens
                        improved = True
                        break

            # Try 2-for-k swaps: remove two evens, add multiple others
            if not improved and time.time() < DEADLINE:
                evens_list = list(cur_evens)
                for i in range(len(evens_list)):
                    for j in range(i + 1, len(evens_list)):
                        rem1, rem2 = evens_list[i], evens_list[j]
                        trial_evens = cur_evens - {rem1, rem2}
                        added = []
                        for e in evens:
                            if e in trial_evens or e in cur_odds:
                                continue
                            if even_conf_odds[e] & cur_odds:
                                continue
                            if even_conf_evens[e] & trial_evens:
                                continue
                            if any(conflicts(e, ae) for ae in added):
                                continue
                            trial_evens.add(e)
                            added.append(e)
                        if len(added) > 2:
                            cur_evens = trial_evens
                            cur = cur_odds | cur_evens
                            improved = True
                            break
                    if improved:
                        break

        return cur, len(cur)

    best_set, best_size = local_search(best_set)

    # Strategy 4: Random restarts with perturbation
    rng = random.Random(42)
    for restart in range(30):
        if time.time() > DEADLINE:
            break
        # Create a perturbed ordering
        shuffled = evens[:]
        for i in range(len(shuffled) - 1, 0, -1):
            if rng.random() < 0.4:
                j = rng.randint(0, i)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        s = try_greedy_full(shuffled)
        # Local search
        s, sz = local_search(s, max_iter=50)
        if sz > best_size:
            best_size = sz
            best_set = s

    # Final local search
    if time.time() < DEADLINE:
        best_set, best_size = local_search(best_set, max_iter=100)

    # Verify the result
    result = sorted(best_set)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fall back to odds only (always valid)
        return odds

    return result
# EVOLVE-BLOCK-END
