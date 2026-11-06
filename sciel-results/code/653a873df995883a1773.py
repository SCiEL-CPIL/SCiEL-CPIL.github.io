import time
import random
from collections import defaultdict
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute all forbidden triples (a,b,c) with a<b<c, 1/a = 1/b + 1/c
    # Equivalent to: c = ab/(b-a)
    triples = []
    triple_map = defaultdict(set)  # element -> set of triple indices

    for a in range(1, 181):
        for b in range(a + 1, 181):
            den = b - a
            if den == 0:
                continue
            num = a * b
            if num % den == 0:
                c = num // den
                if c > b and c <= 180:
                    triple = (a, b, c)
                    triples.append(triple)
                    idx = len(triples) - 1
                    triple_map[a].add(idx)
                    triple_map[b].add(idx)
                    triple_map[c].add(idx)

    n_triples = len(triples)

    # For each element, precompute the pairs that would conflict
    # conflict_pairs[x] = list of (y, z) such that {x,y,z} is a forbidden triple
    conflict_pairs = defaultdict(list)
    for idx, (a, b, c) in enumerate(triples):
        conflict_pairs[a].append((b, c, idx))
        conflict_pairs[b].append((a, c, idx))
        conflict_pairs[c].append((a, b, idx))

    # For each element, list of triple indices it participates in
    elem_to_triples = {x: list(triple_map[x]) for x in range(1, 181)}

    # Conflicts: conflicts[x] = set of elements that appear in a triple with x
    conflicts = defaultdict(set)
    for (a, b, c) in triples:
        conflicts[a].add(b)
        conflicts[a].add(c)
        conflicts[b].add(a)
        conflicts[b].add(c)
        conflicts[c].add(a)
        conflicts[c].add(b)

    def check_conflict(x, S):
        """Check if adding x to set S creates any forbidden triple."""
        for y, z, idx in conflict_pairs[x]:
            if y in S and z in S:
                return True
        return False

    def can_add(x, S):
        """Check if x can be added to S without creating a forbidden triple."""
        return not check_conflict(x, S)

    # Strategy 1: Start with large numbers (91-180), add small ones greedily
    S1 = set(range(91, 181))
    for x in range(90, 0, -1):
        if not check_conflict(x, S1):
            S1.add(x)

    # Strategy 2: Conflict-driven removal (start with all, remove to eliminate conflicts)
    S2 = set(range(1, 181))
    # Track active triples
    triple_active = [True] * n_triples
    conflict_count = {i: 0 for i in range(1, 181)}
    for idx, (a, b, c) in enumerate(triples):
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1

    active_triples = n_triples
    # Use deterministic order: remove elements with highest conflict first
    while active_triples > 0:
        if time.time() > deadline - 50:
            break
        # Find element with max conflict
        max_conf = -1
        to_remove = -1
        for x in range(1, 181):
            if x in S2 and conflict_count[x] > max_conf:
                max_conf = conflict_count[x]
                to_remove = x
        if to_remove == -1 or max_conf == 0:
            break
        S2.remove(to_remove)
        for tidx in elem_to_triples[to_remove]:
            if triple_active[tidx]:
                active_triples -= 1
                triple_active[tidx] = False
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x in S2:
                        conflict_count[x] -= 1

    # Strategy 3: Randomized greedy from scratch
    S3 = set()
    nums = list(range(1, 181))
    random.shuffle(nums)
    for x in nums:
        if not check_conflict(x, S3):
            S3.add(x)

    # Strategy 4: Greedy with degree ordering (fewer conflicts first)
    degree = {x: len(conflicts[x]) for x in range(1, 181)}
    order = sorted(range(1, 181), key=lambda x: degree[x])
    S4 = set()
    for x in order:
        if not check_conflict(x, S4):
            S4.add(x)

    # Pick the best initial solution
    candidates = [S1, S2, S3, S4]
    best_S = max(candidates, key=len)
    best_size = len(best_S)

    # Local search with efficient bookkeeping
    S = set(best_S)
    removed = set(range(1, 181)) - S

    # Precompute for fast swap checks
    # For each element x, store the triples it's in as a list of (other1, other2, tidx)
    elem_triples_data = {}
    for x in range(1, 181):
        elem_triples_data[x] = []
        for tidx in elem_to_triples[x]:
            a, b, c = triples[tidx]
            others = [y for y in (a, b, c) if y != x]
            elem_triples_data[x].append((others[0], others[1], tidx))

    def can_add_fast(x, S_set):
        """Fast check using precomputed data."""
        for y, z, tidx in elem_triples_data[x]:
            if y in S_set and z in S_set:
                return False
        return True

    def local_search(S_set, deadline):
        """Local search: try 0-1 adds, then 1-1 swaps, then 2-2 swaps."""
        S_local = set(S_set)
        rem = set(range(1, 181)) - S_local
        improved = True
        iter_count = 0
        max_iters = 200

        while improved and iter_count < max_iters:
            if time.time() > deadline - 5:
                break
            improved = False
            iter_count += 1

            # Phase 1: Try to add any removed element
            added_any = False
            rem_list = list(rem)
            random.shuffle(rem_list)
            for x in rem_list:
                if can_add_fast(x, S_local):
                    S_local.add(x)
                    rem.discard(x)
                    improved = True
                    added_any = True
            if added_any:
                continue

            # Phase 2: Try 1-1 swaps
            s_list = list(S_local)
            random.shuffle(s_list)
            swaps_tried = 0
            max_swaps = 20
            for y in s_list:
                if swaps_tried >= max_swaps or time.time() > deadline - 5:
                    break
                for x in rem_list:
                    if swaps_tried >= max_swaps:
                        break
                    # Check if S - {y} + {x} is valid
                    # For each triple containing x:
                    #   if y is one of the other two, the triple is broken
                    #   otherwise, check if both others are in S - {y}
                    valid_swap = True
                    for other1, other2, tidx in elem_triples_data[x]:
                        if other1 == y or other2 == y:
                            continue  # y breaks this triple
                        if other1 in S_local and other2 in S_local:
                            valid_swap = False
                            break
                    if valid_swap:
                        S_local.remove(y)
                        S_local.add(x)
                        rem.discard(x)
                        rem.add(y)
                        s_list = list(S_local)
                        random.shuffle(s_list)
                        rem_list = list(rem)
                        random.shuffle(rem_list)
                        improved = True
                        swaps_tried = 0
                        break
                    swaps_tried += 1

            # Phase 3: Try 2-2 swaps (remove two, add two)
            if not improved and time.time() < deadline - 10:
                s_list = list(S_local)
                random.shuffle(s_list)
                rem_list = list(rem)
                random.shuffle(rem_list)
                found = False
                for i in range(len(s_list)):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i + 1, len(s_list)):
                        if time.time() > deadline - 5:
                            break
                        y1, y2 = s_list[i], s_list[j]
                        # Try to find two elements from rem that can replace y1, y2
                        for k in range(len(rem_list)):
                            if time.time() > deadline - 5:
                                break
                            x1 = rem_list[k]
                            # Check if x1 can be added to S - {y1, y2}
                            valid_x1 = True
                            for other1, other2, tidx in elem_triples_data[x1]:
                                if other1 == y1 or other2 == y1 or other1 == y2 or other2 == y2:
                                    continue
                                if other1 in S_local and other2 in S_local:
                                    valid_x1 = False
                                    break
                            if not valid_x1:
                                continue
                            for m in range(k + 1, len(rem_list)):
                                x2 = rem_list[m]
                                # Check if x2 can be added to S - {y1, y2} + {x1}
                                valid_x2 = True
                                for other1, other2, tidx in elem_triples_data[x2]:
                                    if other1 == y1 or other2 == y1 or other1 == y2 or other2 == y2:
                                        continue
                                    if other1 == x1:
                                        # Check if the other element is in S - {y1,y2}
                                        other = other2
                                        if other in S_local and other != y1 and other != y2:
                                            valid_x2 = False
                                            break
                                    elif other2 == x1:
                                        other = other1
                                        if other in S_local and other != y1 and other != y2:
                                            valid_x2 = False
                                            break
                                    else:
                                        if other1 in S_local and other2 in S_local and other1 != y1 and other1 != y2 and other2 != y1 and other2 != y2:
                                            valid_x2 = False
                                            break
                                if valid_x2:
                                    # Also check x1 and x2 don't form a triple together
                                    # They'd need a third element in S - {y1,y2}
                                    # Check all triples containing both x1 and x2
                                    for tidx in elem_to_triples[x1]:
                                        a, b, c = triples[tidx]
                                        if x2 in (a, b, c):
                                            third = [v for v in (a, b, c) if v != x1 and v != x2][0]
                                            if third in S_local and third != y1 and third != y2:
                                                valid_x2 = False
                                                break
                                    if valid_x2:
                                        S_local.remove(y1)
                                        S_local.remove(y2)
                                        S_local.add(x1)
                                        S_local.add(x2)
                                        rem.discard(x1)
                                        rem.discard(x2)
                                        rem.add(y1)
                                        rem.add(y2)
                                        improved = True
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break

        return S_local

    # Run local search
    S = local_search(S, deadline)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)

    # Multiple local search restarts with different initial solutions
    for init_S in [S1, S2, S3, S4]:
        if time.time() > deadline - 20:
            break
        S_trial = local_search(set(init_S), deadline)
        if len(S_trial) > best_size:
            best_size = len(S_trial)
            best_S = set(S_trial)

    # BnB focused on the structure: try to improve by exploring small number inclusions
    # Key insight: large numbers (91-180) rarely conflict with each other
    # The conflicts are mostly between small numbers and their partners
    if time.time() < deadline - 10:
        # Try a targeted search: fix the large numbers, optimize small number selection
        large_set = set(range(91, 181))
        small_numbers = list(range(1, 91))

        # For each small number, check if it can be added to large_set
        # and track which large numbers it conflicts with
        small_candidates = []
        for x in small_numbers:
            if can_add_fast(x, large_set):
                small_candidates.append(x)

        # Now try to add as many small_candidates as possible
        # This is still a hypergraph independent set problem on small_candidates
        # but with the constraint that they also can't form triples among themselves
        # (which is unlikely for small numbers)

        # Greedy: add small candidates in order of fewest conflicts
        S_bnb = set(large_set)
        small_candidates_sorted = sorted(small_candidates, key=lambda x: len(conflicts[x] & set(small_candidates)))
        for x in small_candidates_sorted:
            if can_add_fast(x, S_bnb):
                S_bnb.add(x)

        if len(S_bnb) > best_size:
            best_size = len(S_bnb)
            best_S = set(S_bnb)

        # Try local search on this
        if time.time() < deadline - 5:
            S_bnb = local_search(S_bnb, deadline)
            if len(S_bnb) > best_size:
                best_size = len(S_bnb)
                best_S = set(S_bnb)

    # Final verification
    def verify_set(S):
        for (a, b, c) in triples:
            if a in S and b in S and c in S:
                return False
        return True

    if not verify_set(best_S):
        # Fallback
        result = list(range(91, 181))
        if verify_set(set(result)):
            return result
        return [180]

    result = list(best_S)
    if not result:
        result = list(range(91, 181))

    return result
