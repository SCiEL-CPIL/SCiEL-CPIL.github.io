"""Triple systems on 16 points without K(2,2,2) - simulated annealing with precomputed conflicts."""
import time
import random
import math
from itertools import combinations

def construct():
    n = 16
    deadline = time.time() + 115  # safety margin before 120s limit

    # All possible triples
    all_triples = list(combinations(range(n), 3))
    triple_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)

    # Precompute for each 6-set, the 15 groups of 8 transversal triple indices
    six_sets = list(combinations(range(n), 6))
    six_set_triples = []

    for s in six_sets:
        a, b, c, d, e, f = s
        groups = []

        def get_transversals(p1, p2, p3):
            result = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        result.append(t)
            return result

        pairs_list = [
            [(a,b),(c,d),(e,f)],
            [(a,b),(c,e),(d,f)],
            [(a,b),(c,f),(d,e)],
            [(a,c),(b,d),(e,f)],
            [(a,c),(b,e),(d,f)],
            [(a,c),(b,f),(d,e)],
            [(a,d),(b,c),(e,f)],
            [(a,d),(b,e),(c,f)],
            [(a,d),(b,f),(c,e)],
            [(a,e),(b,c),(d,f)],
            [(a,e),(b,d),(c,f)],
            [(a,e),(b,f),(c,d)],
            [(a,f),(b,c),(d,e)],
            [(a,f),(b,d),(c,e)],
            [(a,f),(b,e),(c,d)],
        ]

        for p1, p2, p3 in pairs_list:
            trans = get_transversals(p1, p2, p3)
            group = tuple(triple_idx[t] for t in trans)
            groups.append(group)

        six_set_triples.append(groups)

    # For each triple, precompute which (six_set, group) pairs it belongs to
    triple_to_conflicts = [[] for _ in range(num_triples)]
    for si, groups in enumerate(six_set_triples):
        for gi, group in enumerate(groups):
            for t in group:
                triple_to_conflicts[t].append((si, gi))

    # State: active triples, and count_6s[si][gi] = number of active triples in group gi of six-set si
    active = [False] * num_triples
    count_6s = [[0]*15 for _ in range(len(six_sets))]

    # Greedy initialization with multiple random seeds, keep best
    best_count = 0
    best_active = None
    best_counts = None

    for seed in [42, 123, 7, 999]:
        random.seed(seed)
        order = list(range(num_triples))
        random.shuffle(order)

        active_init = [False] * num_triples
        counts_init = [[0]*15 for _ in range(len(six_sets))]

        for t in order:
            can_add = True
            for si, gi in triple_to_conflicts[t]:
                if counts_init[si][gi] == 7:
                    can_add = False
                    break
            if can_add:
                active_init[t] = True
                for si, gi in triple_to_conflicts[t]:
                    counts_init[si][gi] += 1

        cnt = sum(active_init)
        if cnt > best_count:
            best_count = cnt
            best_active = active_init[:]
            best_counts = [row[:] for row in counts_init]

    active = best_active[:]
    count_6s = [row[:] for row in best_counts]

    # Simulated annealing
    current_count = best_count
    best_count_sa = best_count
    best_active_sa = active[:]

    # Temperature schedule
    t_start = 2.0
    t_end = 0.01
    num_steps = 200000  # target number of SA steps
    # We'll run as many steps as possible within the deadline

    random.seed(42)

    # Precompute lists for faster access
    active_list = [i for i in range(num_triples) if active[i]]
    inactive_list = [i for i in range(num_triples) if not active[i]]

    step = 0
    no_improve = 0
    max_no_improve = 5000

    while time.time() < deadline:
        # Temperature annealing
        if step < num_steps:
            T = t_start * (t_end / t_start) ** (step / num_steps)
        else:
            T = t_end

        # Choose a random move: add, remove, or swap
        move_type = random.random()
        if move_type < 0.3:
            # Try to add a random inactive triple
            if not inactive_list:
                step += 1
                continue
            t_add = random.choice(inactive_list)
            # Check validity
            can_add = True
            for si, gi in triple_to_conflicts[t_add]:
                if count_6s[si][gi] == 7:
                    can_add = False
                    break
            if not can_add:
                step += 1
                continue

            # Apply add
            active[t_add] = True
            for si, gi in triple_to_conflicts[t_add]:
                count_6s[si][gi] += 1
            active_list.append(t_add)
            inactive_list.remove(t_add)
            current_count += 1

            if current_count > best_count_sa:
                best_count_sa = current_count
                best_active_sa = active[:]
                no_improve = 0
            else:
                no_improve += 1

        elif move_type < 0.5:
            # Try to remove a random active triple (to escape local optima)
            if not active_list:
                step += 1
                continue
            t_rem = random.choice(active_list)
            # Accept removal with some probability (helps escape)
            if random.random() < 0.3:
                active[t_rem] = False
                for si, gi in triple_to_conflicts[t_rem]:
                    count_6s[si][gi] -= 1
                active_list.remove(t_rem)
                inactive_list.append(t_rem)
                current_count -= 1
                no_improve += 1
            else:
                step += 1
                continue

        else:
            # Swap: remove one active, add one inactive
            if not active_list or not inactive_list:
                step += 1
                continue
            t_rem = random.choice(active_list)
            t_add = random.choice(inactive_list)

            # Check if swap is valid
            temp_delta = {}
            valid = True
            for si, gi in triple_to_conflicts[t_rem]:
                key = (si, gi)
                temp_delta[key] = temp_delta.get(key, 0) - 1
            for si, gi in triple_to_conflicts[t_add]:
                key = (si, gi)
                temp_delta[key] = temp_delta.get(key, 0) + 1
                if count_6s[si][gi] + temp_delta[key] >= 8:
                    valid = False
                    break

            if valid:
                # Compute energy change (0 for valid, we just track count)
                # Swap keeps count same, so accept with probability based on T
                # Actually, swap is neutral in count. Accept to explore.
                if random.random() < math.exp(1.0 / T) if T > 0.001 else 0.5:
                    # Apply swap
                    active[t_rem] = False
                    active[t_add] = True
                    for si, gi in triple_to_conflicts[t_rem]:
                        count_6s[si][gi] -= 1
                    for si, gi in triple_to_conflicts[t_add]:
                        count_6s[si][gi] += 1
                    active_list.remove(t_rem)
                    inactive_list.remove(t_add)
                    inactive_list.append(t_rem)
                    active_list.append(t_add)
                    no_improve += 1
            else:
                step += 1
                continue

        step += 1

        # Periodic restart if stuck
        if no_improve > max_no_improve:
            # Try to greedily add any possible triple
            improved = False
            for t in inactive_list[:]:
                can_add = True
                for si, gi in triple_to_conflicts[t]:
                    if count_6s[si][gi] == 7:
                        can_add = False
                        break
                if can_add:
                    active[t] = True
                    for si, gi in triple_to_conflicts[t]:
                        count_6s[si][gi] += 1
                    active_list.append(t)
                    inactive_list.remove(t)
                    current_count += 1
                    improved = True

            if not improved:
                # Remove a random triple to break symmetry
                if active_list:
                    t_rem = random.choice(active_list)
                    active[t_rem] = False
                    for si, gi in triple_to_conflicts[t_rem]:
                        count_6s[si][gi] -= 1
                    active_list.remove(t_rem)
                    inactive_list.append(t_rem)
                    current_count -= 1
            no_improve = 0

    # Final greedy fill
    for t in inactive_list[:]:
        can_add = True
        for si, gi in triple_to_conflicts[t]:
            if count_6s[si][gi] == 7:
                can_add = False
                break
        if can_add:
            active[t] = True
            for si, gi in triple_to_conflicts[t]:
                count_6s[si][gi] += 1
            inactive_list.remove(t)

    # Use best found
    if best_count_sa >= current_count:
        result = [list(all_triples[i]) for i in range(num_triples) if best_active_sa[i]]
    else:
        result = [list(all_triples[i]) for i in range(num_triples) if active[i]]

    return result
