import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0  # Safety margin from 90s

    # Precompute APs: (start, step) -> list of indices
    # AP: start, start+step, start+2*step, ... < n
    aps = []
    index_to_aps = [[] for _ in range(n)]
    index_to_pos = [{} for _ in range(n)]  # index_to_pos[i][ap_id] = position in AP

    for d in range(1, n + 1):
        for start in range(d):
            idxs = list(range(start, n, d))
            if idxs:
                ap_id = len(aps)
                aps.append(idxs)
                for pos, idx in enumerate(idxs):
                    index_to_aps[idx].append(ap_id)
                    index_to_pos[idx][ap_id] = pos

    num_aps = len(aps)

    def compute_max_discrepancy(signs):
        """Compute max |partial sum| over all APs."""
        max_disc = 0
        for idxs in aps:
            curr = 0
            for idx in idxs:
                curr += signs[idx]
                if abs(curr) > max_disc:
                    max_disc = abs(curr)
        return max_disc

    def local_search(signs, max_iterations, rng, time_limit, start_time):
        """Multi-flip local search with constraint-directed bias."""
        signs = signs.copy()
        current_cost = compute_max_discrepancy(signs)
        best_cost = current_cost
        best_signs = signs.copy()

        # Precompute initial partial sums for speed
        partial_sums = []
        for idxs in aps:
            ps = [0] * len(idxs)
            curr = 0
            for k, idx in enumerate(idxs):
                curr += signs[idx]
                ps[k] = curr
            partial_sums.append(ps)

        max_sums = [max(abs(x) for x in ps) for ps in partial_sums]

        T = 5.0
        cooling = 0.998
        min_T = 0.05
        iter_count = 0

        while T > min_T and time.time() - start_time < time_limit:
            # Determine number of flips (1-4)
            num_flips = rng.choice([1, 1, 2, 2, 3, 4])
            num_flips = min(num_flips, n)

            # Constraint-directed: find APs with max discrepancy
            current_max = max(max_sums)
            worst_aps = [i for i in range(num_aps) if max_sums[i] == current_max]

            if worst_aps:
                # Pick a random worst AP and bias selection toward its indices
                target_ap = rng.choice(worst_aps)
                target_idxs = aps[target_ap]
                # 50% chance pick from target AP, 50% random
                if rng.random() < 0.5:
                    i = rng.choice(target_idxs)
                else:
                    i = rng.randrange(n)
            else:
                i = rng.randrange(n)

            # Build flip set
            flip_set = {i}
            for _ in range(num_flips - 1):
                j = rng.randrange(n)
                if j not in flip_set:
                    flip_set.add(j)

            # Compute delta for each flipped index
            # For each AP, track the net change in partial sums
            affected = set()
            for idx in flip_set:
                for ap_id in index_to_aps[idx]:
                    affected.add(ap_id)

            old_max_sums = {}
            new_max_sums = {}

            for ap_id in affected:
                idxs = aps[ap_id]
                ps = partial_sums[ap_id]
                old_max = max_sums[ap_id]
                old_max_sums[ap_id] = old_max

                # Compute changes: for each position in AP, if index is in flip_set, add 2*new_sign - 2*old_sign
                # Actually, we flip signs, so delta at each position = 2 * new_sign
                # But we need to track cumulative effect on partial sums
                # Simpler: compute new partial sums from scratch for this AP (small)
                new_ps = [0] * len(idxs)
                curr = 0
                for k, idx in enumerate(idxs):
                    if idx in flip_set:
                        curr += -signs[idx]  # flipped
                    else:
                        curr += signs[idx]
                    new_ps[k] = curr
                new_max = max(abs(x) for x in new_ps)
                new_max_sums[ap_id] = new_max

            # Compute new global cost
            # Only affected APs changed, so new global max = max(max of unaffected, max of new affected)
            # Unaffected max: we need to know if any unaffected AP had the global max
            # Simplification: recompute global max (num_aps ~1400, fast)
            # But we don't have all max_sums updated yet. Let's just compute new global cost.
            new_global = 0
            for ap_id in range(num_aps):
                if ap_id in new_max_sums:
                    if new_max_sums[ap_id] > new_global:
                        new_global = new_max_sums[ap_id]
                else:
                    if max_sums[ap_id] > new_global:
                        new_global = max_sums[ap_id]

            # Acceptance
            if new_global < current_cost:
                # Apply
                for ap_id in affected:
                    idxs = aps[ap_id]
                    ps = partial_sums[ap_id]
                    # Recompute ps
                    curr = 0
                    for k, idx in enumerate(idxs):
                        if idx in flip_set:
                            curr += -signs[idx]
                        else:
                            curr += signs[idx]
                        ps[k] = curr
                    max_sums[ap_id] = new_max_sums[ap_id]
                for idx in flip_set:
                    signs[idx] = -signs[idx]
                current_cost = new_global
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_signs = signs.copy()
            else:
                delta_cost = new_global - current_cost
                if delta_cost < 0:
                    accept = True
                else:
                    if T < 1e-9:
                        accept = False
                    else:
                        prob = math.exp(-delta_cost / T)
                        accept = rng.random() < prob
                if accept:
                    for ap_id in affected:
                        idxs = aps[ap_id]
                        ps = partial_sums[ap_id]
                        curr = 0
                        for k, idx in enumerate(idxs):
                            if idx in flip_set:
                                curr += -signs[idx]
                            else:
                                curr += signs[idx]
                            ps[k] = curr
                        max_sums[ap_id] = new_max_sums[ap_id]
                    for idx in flip_set:
                        signs[idx] = -signs[idx]
                    current_cost = new_global
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_signs = signs.copy()
                # else: revert (do nothing, signs unchanged)

            T *= cooling
            iter_count += 1

        return best_signs, best_cost

    # Iterated local search with restarts
    rng = random.Random(42)
    best_overall = None
    best_cost_overall = float('inf')

    # Initial random signs
    signs0 = [rng.choice([-1, 1]) for _ in range(n)]

    # Run multiple phases
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break

        if r == 0:
            init_signs = signs0
        else:
            # Perturb best so far
            if best_overall is not None:
                init_signs = best_overall.copy()
                # Randomly flip 5-15 indices
                num_perturb = rng.randint(5, 15)
                for _ in range(num_perturb):
                    idx = rng.randrange(n)
                    init_signs[idx] = -init_signs[idx]
            else:
                init_signs = [rng.choice([-1, 1]) for _ in range(n)]

        # Run local search with time budget
        phase_time = min(3.0, time_limit - (time.time() - start_time))
        if phase_time < 0.5:
            break

        res_signs, res_cost = local_search(
            init_signs,
            max_iterations=100000,
            rng=rng,
            time_limit=time_limit - (time.time() - start_time),
            start_time=start_time
        )

        if res_cost < best_cost_overall:
            best_cost_overall = res_cost
            best_overall = res_signs

        if best_cost_overall <= 3:
            # Very good, stop early
            break

    if best_overall is None:
        best_overall = signs0

    return best_overall
