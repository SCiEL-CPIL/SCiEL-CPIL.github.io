import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin for 150s limit

    def compute_score_fft(U):
        """Compute score using FFT for O(M log M) set size computation."""
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0

        n = len(U)

        # For small sets, direct computation is faster
        if n <= 30:
            sumset = set()
            diffset = set()
            for i in range(n):
                for j in range(n):
                    sumset.add(U[i] + U[j])
                    diffset.add(U[i] - U[j])
            size_sum = len(sumset)
            size_diff = len(diffset)
        else:
            # FFT-based computation
            size = 2 * max_u + 1
            fft_size = 1
            while fft_size < size:
                fft_size *= 2

            f = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                f[u] = 1.0

            F = np.fft.fft(f)
            F2 = F * F
            conv = np.fft.ifft(F2).real

            threshold = 0.5
            size_sum = int(np.sum(conv[:size] > threshold))

            # Difference set via convolution with reversed signal
            g = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                if u == 0:
                    g[0] = 1.0
                else:
                    g[fft_size - u] = 1.0

            G = np.fft.fft(g)
            D = np.fft.ifft(F * G).real

            diff_nonzero = 0
            for k in range(0, max_u + 1):
                if abs(D[k]) > threshold:
                    diff_nonzero += 1
            for k in range(fft_size - max_u, fft_size):
                if abs(D[k]) > threshold:
                    diff_nonzero += 1

            size_diff = diff_nonzero

        if size_sum == 0:
            return 1.0

        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0

        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0

        return 1.0 + math.log(ratio) / denom

    def compute_score_direct(U):
        """Direct computation for verification / small sets."""
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        size_sum = len(sumset)
        size_diff = len(diffset)
        if size_sum == 0:
            return 1.0
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
        return 1.0 + math.log(ratio) / denom

    # Known optimal Golomb rulers
    golomb_rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 12, 17, 25, 29],
        9: [0, 1, 4, 10, 12, 17, 25, 29, 34],
        10: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41],
        11: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41, 52],
    }

    # Build initial candidate pool
    initial_sets = []
    for n in range(1, 12):
        if n in golomb_rulers:
            initial_sets.append(list(golomb_rulers[n]))

    # Extra hand-crafted / diverse candidates
    extra_sets = [
        [0, 1, 3, 7],
        [0, 1, 4, 6, 10],
        [0, 2, 5, 11],
        [0, 1, 5, 11],
        [0, 3, 7, 12],
        [0, 1, 4, 9, 11, 18],
        [0, 2, 7, 12, 18],
        [0, 1, 5, 10, 16],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 18],
        [0, 2, 3, 7, 12, 18],
        [0, 1, 5, 8, 12, 17],
    ]
    initial_sets.extend(extra_sets)

    # Greedy ratio-based construction: build a set by greedily adding the
    # element that maximizes the score at each step.
    def greedy_construct(target_size, time_limit=10):
        t0 = time.time()
        U = [0]
        for _ in range(target_size - 1):
            if time.time() - t0 > time_limit:
                break
            max_val = max(U)
            best_candidate = None
            best_score = -1.0
            # Search range: from 1 up to max_val + target_size * 3
            upper = max_val + target_size * 3 + 10
            for v in range(1, upper + 1):
                if v in U:
                    continue
                cand = sorted(U + [v])
                s = compute_score_direct(cand)
                if s > best_score:
                    best_score = s
                    best_candidate = cand
            if best_candidate is None:
                break
            U = best_candidate
        return U

    # Add greedy constructions
    for gs in [5, 7, 9]:
        if time.time() < deadline - 30:
            g = greedy_construct(gs, time_limit=8)
            if g and len(g) >= 3:
                initial_sets.append(g)

    # Evaluate all initial sets
    best_score = 0.0
    best_set = [0]
    for s in initial_sets:
        if time.time() > deadline:
            break
        s = sorted(set(s))
        if 0 not in s:
            s = [0] + s
        score = compute_score_fft(s)
        if score > best_score:
            best_score = score
            best_set = list(s)

    # Local search with wider moves and 2-swap
    def local_search(start_set, time_budget=30):
        current_set = sorted(set(start_set))
        if 0 not in current_set:
            current_set = [0] + current_set
        current_score = compute_score_fft(current_set)
        best_local_score = current_score
        best_local_set = list(current_set)

        no_improve = 0
        max_no_improve = 2000
        t_start = time.time()

        while time.time() - t_start < time_budget and no_improve < max_no_improve:
            if time.time() > deadline:
                break

            neighbor = list(current_set)
            op = random.randint(0, 5)

            if op == 0 and len(neighbor) > 1:
                # Remove a random element (not 0)
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
                if len(neighbor) == 0:
                    continue
                if 0 not in neighbor:
                    neighbor.append(0)
                    neighbor.sort()

            elif op == 1:
                # Add a random element (wider range)
                max_val = max(neighbor) if neighbor else 0
                upper_bound = max(max_val + 25, 30)
                new_val = random.randint(1, upper_bound)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
                else:
                    continue

            elif op == 2:
                # Modify a random element (not 0) with wider delta
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    # Wider perturbation range
                    delta = random.randint(-20, 20)
                    new_val = old_val + delta
                    if new_val >= 0 and new_val != 0 and new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        # Try a few more deltas
                        found = False
                        for d in random.sample(range(-20, 21), min(10, len(range(-20, 21)))):
                            nv = old_val + d
                            if nv >= 0 and nv != 0 and nv not in neighbor:
                                neighbor[idx] = nv
                                neighbor.sort()
                                found = True
                                break
                        if not found:
                            continue

            elif op == 3:
                # Swap: remove one element and add another
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    removed_val = neighbor.pop(idx)
                    max_val = max(neighbor) if neighbor else 0
                    upper_bound = max(max_val + 25, 30)
                    new_val = random.randint(1, upper_bound)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                    else:
                        neighbor.append(removed_val)
                        neighbor.sort()
                        continue

            elif op == 4:
                # 2-swap: remove two elements, add two new ones
                if len(neighbor) > 2:
                    idxs = random.sample(range(1, len(neighbor)), 2)
                    removed = [neighbor[i] for i in idxs]
                    for i in sorted(idxs, reverse=True):
                        neighbor.pop(i)
                    max_val = max(neighbor) if neighbor else 0
                    upper_bound = max(max_val + 25, 30)
                    new_vals = set()
                    for _ in range(2):
                        nv = random.randint(1, upper_bound)
                        if nv not in neighbor and nv not in new_vals:
                            new_vals.add(nv)
                    if len(new_vals) == 2:
                        for nv in new_vals:
                            neighbor.append(nv)
                        neighbor.sort()
                    else:
                        # Revert
                        for rv in removed:
                            neighbor.append(rv)
                        neighbor.sort()
                        continue

            else:
                # Shift a random element to a random value in a wider range
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    max_val = max(neighbor)
                    upper = max_val + 20
                    new_val = random.randint(1, upper)
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        continue

            if not neighbor or 0 not in neighbor:
                continue

            neighbor = sorted(set(neighbor))
            new_score = compute_score_fft(neighbor)

            if new_score > current_score:
                current_set = neighbor
                current_score = new_score
                no_improve = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = list(current_set)
            else:
                no_improve += 1
                # Simulated-annealing-like acceptance
                if random.random() < 0.008:
                    current_set = neighbor
                    current_score = new_score

        return best_local_set, best_local_score

    # Run multiple local searches from different starting points
    max_restarts = 8
    best_overall_score = best_score
    best_overall_set = best_set

    # First: local search from the best initial set
    if time.time() < deadline - 5:
        ls_set, ls_score = local_search(best_set, time_budget=20)
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_set = ls_set

    # Then: local search from other diverse starting points
    starting_points = []
    # From best Golomb rulers
    for n in [6, 7, 8, 9, 10, 11]:
        if n in golomb_rulers:
            starting_points.append(list(golomb_rulers[n]))
    # From extra sets
    starting_points.extend(extra_sets[:6])
    # From greedy constructions
    if time.time() < deadline - 30:
        for gs in [6, 8]:
            g = greedy_construct(gs, time_limit=5)
            if g and len(g) >= 3:
                starting_points.append(g)

    random.shuffle(starting_points)

    for sp in starting_points:
        if time.time() > deadline - 5:
            break
        if time.time() > deadline:
            break
        # Skip if we already have a very good result and this start is likely similar
        sp = sorted(set(sp))
        if 0 not in sp:
            sp = [0] + sp
        sp_score = compute_score_fft(sp)
        # Only do local search if this start is competitive or very different
        if sp_score >= best_overall_score - 0.005 or random.random() < 0.3:
            remaining = deadline - 5 - time.time()
            if remaining < 2:
                break
            ls_set, ls_score = local_search(sp, time_budget=min(remaining * 0.4, 15))
            if ls_score > best_overall_score:
                best_overall_score = ls_score
                best_overall_set = ls_set

    # Final polish: a focused local search on the best set found
    if time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        if remaining > 3:
            ls_set, ls_score = local_search(best_overall_set, time_budget=min(remaining, 10))
            if ls_score > best_overall_score:
                best_overall_score = ls_score
                best_overall_set = ls_set

    return sorted(best_overall_set)
# EVOLVE-BLOCK-END
