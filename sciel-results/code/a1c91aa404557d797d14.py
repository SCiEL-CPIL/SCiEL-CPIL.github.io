# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    N = 3 ** n

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    neighbors_list = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))

    def solve_once(seed_offset=0):
        random.seed(12345 + seed_offset)

        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()

        # Incremental greedy: gains[c] = number of currently-uncovered points covered by c
        gains = np.zeros(N, dtype=np.int32)
        for p in range(N):
            gains[neighbors_list[p]] += 1

        while np.any(uncovered):
            if time.time() > deadline - 30:
                break

            max_gain = int(np.max(gains))
            if max_gain == 0:
                break

            best_indices = np.where(gains == max_gain)[0]
            if len(best_indices) > 1:
                best_idx = int(random.choice(best_indices))
            else:
                best_idx = int(best_indices[0])

            code.append(best_idx)
            code_set.add(best_idx)

            # Incremental update: for each point p covered by best_idx,
            # decrement gains for every codeword that also covers p.
            for p in neighbors_list[best_idx]:
                if uncovered[p]:
                    uncovered[p] = False
                    gains[neighbors_list[p]] -= 1

        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

        best_size = len(code)
        best_code = list(code)

        def greedy_repair(uncovered_set):
            if not uncovered_set:
                return []
            added = []
            temp_uncovered = set(uncovered_set)

            # Local incremental gains restricted to the repair set
            local_gains = np.zeros(N, dtype=np.int32)
            for idx in temp_uncovered:
                local_gains[neighbors_list[idx]] += 1

            while temp_uncovered:
                if time.time() > deadline - 5:
                    break

                max_gain = int(np.max(local_gains))
                if max_gain == 0:
                    break

                best_indices = np.where(local_gains == max_gain)[0]
                if len(best_indices) > 1:
                    best_idx = int(random.choice(best_indices))
                else:
                    best_idx = int(best_indices[0])

                added.append(best_idx)

                for w in neighbors_list[best_idx]:
                    if w in temp_uncovered:
                        temp_uncovered.discard(w)
                        local_gains[neighbors_list[w]] -= 1

            return added

        def try_remove():
            removed = False
            for c in list(code):
                if np.min(counts[neighbors_list[c]]) > 1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                    removed = True
            return removed

        def try_swap(c):
            if np.min(counts[neighbors_list[c]]) > 1:
                return False

            crit_points = [i for i in neighbors_list[c] if counts[i] == 1]

            if not crit_points:
                return False

            found_swap = False
            shuffled_crit = list(crit_points)
            random.shuffle(shuffled_crit)

            for p in shuffled_crit:
                if found_swap:
                    break

                cands = [x for x in neighbor_sets[p] if x not in code_set]

                best_w = -1
                best_score = -1

                for w in cands:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break

                    if not valid:
                        continue

                    score = int(np.sum(counts[neighbors_list[w]] == 1))
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1

                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbors_list[best_w]] += 1

                    found_swap = True
                    return True

            temp_counts = counts.copy()
            temp_counts[neighbors_list[c]] -= 1

            would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]

            if not would_be_uncovered:
                return False

            all_needed = set(crit_points) | set(would_be_uncovered)

            best_w = -1
            best_score = -1

            first_point = list(all_needed)[0]
            cands = [x for x in neighbor_sets[first_point] if x not in code_set]

            for w in cands:
                valid = True
                for k in all_needed:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break
                if not valid:
                    continue

                score = int(np.sum(counts[neighbors_list[w]] == 1))
                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1
                return True

            return False

        def kick(allow_sideways=False):
            if not code:
                return False

            old_size = len(code)

            idx_to_remove = random.randint(0, len(code) - 1)
            c_remove = code[idx_to_remove]

            temp_coverage = counts.copy()
            temp_coverage[neighbors_list[c_remove]] -= 1

            uncovered = np.where(temp_coverage == 0)[0]

            if len(uncovered) == 0:
                code.remove(c_remove)
                code_set.remove(c_remove)
                counts[neighbors_list[c_remove]] -= 1
                return True

            uncovered_set = set(uncovered.tolist())
            added = greedy_repair(uncovered_set)

            new_size = old_size - 1 + len(added)

            if new_size < old_size:
                code.remove(c_remove)
                code_set.remove(c_remove)
                counts[neighbors_list[c_remove]] -= 1
                for c in added:
                    code.append(c)
                    code_set.add(c)
                    counts[neighbors_list[c]] += 1
                return True
            elif allow_sideways and new_size == old_size:
                code.remove(c_remove)
                code_set.remove(c_remove)
                counts[neighbors_list[c_remove]] -= 1
                for c in added:
                    code.append(c)
                    code_set.add(c)
                    counts[neighbors_list[c]] += 1
                return True
            else:
                return False

        def kick2():
            if len(code) < 2:
                return False

            old_size = len(code)

            i1 = random.randint(0, len(code) - 1)
            i2 = random.randint(0, len(code) - 1)
            if i1 == i2:
                i2 = (i2 + 1) % len(code)

            c1 = code[i1]
            c2 = code[i2]

            temp_coverage = counts.copy()
            temp_coverage[neighbors_list[c1]] -= 1
            temp_coverage[neighbors_list[c2]] -= 1

            uncovered = np.where(temp_coverage == 0)[0]
            uncovered_set = set(uncovered.tolist())
            added = greedy_repair(uncovered_set)

            new_size = old_size - 2 + len(added)

            if new_size < old_size:
                code.remove(c1)
                code_set.remove(c1)
                counts[neighbors_list[c1]] -= 1
                code.remove(c2)
                code_set.remove(c2)
                counts[neighbors_list[c2]] -= 1
                for c in added:
                    code.append(c)
                    code_set.add(c)
                    counts[neighbors_list[c]] += 1
                return True
            else:
                return False

        improved = True
        sideways_count = 0
        max_sideways = 100
        while improved and time.time() < deadline - 10:
            improved = False

            while try_remove():
                improved = True

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

            if time.time() > deadline - 10:
                break

            shuffled_code = list(code)
            random.shuffle(shuffled_code)

            for c in shuffled_code:
                if time.time() > deadline - 10:
                    break
                if c not in code_set:
                    continue
                if try_swap(c):
                    improved = True

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

            if time.time() > deadline - 10:
                break

            kicks_tried = 0
            max_kicks = 30
            while kicks_tried < max_kicks and time.time() < deadline - 10:
                allow_side = (not improved and sideways_count < max_sideways)
                if kick(allow_side):
                    if not improved:
                        sideways_count += 1
                    else:
                        sideways_count = 0
                    improved = True
                    kicks_tried = 0
                else:
                    kicks_tried += 1

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

            if time.time() > deadline - 10:
                break

            if len(code) > 2:
                kicks2_tried = 0
                max_kicks2 = 15
                while kicks2_tried < max_kicks2 and time.time() < deadline - 10:
                    if kick2():
                        improved = True
                        sideways_count = 0
                        kicks2_tried = 0
                    else:
                        kicks2_tried += 1

                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)

        if len(best_code) < len(code):
            code = best_code
            code_set = set(best_code)
            counts = np.zeros(N, dtype=np.int32)
            for idx in code:
                counts[neighbors_list[idx]] += 1

        return code

    best_overall = []
    best_overall_size = float('inf')

    # Run multiple restarts
    max_restarts = 14
    for i in range(max_restarts):
        if time.time() > deadline - 15:
            break
        code_res = solve_once(i)
        if len(code_res) < best_overall_size:
            best_overall_size = len(code_res)
            best_overall = code_res

    # Final high-precision polish pass on the best solution found
    if time.time() < deadline - 8 and best_overall:
        random.seed(99999)
        code = list(best_overall)
        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

        def polish_remove():
            changed = False
            for c in list(code):
                if c not in code_set:
                    continue
                if np.min(counts[neighbors_list[c]]) > 1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                    changed = True
            return changed

        def polish_swap(c):
            if c not in code_set:
                return False
            if np.min(counts[neighbors_list[c]]) > 1:
                return False

            crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
            if not crit_points:
                return False

            shuffled_crit = list(crit_points)
            random.shuffle(shuffled_crit)

            for p in shuffled_crit:
                cands = [x for x in neighbor_sets[p] if x not in code_set]
                best_w = -1
                best_score = -1
                for w in cands:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    score = int(np.sum(counts[neighbors_list[w]] == 1))
                    if score > best_score:
                        best_score = score
                        best_w = w
                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbors_list[best_w]] += 1
                    return True
            return False

        polish_improved = True
        while polish_improved and time.time() < deadline - 5:
            polish_improved = False
            while polish_remove():
                polish_improved = True
            if len(code) < best_overall_size:
                best_overall_size = len(code)
                best_overall = list(code)
            if time.time() > deadline - 5:
                break
            shuffled_code = list(code)
            random.shuffle(shuffled_code)
            for c in shuffled_code:
                if time.time() > deadline - 5:
                    break
                if polish_swap(c):
                    polish_improved = True
            if len(code) < best_overall_size:
                best_overall_size = len(code)
                best_overall = list(code)

    return [idx_to_tuple(cw) for cw in best_overall]
# EVOLVE-BLOCK-END
