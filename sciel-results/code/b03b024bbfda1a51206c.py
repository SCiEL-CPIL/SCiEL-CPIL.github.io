import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin under the 150s wall-clock kill

    total_words = q ** n

    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Build neighbor lists (each codeword covers itself + all radius-1 neighbors)
    neighbors = []
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                idxs.append(tuple_to_idx(tuple(new_w)))
        neighbors.append(idxs)

    # Reverse cover: for each word, which codewords cover it
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in neighbors[i]:
            reverse_cover[j].append(i)

    # Bitsets (Python ints, 2187 bits each)
    cover_bitsets = []
    for i in range(total_words):
        bs = 0
        for j in neighbors[i]:
            bs |= (1 << j)
        cover_bitsets.append(bs)

    full_mask = (1 << total_words) - 1

    # NumPy neighbor matrix for vectorized greedy
    nb_np = np.array(neighbors, dtype=np.int32)  # shape (total_words, radius+1)

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)

        uncovered = np.ones(total_words, dtype=bool)
        code = []

        while uncovered.any():
            # Vectorized gain: number of newly-covered words per candidate
            gains = nb_np[None, :, :].T @ uncovered.astype(np.int32)
            best_gain = int(gains.max())
            if best_gain <= 0:
                break
            best_candidates = np.where(gains == best_gain)[0]
            chosen = int(best_candidates[random.randrange(len(best_candidates))])
            code.append(chosen)
            uncovered[nb_np[chosen]] = False

        return code

    def reduce_code(code):
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1

        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                for w in neighbors[c]:
                    counts[w] -= 1

                can_remove = True
                for w in neighbors[c]:
                    if counts[w] == 0:
                        can_remove = False
                        break

                if can_remove:
                    changed = True
                else:
                    for w in neighbors[c]:
                        counts[w] += 1
                    new_code.append(c)
            code = new_code
        return code

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code

        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1

        # Precompute criticality of each codeword for ordering
        crit = [0] * len(code)
        for i, c in enumerate(code):
            crit[i] = sum(1 for w in neighbors[c] if counts[w] == 1)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)

            for i in indices:
                if time.time() >= deadline:
                    break

                c_remove = code[i]

                # 1. Try removal
                for w in neighbors[c_remove]:
                    counts[w] -= 1

                holes = []
                for w in neighbors[c_remove]:
                    if counts[w] == 0:
                        holes.append(w)

                if not holes:
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Restore counts
                    for w in neighbors[c_remove]:
                        counts[w] += 1

                    # 2. Try swap
                    # Find candidates covering all holes
                    holes_set = set(holes)
                    # Sort holes by reverse-cover size for faster intersection
                    holes_sorted = sorted(holes, key=lambda x: len(reverse_cover[x]))
                    candidates = set(reverse_cover[holes_sorted[0]])
                    for h in holes_sorted[1:]:
                        candidates &= set(reverse_cover[h])
                        if not candidates:
                            break
                    candidates.discard(c_remove)

                    if candidates:
                        cand_list = list(candidates)
                        if len(cand_list) > 30:
                            best_cand = random.choice(cand_list)
                        else:
                            best_score = None
                            best_cand = None
                            for c_new in cand_list:
                                loss = 0
                                for w in neighbors[c_new]:
                                    if w not in holes_set and counts[w] == 1:
                                        loss += 1
                                if best_score is None or loss < best_score:
                                    best_score = loss
                                    best_cand = c_new

                        if best_cand is not None:
                            for w in neighbors[c_remove]:
                                counts[w] -= 1
                            for w in neighbors[best_cand]:
                                counts[w] += 1
                            code[i] = best_cand
                            improved = True
                            break
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code

        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1

        critical_scores = []
        for i, c in enumerate(code):
            crit = 0
            for w in neighbors[c]:
                if counts[w] == 1:
                    crit += 1
            critical_scores.append(crit)

        indices = list(range(len(code)))
        indices.sort(key=lambda x: critical_scores[x])
        remove_indices = indices[:k]

        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)

        # Repair using vectorized greedy
        covered = np.zeros(total_words, dtype=bool)
        for c in code:
            covered[nb_np[c]] = True
        uncovered = ~covered

        while uncovered.any() and time.time() < deadline:
            gains = nb_np[None, :, :].T @ uncovered.astype(np.int32)
            best_gain = int(gains.max())
            if best_gain <= 0:
                break
            best_candidates = np.where(gains == best_gain)[0]
            chosen = int(best_candidates[random.randrange(len(best_candidates))])
            code.append(chosen)
            uncovered[nb_np[chosen]] = False

        return code

    best_code = None
    best_size = float('inf')

    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)

        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        # Perturb-repair only if enough time remains
        if time.time() < deadline - 25 and best_code is not None:
            k = random.randint(2, 5)
            perturbed = perturb_repair(best_code, k, deadline - 12)
            perturbed = reduce_code(perturbed)

            if time.time() < deadline - 5:
                perturbed = local_search(perturbed, deadline - 2)
                perturbed = reduce_code(perturbed)

            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = list(perturbed)

        seed += 1

    if best_code is None:
        best_code = greedy_construct(0)

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))

    return result
