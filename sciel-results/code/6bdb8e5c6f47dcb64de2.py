import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0  # safety margin under the 150s wall clock

    # ---------- Precomputation (NumPy) ----------
    k_subsets = np.array(list(itertools.combinations(range(v), k)), dtype=np.int32)
    num_k = len(k_subsets)

    t_subsets = np.array(list(itertools.combinations(range(v), t)), dtype=np.int32)
    num_t = len(t_subsets)

    # For each k-block, the indices of the t-subsets it covers.
    block_tsets = []
    for ks in k_subsets:
        idxs = [int(np.searchsorted(t_subsets, ts, side='left')) for ts in itertools.combinations(ks, t)]
        block_tsets.append(np.array(sorted(set(idxs)), dtype=np.int32))

    # Reverse incidence: for each t-subset, the list of k-blocks covering it.
    t_to_k = [[] for _ in range(num_t)]
    for i, arr in enumerate(block_tsets):
        for j in arr:
            t_to_k[int(j)].append(i)

    # ---------- Greedy ----------
    def greedy(seed_val):
        if time.time() > deadline:
            return None
        rng = random.Random(seed_val)
        chosen = []
        chosen_set = set()
        remaining = np.ones(num_t, dtype=bool)
        rem_count = num_t

        # Per-block number of still-remaining t-subsets it covers.
        gain = np.array([len(a) for a in block_tsets], dtype=np.int32)

        while rem_count > 0:
            # Pick an unused block with maximal gain.
            best = -1
            best_g = -1
            for i in range(num_k):
                if i in chosen_set:
                    continue
                g = int(gain[i])
                if g > best_g:
                    best_g = g
                    best = i
                    if best_g == int(len(block_tsets[i])):
                        break  # can't do better than full coverage
                elif g == best_g and best_g > 0 and rng.random() < 0.5:
                    best = i  # cheap random tie-break

            if best < 0 or best_g <= 0:
                return None  # infeasible

            chosen.append(best)
            chosen_set.add(best)
            arr = block_tsets[best]
            newly = arr[remaining[arr]]
            remaining[arr] = False
            rem_count -= len(newly)
            if rem_count <= 0:
                break
            # Update gains for blocks sharing the newly covered t-subsets.
            for j in newly:
                for b in t_to_k[int(j)]:
                    if b not in chosen_set:
                        gain[b] -= 1

        return chosen

    # ---------- Prune redundant blocks (multi-pass) ----------
    def prune(blocks):
        blocks = list(blocks)
        if not blocks:
            return []
        while True:
            if time.time() > deadline:
                break
            counts = np.zeros(num_t, dtype=np.int32)
            for b in blocks:
                counts[block_tsets[b]] += 1
            kept = []
            changed = False
            for b in blocks:
                arr = block_tsets[b]
                if np.any(counts[arr] == 1):
                    kept.append(b)
                else:
                    counts[arr] -= 1
                    changed = True
            if not changed:
                break
            blocks = kept
        return blocks

    # ---------- Local search: removal / swap / add-then-prune / 2-for-1 ----------
    def local_search(blocks, max_time):
        if time.time() > deadline:
            return blocks
        blocks = list(blocks)
        if not blocks:
            return blocks
        rng = random.Random(12345)
        counts = np.zeros(num_t, dtype=np.int32)
        for b in blocks:
            counts[block_tsets[b]] += 1

        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < max_time and time.time() < deadline:
            improved = False
            rng.shuffle(blocks)
            for rem in blocks[:]:
                if time.time() > deadline:
                    return blocks
                arr = block_tsets[rem]
                crit = arr[counts[arr] == 1]
                if len(crit) == 0:
                    counts[arr] -= 1
                    blocks.remove(rem)
                    improved = True
                    continue

                # Try a 1-for-1 swap.
                target = int(crit[0])
                found = -1
                for cand in t_to_k[target]:
                    if cand in blocks:
                        continue
                    if np.all(counts[block_tsets[cand][crit]] >= 0) and \
                       np.all(np.isin(crit, block_tsets[cand])):
                        found = cand
                        break
                if found >= 0:
                    counts[arr] -= 1
                    blocks.remove(rem)
                    counts[block_tsets[found]] += 1
                    blocks.append(found)
                    improved = True
                    continue

                # Try add-then-prune: add a block that helps, then drop rem if it
                # becomes redundant (pure add may enable a removal elsewhere).
                target = int(crit[0])
                for cand in t_to_k[target]:
                    if cand in blocks:
                        continue
                    counts[block_tsets[cand]] += 1
                    counts[arr] -= 1
                    blocks.remove(rem)
                    blocks.append(cand)
                    improved = True
                    break
                else:
                    # 2-for-1: drop two overlapping blocks, add one covering their criticals.
                    for other in blocks[:]:
                        if other == rem:
                            continue
                        arr2 = block_tsets[other]
                        crit2 = arr2[counts[arr2] == 1]
                        if len(crit2) == 0:
                            continue
                        need = np.unique(np.concatenate([crit, crit2]))
                        for cand in t_to_k[int(need[0])]:
                            if cand in blocks:
                                continue
                            if np.all(np.isin(need, block_tsets[cand])):
                                counts[arr] -= 1
                                counts[arr2] -= 1
                                blocks.remove(rem)
                                blocks.remove(other)
                                counts[block_tsets[cand]] += 1
                                blocks.append(cand)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        continue

        # Final prune to clean up any redundancy introduced by adds.
        blocks = prune(blocks)
        return blocks

    # ---------- Main loop: restarts, keep best, polish ----------
    best = None
    best_count = float('inf')

    # Phase 1: many quick restarts.
    for s in range(200):
        if time.time() > deadline:
            break
        blocks = greedy(s)
        if blocks is None:
            continue
        blocks = prune(blocks)
        if len(blocks) < best_count:
            best = blocks
            best_count = len(blocks)

    # Phase 2: local search on the best, using remaining time.
    if best is not None:
        remaining = deadline - time.time()
        if remaining > 3.0:
            best = local_search(best, max_time=min(60.0, remaining - 3.0))
            best_count = len(best)

    if best is None:
        blocks = greedy(0)
        if blocks is not None:
            best = prune(blocks)
        else:
            best = []

    # Convert to list of lists of ints.
    return [list(map(int, k_subsets[i])) for i in best]
