import itertools
import time
import random
import heapq


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0   # safety margin before the 150s kill
    KNOWN_BEST = 151                # La Jolla best known for C(19,7,4)

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubs = list(itertools.combinations(points, t))
    n_tsubs = len(all_tsubs)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # ---- Precomputation ----
    block_tsubs = []                 # list of t-subset indices per block
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for block in all_blocks:
        lst = []
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            lst.append(idx)
            tsub_blocks[idx].append(block)  # placeholder, replaced below
    # rebuild tsub_blocks with block indices
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            tsub_blocks[tsub_idx[ts]].append(b_idx)
        block_tsubs.append([tsub_idx[ts] for ts in itertools.combinations(block, t)])

    full_mask = (1 << n_tsubs) - 1

    def valid(blocks):
        covered = 0
        for b in blocks:
            covered |= 0
            for ts in block_tsubs[b]:
                covered |= (1 << ts)
        return covered == full_mask

    # ---- Heap-based greedy ----
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()

        gains = [len(block_tsubs[b]) for b in range(n_blocks)]
        heap = [(-gains[b], b) for b in range(n_blocks)]
        heapq.heapify(heap)

        while uncovered:
            if time.time() > deadline - 12.0:
                break
            best_block = -1
            best_gain = 0
            while heap:
                neg_g, b = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b] and b not in selected_set:
                    best_block = b
                    best_gain = g
                    break
            if best_block == -1:
                break

            selected.append(best_block)
            selected_set.add(best_block)
            gains[best_block] = -1

            new_covered = 0
            for ts in block_tsubs[best_block]:
                if uncovered & (1 << ts):
                    new_covered |= (1 << ts)
            uncovered &= ~new_covered

            # update gains of blocks covering newly covered tsubs
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                for b2 in tsub_blocks[ts_idx]:
                    if b2 not in selected_set:
                        gains[b2] -= 1
                        heapq.heappush(heap, (-gains[b2], b2))
                m ^= lsb

        return selected

    # ---- Local search (incremental cover counts) ----
    def local_search(blocks, seed):
        rng = random.Random(seed)
        blocks = list(blocks)
        block_set = set(blocks)

        cover_count = [0] * n_tsubs
        for b in block_set:
            for ts in block_tsubs[b]:
                cover_count[ts] += 1

        improved = True
        while improved and time.time() < deadline - 6.0:
            improved = False

            # Phase 1: remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts in block_tsubs[b]:
                    if cover_count[ts] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts in block_tsubs[b]:
                        cover_count[ts] -= 1
                improved = True
                continue

            # Phase 2: 1-for-1 swaps (intersection-restricted candidates)
            current = list(block_set)
            rng.shuffle(current)
            for b in current:
                if time.time() > deadline - 6.0:
                    break
                if b not in block_set:
                    continue
                unique = [ts for ts in block_tsubs[b] if cover_count[ts] == 1]
                if not unique:
                    block_set.remove(b)
                    for ts in block_tsubs[b]:
                        cover_count[ts] -= 1
                    improved = True
                    break
                min_ts = min(unique, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                for ts in unique[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts])
                candidates.discard(b)
                candidates = [c for c in candidates if c not in block_set]
                if not candidates:
                    continue
                rng.shuffle(candidates)
                for r in candidates[:25]:
                    # apply swap
                    for ts in block_tsubs[b]:
                        cover_count[ts] -= 1
                    for ts in block_tsubs[r]:
                        cover_count[ts] += 1
                    # check feasibility
                    if all(cover_count[ts] >= 1 for ts in block_tsubs[b]):
                        block_set.remove(b)
                        block_set.add(r)
                        improved = True
                        break
                    else:
                        # revert
                        for ts in block_tsubs[r]:
                            cover_count[ts] -= 1
                        for ts in block_tsubs[b]:
                            cover_count[ts] += 1
                if improved:
                    break

            # Phase 3: 2-for-1 merge (remove two, add one covering their unique union)
            if not improved and time.time() < deadline - 8.0:
                cur2 = list(block_set)
                rng.shuffle(cur2)
                done = False
                for i in range(len(cur2)):
                    if time.time() > deadline - 8.0:
                        break
                    b1 = cur2[i]
                    if b1 not in block_set:
                        continue
                    u1 = [ts for ts in block_tsubs[b1] if cover_count[ts] == 1]
                    for j in range(i + 1, len(cur2)):
                        b2 = cur2[j]
                        if b2 not in block_set:
                            continue
                        u2 = [ts for ts in block_tsubs[b2] if cover_count[ts] == 1]
                        union = set(u1) | set(u2)
                        if not union:
                            continue
                        # candidate must cover all in union
                        min_ts = min(union, key=lambda x: len(tsub_blocks[x]))
                        candidates = set(tsub_blocks[min_ts])
                        for ts in union:
                            candidates &= set(tsub_blocks[ts])
                        candidates.discard(b1)
                        candidates.discard(b2)
                        candidates = [c for c in candidates if c not in block_set]
                        if not candidates:
                            continue
                        rng.shuffle(candidates)
                        for r in candidates[:15]:
                            # apply: remove b1,b2 add r
                            for ts in block_tsubs[b1]:
                                cover_count[ts] -= 1
                            for ts in block_tsubs[b2]:
                                cover_count[ts] -= 1
                            for ts in block_tsubs[r]:
                                cover_count[ts] += 1
                            if all(cover_count[ts] >= 1 for ts in block_tsubs[b1]) and \
                               all(cover_count[ts] >= 1 for ts in block_tsubs[b2]):
                                block_set.remove(b1)
                                block_set.remove(b2)
                                block_set.add(r)
                                improved = True
                                done = True
                                break
                            else:
                                for ts in block_tsubs[r]:
                                    cover_count[ts] -= 1
                                for ts in block_tsubs[b2]:
                                    cover_count[ts] += 1
                                for ts in block_tsubs[b1]:
                                    cover_count[ts] += 1
                        if done:
                            break
                    if done:
                        break

            # Phase 4: add best block if something became uncovered (shouldn't, but safety)
            if not improved:
                missing = [ts for ts in range(n_tsubs) if cover_count[ts] == 0]
                if missing:
                    best_b = -1
                    best_gain = 0
                    for b in range(n_blocks):
                        if b in block_set:
                            continue
                        g = sum(1 for ts in block_tsubs[b] if cover_count[ts] == 0)
                        if g > best_gain:
                            best_gain = g
                            best_b = b
                    if best_b != -1:
                        block_set.add(best_b)
                        for ts in block_tsubs[best_b]:
                            cover_count[ts] += 1
                        improved = True

        blocks = list(block_set)
        return blocks

    # ---- Main loop with restarts ----
    best_blocks = None
    best_len = float('inf')

    init = greedy_cover(0)
    if valid(init):
        best_blocks = init
        best_len = len(init)
    else:
        # fallback: brute force greedy until covered
        best_blocks = list(range(n_blocks))
        best_len = n_blocks

    if best_len <= KNOWN_BEST:
        return [list(all_blocks[b]) for b in best_blocks]

    refined = local_search(best_blocks, 1)
    if valid(refined) and len(refined) < best_len:
        best_blocks = refined
        best_len = len(refined)

    seed = 2
    while time.time() < deadline - 10.0:
        if best_len <= KNOWN_BEST:
            break
        perturbed = list(best_blocks)
        for _ in range(random.randint(2, 5)):
            if perturbed:
                perturbed.pop(random.randint(0, len(perturbed) - 1))
        refined = local_search(perturbed, seed)
        if valid(refined) and len(refined) < best_len:
            best_blocks = refined
            best_len = len(refined)
        seed += 1

    return [list(all_blocks[b]) for b in best_blocks]
