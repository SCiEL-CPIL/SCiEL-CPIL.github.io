import numpy as np
import itertools
import time
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # ---- 1. Enumerate the whole space and precompute structure ----
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # balls[i]: bitset of words within `radius` of words[i]
    balls = [0] * N
    ball_lists = [[] for _ in range(N)]
    for i, w in enumerate(words):
        mask = 0
        w_list = list(w)
        mask |= (1 << i)
        ball_lists[i].append(i)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                mask |= (1 << idx)
                ball_lists[i].append(idx)
                w_list[pos] = orig_val
        balls[i] = mask

    ball_size = len(ball_lists[0])

    # cover_map[w]: list of codewords that cover word w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in ball_lists[c]:
            cover_map[w].append(c)

    # ball_sets[c]: set of words covered by codeword c (for fast swap membership)
    ball_sets = [set(b) for b in ball_lists]

    UNIVERSE = (1 << N) - 1

    # ---- 2. Heap-based greedy set cover (incremental gain) ----
    def greedy_cover(uncovered_mask, start_code):
        """Cover uncovered_mask, starting from start_code (list of codewords).
        Returns (code_list, counts, uncovered_mask)."""
        counts = [0] * N
        covered_mask = 0
        code = []
        code_set = set()
        for c in start_code:
            code.append(c)
            code_set.add(c)
            m = balls[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
            covered_mask |= balls[c]

        uncovered = uncovered_mask & ~covered_mask
        uncovered_count = uncovered.bit_count()
        if uncovered_count == 0:
            return code, counts, uncovered

        # gain[i] = number of currently-uncovered words in ball i
        gain = [0] * N
        heap = []
        m = uncovered
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            for c in cover_map[w]:
                gain[c] += 1
            m ^= lsb
        for i in range(N):
            if gain[i] > 0:
                heapq.heappush(heap, (-gain[i], i))

        while uncovered_count > 0:
            if time.time() > deadline - 2:
                break
            # pop valid candidate
            c = -1
            while heap:
                neg, i = heapq.heappop(heap)
                if gain[i] == -neg and gain[i] > 0:
                    c = i
                    break
            if c < 0:
                break

            code.append(c)
            code_set.add(c)
            m = balls[c] & uncovered
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                counts[w] += 1
                uncovered_count -= 1
                for c2 in cover_map[w]:
                    gain[c2] -= 1
                m ^= lsb
            uncovered &= ~balls[c]
            for i in range(N):
                if gain[i] > 0:
                    heapq.heappush(heap, (-gain[i], i))

        return code, counts, uncovered

    # ---- 3. Polish: repeatedly delete redundant codewords ----
    def polish(code, counts):
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                redundant = True
                m = balls[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    code.pop(i)
                    m = balls[c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    changed = True

    # ---- 4. Local search with deletion + 1-swap + 2-swap ----
    def local_search(code, counts, time_limit):
        if not code:
            return code, counts
        in_code = [False] * N
        for c in code:
            in_code[c] = True

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code[i]

                # tentatively remove
                newly_uncovered = []
                m = balls[c_remove]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    counts[w] -= 1
                    if counts[w] == 0:
                        newly_uncovered.append(w)
                    m ^= lsb

                if not newly_uncovered:
                    # pure deletion
                    code.pop(i)
                    in_code[c_remove] = False
                    improved = True
                    break

                # try 1-swap: single codeword covering all newly_uncovered
                found = -1
                if len(newly_uncovered) <= 6:
                    crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                    cand_pool = None
                    for w in crit_sorted:
                        cands_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                        if cand_pool is None:
                            cand_pool = set(cands_w)
                        else:
                            cand_pool.intersection_update(cands_w)
                        if not cand_pool:
                            break
                    if cand_pool:
                        bs_check = None
                        for c_add in cand_pool:
                            bs = ball_sets[c_add]
                            if all(w in bs for w in newly_uncovered):
                                found = c_add
                                break

                if found >= 0:
                    code[i] = found
                    in_code[c_remove] = False
                    in_code[found] = True
                    m = balls[found]
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        counts[w] += 1
                        m ^= lsb
                    improved = True
                    break
                else:
                    # restore counts
                    for w in newly_uncovered:
                        counts[w] += 1

                # try 2-swap: remove c_remove + one other, add two
                if len(newly_uncovered) <= 10 and len(code) >= 2:
                    # candidate pool from the most constrained uncovered word
                    w0 = min(newly_uncovered, key=lambda w: len(cover_map[w]))
                    pool = [c2 for c2 in cover_map[w0]
                            if not in_code[c2] and c2 != c_remove]
                    if len(pool) > 1:
                        if len(pool) > 40:
                            pool = random.sample(pool, 40)
                        # build candidate add-sets of size 2 covering all newly_uncovered
                        add_sets = []
                        p = len(pool)
                        for a in range(p):
                            ca = pool[a]
                            bsa = ball_sets[ca]
                            for b in range(a + 1, p):
                                cb = pool[b]
                                bsb = ball_sets[cb]
                                ok = True
                                for w in newly_uncovered:
                                    if w not in bsa and w not in bsb:
                                        ok = False
                                        break
                                if ok:
                                    add_sets.append((ca, cb))
                        if add_sets:
                            # choose which codeword to also remove (j != i)
                            jidx = list(range(len(code)))
                            jidx.remove(i)
                            random.shuffle(jidx)
                            done = False
                            for j in jidx:
                                if time.time() > time_limit:
                                    break
                                c_rem2 = code[j]
                                # combined uncovered if we remove both c_remove and c_rem2
                                # count after removing c_remove already applied; now also remove c_rem2
                                still_uncovered = []
                                m = balls[c_rem2]
                                while m:
                                    lsb = m & -m
                                    w = lsb.bit_length() - 1
                                    counts[w] -= 1
                                    if counts[w] == 0:
                                        still_uncovered.append(w)
                                    m ^= lsb
                                # need add-set to cover all still_uncovered
                                good = None
                                for (ca, cb) in add_sets:
                                    ok = True
                                    for w in still_uncovered:
                                        if w not in ball_sets[ca] and w not in ball_sets[cb]:
                                            ok = False
                                            break
                                    if ok:
                                        good = (ca, cb)
                                        break
                                if good is not None:
                                    ca, cb = good
                                    # commit: remove both, add both
                                    code.pop(j)
                                    code.pop(i)
                                    in_code[c_rem2] = False
                                    in_code[c_remove] = False
                                    # restore counts for c_rem2's ball (we decremented)
                                    m = balls[c_rem2]
                                    while m:
                                        lsb = m & -m
                                        w = lsb.bit_length() - 1
                                        counts[w] += 1
                                        m ^= lsb
                                    # add ca, cb
                                    for c_add in (ca, cb):
                                        code.append(c_add)
                                        in_code[c_add] = True
                                        m = balls[c_add]
                                        while m:
                                            lsb = m & -m
                                            w = lsb.bit_length() - 1
                                            counts[w] += 1
                                            m ^= lsb
                                    improved = True
                                    done = True
                                    break
                                else:
                                    # restore c_rem2 counts
                                    m = balls[c_rem2]
                                    while m:
                                        lsb = m & -m
                                        w = lsb.bit_length() - 1
                                        counts[w] += 1
                                        m ^= lsb
                                if done:
                                    break
                            if done:
                                break

        return code, counts

    # ---- 5. Main driver ----
    # Initial greedy cover from scratch
    best_code, best_counts, _ = greedy_cover(UNIVERSE, [])
    polish(best_code, best_counts)
    best_size = len(best_code)

    # Local search to polish the initial solution
    best_code, best_counts = local_search(best_code, best_counts, deadline - 8)
    polish(best_code, best_counts)
    best_size = len(best_code)

    # Kick-repair loop
    kick_sizes = [1, 2, 3, 4, 5]
    while time.time() < deadline - 8:
        k = random.choice(kick_sizes)
        if len(best_code) <= k:
            continue

        curr_code = list(best_code)
        curr_counts = best_counts[:]

        # Kick: remove k random codewords
        code_list = list(curr_code)
        random.shuffle(code_list)
        to_remove = code_list[:k]
        to_remove_set = set(to_remove)
        new_code = [c for c in curr_code if c not in to_remove_set]
        for c in to_remove:
            m = balls[c]
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                curr_counts[w] -= 1
                m ^= lsb

        # Find uncovered words
        uncovered = 0
        for w in range(N):
            if curr_counts[w] == 0:
                uncovered |= (1 << w)

        # Repair with heap-greedy
        if uncovered:
            new_code, curr_counts, _ = greedy_cover(uncovered, new_code)

        # Polish
        polish(new_code, curr_counts)

        # Local search
        new_code, curr_counts = local_search(new_code, curr_counts, deadline - 3)
        polish(new_code, curr_counts)

        if len(new_code) < best_size:
            best_code = new_code
            best_counts = curr_counts
            best_size = len(best_code)

    return [list(words[i]) for i in best_code]
