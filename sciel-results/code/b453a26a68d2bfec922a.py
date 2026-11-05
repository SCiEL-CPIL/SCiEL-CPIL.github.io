import numpy as np
import itertools
import time
import random
import heapq
from collections import deque

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # General setup (works for q=3,n=7,r=1 but also general)
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Precompute balls (neighbors including self)
    balls = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        balls.append(neigh)

    ball_size = len(balls[0])  # 1 + n*(q-1) = 15 for q=3,n=7

    # Precompute cover_map: for each word w, list of codewords whose ball contains w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # Ball sets for fast membership
    ball_sets = [set(b) for b in balls]
    # Ball as numpy arrays for vectorized count updates
    ball_arrays = [np.array(b, dtype=np.int32) for b in balls]

    # ---------- Greedy set cover with heap and dynamic candidate counts ----------
    def greedy_cover(seed, start_code=None):
        rng = random.Random(seed)
        covered = [False] * N
        # cnt[c] = number of currently-uncovered words in ball(c)
        cnt = [ball_size] * N
        code = []
        in_code = [False] * N
        uncovered_count = N

        if start_code:
            for c in start_code:
                code.append(c)
                in_code[c] = True
                for w in balls[c]:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1

        # Max-heap via negative gain; lazy deletion by checking cnt
        heap = []
        for i in range(N):
            if not in_code[i]:
                heap.append((-cnt[i], i))
        heapq.heapify(heap)

        while uncovered_count > 0:
            # Pop until we find a valid candidate
            valid = None
            while heap:
                neg_val, i = heapq.heappop(heap)
                if cnt[i] == -neg_val and not in_code[i] and cnt[i] > 0:
                    valid = i
                    break
            if valid is None:
                break

            c = valid
            code.append(c)
            in_code[c] = True

            for w in balls[c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    for c2 in cover_map[w]:
                        cnt[c2] -= 1
                        if cnt[c2] > 0 and not in_code[c2]:
                            heapq.heappush(heap, (-cnt[c2], c2))

        return code

    # ---------- Refine: remove redundant codewords ----------
    def refine(code):
        if not code:
            return code
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count[ball_arrays[c]] += 1

        code_list = list(code)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                if np.min(cover_count[ball_arrays[c]]) > 1:
                    cover_count[ball_arrays[c]] -= 1
                    code_list.pop(i)
                    changed = True
                    break
        return code_list

    # ---------- Local search ----------
    def local_search(code, seed, time_limit):
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[ball_arrays[c]] += 1

        in_code = [False] * N
        for c in code_list:
            in_code[c] = True

        def try_delete(c, i):
            """Try to delete codeword c at index i. Return True if deleted."""
            newly_uncovered = []
            for w in ball_arrays[c]:
                cover_count[w] -= 1
                if cover_count[w] == 0:
                    newly_uncovered.append(w)
            if not newly_uncovered:
                code_list.pop(i)
                in_code[c] = False
                return True
            # Roll back
            for w in newly_uncovered:
                cover_count[w] += 1
            return False

        def try_swap(c_remove, i):
            """Try to replace c_remove with a single codeword covering all newly-uncovered."""
            newly_uncovered = []
            for w in ball_arrays[c_remove]:
                cover_count[w] -= 1
                if cover_count[w] == 0:
                    newly_uncovered.append(w)
            if not newly_uncovered:
                code_list.pop(i)
                in_code[c_remove] = False
                return True  # deletion succeeded

            # Build candidate pool via intersection over rarest critical word
            crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
            cand_pool = None
            for w in crit_sorted:
                candidates_w = set(cover_map[w])
                candidates_w.discard(c_remove)
                if cand_pool is None:
                    cand_pool = candidates_w
                else:
                    cand_pool.intersection_update(candidates_w)
                if not cand_pool:
                    break

            if cand_pool:
                # Score candidates: prefer those that maximize redundancy of words with count==1
                # (i.e., words that would become singly-covered after swap)
                best_c = -1
                best_score = -1
                for c2 in cand_pool:
                    if in_code[c2]:
                        continue
                    # Check c2 covers all newly_uncovered
                    if not all(w in ball_sets[c2] for w in newly_uncovered):
                        continue
                    # Score: number of words in ball(c2) that currently have count==1
                    # (these would become count==2 after adding c2, increasing redundancy)
                    score = int(np.sum(cover_count[ball_arrays[c2]] == 1))
                    if score > best_score:
                        best_score = score
                        best_c = c2

                if best_c != -1:
                    code_list[i] = best_c
                    in_code[c_remove] = False
                    in_code[best_c] = True
                    for w in ball_arrays[best_c]:
                        cover_count[w] += 1
                    return True

            # Roll back
            for w in newly_uncovered:
                cover_count[w] += 1
            return False

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)

            # --- Deletion + 1-swap pass ---
            for i in indices:
                if time.time() > time_limit:
                    break
                c = code_list[i]
                if try_delete(c, i):
                    improved = True
                    break
                if try_swap(c, i):
                    improved = True
                    break

            if improved:
                continue

            # --- 2-for-1 replacement pass ---
            K = len(code_list)
            if K >= 2:
                num_pairs = min(300, K * (K - 1) // 2)
                found = False
                for _ in range(num_pairs):
                    if time.time() > time_limit:
                        break
                    i, j = rng.sample(range(K), 2)
                    if i == j:
                        continue
                    c1 = code_list[i]
                    c2 = code_list[j]

                    # Temporarily remove c1, c2
                    union = set(balls[c1]).union(balls[c2])
                    newly_uncovered = []
                    for w in union:
                        cover_count[w] -= 2
                    for w in union:
                        if cover_count[w] < 0:
                            cover_count[w] = 0
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)

                    if not newly_uncovered:
                        # Both redundant
                        code_list.pop(j)
                        code_list.pop(i)
                        in_code[c1] = False
                        in_code[c2] = False
                        improved = True
                        found = True
                        break

                    # Find c3 covering all newly_uncovered
                    crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                    cand_pool = None
                    for w in crit_sorted:
                        candidates_w = set(cover_map[w])
                        candidates_w.discard(c1)
                        candidates_w.discard(c2)
                        if cand_pool is None:
                            cand_pool = candidates_w
                        else:
                            cand_pool.intersection_update(candidates_w)
                        if not cand_pool:
                            break

                    if cand_pool:
                        best_c3 = -1
                        best_score = -1
                        for c3 in cand_pool:
                            if in_code[c3]:
                                continue
                            if not all(w in ball_sets[c3] for w in newly_uncovered):
                                continue
                            score = int(np.sum(cover_count[ball_arrays[c3]] <= 1))
                            if score > best_score:
                                best_score = score
                                best_c3 = c3
                        if best_c3 != -1:
                            code_list.pop(j)
                            code_list.pop(i)
                            in_code[c1] = False
                            in_code[c2] = False
                            code_list.append(best_c3)
                            in_code[best_c3] = True
                            for w in ball_arrays[best_c3]:
                                cover_count[w] += 1
                            improved = True
                            found = True
                            break

                    # Roll back
                    for w in union:
                        cover_count[w] += 2
                    if not found:
                        continue

                if found:
                    continue

        return code_list

    # ---------- Peeling polish: greedy max-min redundancy removal ----------
    def peeling_polish(code, time_limit):
        """
        Iteratively remove codewords that are 'safely' redundant, prioritizing
        removals that maximize the minimum resulting cover_count (peeling by margin).
        """
        if not code:
            return code
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count[ball_arrays[c]] += 1

        code_list = list(code)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            best_idx = -1
            best_margin = 0
            for i, c in enumerate(code_list):
                # min cover_count in ball(c) after removal
                min_after = int(np.min(cover_count[ball_arrays[c]] - 1))
                if min_after > best_margin:
                    best_margin = min_after
                    best_idx = i
            if best_idx >= 0:
                c = code_list[best_idx]
                cover_count[ball_arrays[c]] -= 1
                code_list.pop(best_idx)
                changed = True
        return code_list

    # ---------- Main driver ----------
    best_code = None
    best_size = float('inf')

    num_starts = 10
    for seed in range(num_starts):
        if time.time() > deadline - 45:
            break

        code = greedy_cover(seed)
        code = refine(code)

        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

        # Local search
        refined = local_search(code, seed, deadline - 15)
        refined = refine(refined)

        if len(refined) < best_size:
            best_code = refined
            best_size = len(refined)

    # Final peeling polish on best solution
    if best_code is not None and time.time() < deadline - 10:
        polished = peeling_polish(best_code, deadline - 5)
        if len(polished) < best_size:
            best_code = polished
            best_size = len(polished)

    # One more local search + polish cycle if time allows
    if best_code is not None and time.time() < deadline - 15:
        refined2 = local_search(best_code, 999, deadline - 10)
        refined2 = refine(refined2)
        polished2 = peeling_polish(refined2, deadline - 5)
        if len(polished2) < best_size:
            best_code = polished2
            best_size = len(polished2)

    if best_code is None:
        # Fallback: trivial cover (all words) — valid but large
        best_code = list(range(N))

    # Verify coverage (defensive)
    covered = np.zeros(N, dtype=bool)
    for c in best_code:
        covered[ball_arrays[c]] = True
    if not covered.all():
        # Add missing words as codewords (shouldn't happen)
        missing = np.where(~covered)[0]
        best_code.extend(missing.tolist())

    return [list(words[i]) for i in best_code]
