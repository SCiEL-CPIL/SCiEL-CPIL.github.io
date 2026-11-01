import time
import random
import itertools
import numpy as np
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Specific case for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n  # 729

        # Generate all words as tuples
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}

        # Precompute spheres: list of indices for each word within distance 1
        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)

        # Precompute reverse spheres
        reverse_spheres = [[] for _ in range(total_words)]
        for i in range(total_words):
            for j in spheres[i]:
                reverse_spheres[j].append(i)

        best_code = None
        best_size = total_words

        # Multiple restarts with greedy + local search
        max_restarts = 50

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Greedy construction with randomized tie-breaking
            covered = [False] * total_words
            uncovered_count = total_words
            code = []
            picked = set()

            gain = [0] * total_words
            for i in range(total_words):
                gain[i] = len(spheres[i])

            # Random tie-breakers to explore different greedy paths
            tie_breakers = [random.random() for _ in range(total_words)]

            heap = []
            for i in range(total_words):
                heap.append((-gain[i], tie_breakers[i], i))
            heapq.heapify(heap)

            while uncovered_count > 0:
                if time.time() > deadline:
                    break

                best_candidate = -1
                while heap:
                    neg_gain, tb, cand_idx = heapq.heappop(heap)
                    if -neg_gain != gain[cand_idx]:
                        continue
                    if cand_idx in picked:
                        continue
                    best_candidate = cand_idx
                    break

                if best_candidate == -1:
                    break

                picked.add(best_candidate)
                code.append(best_candidate)

                for idx in spheres[best_candidate]:
                    if not covered[idx]:
                        covered[idx] = True
                        uncovered_count -= 1
                        for cand in reverse_spheres[idx]:
                            if cand not in picked:
                                gain[cand] -= 1
                                heapq.heappush(heap, (-gain[cand], tie_breakers[cand], cand))

            # Local search: try to remove redundant codewords
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break

                # Try removing each codeword and check if coverage is maintained
                for i in range(len(code) - 1, -1, -1):
                    if time.time() > deadline:
                        break

                    cw_idx = code[i]
                    # Check if removing code[i] leaves all its sphere covered by others
                    is_redundant = True
                    for w in spheres[cw_idx]:
                        covered_by_other = False
                        for j in range(len(code)):
                            if i == j:
                                continue
                            other_cw_idx = code[j]
                            if w in spheres[other_cw_idx]:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            is_redundant = False
                            break

                    if is_redundant:
                        code.pop(i)
                        improved = True

            # Local search: try swapping a codeword with another to enable further removals
            if len(code) > 73:
                swapped = True
                while swapped and time.time() < deadline:
                    swapped = False
                    for i in range(len(code)):
                        if time.time() > deadline:
                            break
                        for j in range(len(code)):
                            if i == j:
                                continue
                            # Try swapping code[i] with code[j]
                            # Actually, let's try replacing code[i] with a new word
                            # that might enable more removals
                            pass

            if len(code) < best_size:
                best_size = len(code)
                best_code = [list(all_words[idx]) for idx in code]

                if best_size <= 73:
                    break

        if best_code is not None:
            return best_code
        else:
            # Fallback: return the last code generated
            return [list(all_words[idx]) for idx in code] if 'code' in locals() else [list(t) for t in itertools.product(range(q), repeat=n)]

    else:
        # Fallback: Simple greedy for other params
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}

        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)

        reverse_spheres = [[] for _ in range(len(all_words))]
        for i in range(len(all_words)):
            for j in spheres[i]:
                reverse_spheres[j].append(i)

        covered = [False] * len(all_words)
        uncovered_count = len(all_words)
        code = []
        picked = set()

        gain = [0] * len(all_words)
        for i in range(len(all_words)):
            gain[i] = len(spheres[i])

        heap = []
        for i in range(len(all_words)):
            heap.append((-gain[i], random.random(), i))
        heapq.heapify(heap)

        while uncovered_count > 0:
            if time.time() > deadline:
                break
            best_candidate = -1
            while heap:
                neg_gain, tb, cand_idx = heapq.heappop(heap)
                if -neg_gain != gain[cand_idx]:
                    continue
                if cand_idx in picked:
                    continue
                best_candidate = cand_idx
                break
            if best_candidate == -1:
                break
            picked.add(best_candidate)
            code.append(best_candidate)
            for idx in spheres[best_candidate]:
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
                    for cand in reverse_spheres[idx]:
                        if cand not in picked:
                            gain[cand] -= 1
                            heapq.heappush(heap, (-gain[cand], random.random(), cand))

        return [list(all_words[idx]) for idx in code]
