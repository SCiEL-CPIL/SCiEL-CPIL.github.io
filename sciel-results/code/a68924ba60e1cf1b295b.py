"""Superpermutation for n symbols - structural change: direct window-adjacency search."""
import time
import random
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115

    if n <= 6:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Standard recursive construction for S_6
        word = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new

        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)

        N = len(perms)

        # Build overlap matrix: overlap[i][j] = longest suffix of perms[i] matching prefix of perms[j]
        overlap_mat = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_mat[i, j] = 6
                    continue
                pj = perms[j]
                ov = 0
                for k in range(5, 0, -1):
                    if pi[6-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat[i, j] = ov

        # Convert to numpy for speed
        overlap_np = overlap_mat.astype(np.int32)

        # Build the word from a path of permutation indices
        # Each block is: perm + [7] + perm
        # But we need to handle overlaps between blocks
        def build_word_from_path(p):
            w = []
            for idx in p:
                perm = perms[idx]
                block = list(perm) + [7] + list(perm)
                overlap = 0
                min_len = min(len(w), len(block))
                for o in range(min_len, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if w[len(w) - o:] == block[:o]:
                        overlap = o
                        break
                w += block[overlap:]
            return "".join(map(str, w))

        # Calculate total overlap for a path (for optimization)
        def calc_total_overlap(path):
            if len(path) <= 1:
                return 0
            return int(np.sum(overlap_np[np.array(path[:-1]), np.array(path[1:])]))

        # 1. Naive path (standard order)
        naive_path = list(range(N))
        naive_word = build_word_from_path(naive_path)
        best_word = naive_word
        best_overlap = calc_total_overlap(naive_path)
        best_path = naive_path[:]

        # 2. Nearest Neighbor initialization
        random.seed(42)
        unvisited = list(range(N))
        random.shuffle(unvisited)
        path = [unvisited.pop()]

        while unvisited:
            last = path[-1]
            best_idx = -1
            best_ov = -1
            for idx in unvisited:
                ov = overlap_mat[last][idx]
                if ov > best_ov:
                    best_ov = ov
                    best_idx = idx
            path.append(best_idx)
            unvisited.remove(best_idx)

        current_overlap = calc_total_overlap(path)
        if current_overlap > best_overlap:
            best_overlap = current_overlap
            best_path = path[:]

        # 3. 2-opt improvement
        improved = True
        no_improve_count = 0

        while improved and time.time() < deadline - 25:
            improved = False
            no_improve_count += 1

            for i in range(N - 2):
                if time.time() > deadline - 25:
                    break
                for j in range(i + 1, N - 1):
                    if j == N - 1:
                        old_ov = overlap_mat[path[i]][path[i + 1]]
                        new_ov = overlap_mat[path[i]][path[j]]
                        if new_ov > old_ov:
                            path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                    else:
                        old_ov1 = overlap_mat[path[i]][path[i + 1]]
                        old_ov2 = overlap_mat[path[j]][path[j + 1]]
                        new_ov1 = overlap_mat[path[i]][path[j]]
                        new_ov2 = overlap_mat[path[i + 1]][path[j + 1]]
                        if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                            path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                            current_overlap += (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
                            improved = True
                            break
                if improved:
                    no_improve_count = 0
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = path[:]

            if not improved:
                # Random perturbation
                if no_improve_count > 5 and time.time() < deadline - 25:
                    i = random.randint(0, N - 3)
                    j = random.randint(i + 2, N - 1)
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    current_overlap = calc_total_overlap(path)
                    improved = True
                    no_improve_count = 0
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = path[:]

        # Build final word from best_path
        final_word = build_word_from_path(best_path)

        if len(final_word) < len(naive_word):
            return final_word
        else:
            return naive_word

    # Fallback for n > 7
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
