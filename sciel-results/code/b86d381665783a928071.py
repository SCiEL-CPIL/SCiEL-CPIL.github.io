"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

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
        # Build S_6 standard construction
        word = [1]
        for m in range(2, 7):
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

        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)

        N = len(perms)

        # Precompute overlap matrix using NumPy for speed
        # overlap[i][j] = longest suffix of perms[i] that is prefix of perms[j]
        perm_array = np.array(perms, dtype=np.int8)  # shape (N, 6)
        
        overlap_mat = np.zeros((N, N), dtype=np.int8)
        for k in range(1, 7):
            # For overlap of length k: suffix of length k of perm i == prefix of length k of perm j
            suffixes = perm_array[:, 6-k:]  # shape (N, k)
            prefixes = perm_array[:, :k]     # shape (N, k)
            # For each k, find which pairs have matching suffix/prefix
            # This is expensive to do naively, but N=720 is small
            # Use a dictionary approach for each k
            for k_val in range(1, 7):
                if k_val != k:
                    continue
                suffix_dict = {}
                for i in range(N):
                    key = tuple(perms[i][6-k_val:])
                    suffix_dict[key] = i
                for j in range(N):
                    prefix_key = tuple(perms[j][:k_val])
                    if prefix_key in suffix_dict:
                        i = suffix_dict[prefix_key]
                        if i != j:
                            overlap_mat[i, j] = max(overlap_mat[i, j], k_val)
            if k == 6:
                # Diagonal: full overlap
                np.fill_diagonal(overlap_mat, 6)
                break

        # Actually, let me redo this more carefully
        overlap_mat = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_mat[i, j] = 6
                    continue
                pj = perms[j]
                ov = 0
                for k in range(5, -1, -1):
                    if k == 0:
                        break
                    if pi[6-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat[i, j] = ov

        # Convert to numpy for faster operations
        overlap_np = overlap_mat.astype(np.int32)

        # Helper to calculate total overlap for a path
        def calc_total_overlap(path):
            if len(path) <= 1:
                return 0
            return int(np.sum(overlap_np[np.array(path[:-1]), np.array(path[1:])]))

        # Helper to build word from path
        def build_word_from_path(p):
            w = []
            for idx in p:
                perm = perms[idx]
                block = list(perm) + [7] + list(perm)
                overlap = 0
                min_len = min(len(w), len(block))
                for o in range(min_len, -1, -1):
                    if w[len(w) - o:] == block[:o]:
                        overlap = o
                        break
                w += block[overlap:]
            return "".join(map(str, w))

        # 1. Generate Naive Word (standard order)
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

        # 3. 2-opt improvement with efficient delta calculation
        # For 2-opt: reversing path[i+1..j] changes two edges
        # Old: (path[i], path[i+1]) and (path[j], path[j+1]) [if j < N-1]
        # New: (path[i], path[j]) and (path[i+1], path[j+1]) [if j < N-1]
        
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline - 20:
            improved = False
            no_improve_count += 1
            
            # 2-Opt
            for i in range(N - 2):
                if time.time() > deadline - 20:
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
            
            # Or-Opt: move segment of length 1 or 2
            if not improved and time.time() < deadline - 20:
                for L in [1, 2]:
                    if time.time() > deadline - 20:
                        break
                    for i in range(N - L):
                        if time.time() > deadline - 20:
                            break
                        # Segment: path[i : i+L]
                        seg = path[i:i+L]
                        
                        # Remove segment
                        if i > 0:
                            left_neighbor = path[i-1]
                        else:
                            left_neighbor = None
                        if i + L < N:
                            right_neighbor = path[i+L]
                        else:
                            right_neighbor = None
                        
                        # Try inserting at each position
                        for j in range(N - L + 1):
                            if j == i or (i > 0 and j == i + L - 1):
                                continue
                            if i > 0 and j >= i and j <= i + L - 1:
                                continue
                            
                            # Calculate delta
                            # Remove edges: (left, seg[0]) and (seg[-1], right) if they exist
                            # Add edges: (new_left, seg[0]) and (seg[-1], new_right)
                            
                            # This is complex; skip for now and rely on 2-opt
                            pass

            if not improved:
                # Try a random perturbation
                if no_improve_count > 5 and time.time() < deadline - 20:
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
        
        # Compare with naive
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
# EVOLVE-BLOCK-END
