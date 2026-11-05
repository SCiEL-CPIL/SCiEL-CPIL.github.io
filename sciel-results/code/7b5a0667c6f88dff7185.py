# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

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

        # Precompute overlap matrix
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = 6
                    continue
                q = perms[j]
                best = 0
                for k in range(5, 0, -1):
                    if p[-k:] == q[:k]:
                        best = k
                        break
                ov[i][j] = best

        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total

        def build_word_from_path(path):
            new_word = []
            for idx in path:
                p = perms[idx]
                block = list(p) + [7] + list(p)
                overlap = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            return new_word

        # Initialize with standard order
        best_path = list(range(N))
        best_overlap = calc_total_overlap(best_path)
        current_path = best_path[:]
        current_overlap = best_overlap

        # Local Search with 1-Opt, 2-Opt-Relocate, and 2-Opt-Reverse (with precomputed prefix sums)
        
        # Precompute prefix sums for forward and reverse overlaps to enable fast 2-Opt-Reverse evaluation
        fwd_prefix = [0] * (N + 1)
        for i in range(N - 1):
            fwd_prefix[i + 1] = fwd_prefix[i] + ov[current_path[i]][current_path[i+1]]
        
        bwd_prefix = [0] * (N + 1)
        for i in range(N - 1):
            bwd_prefix[i + 1] = bwd_prefix[i] + ov[current_path[i+1]][current_path[i]]

        def update_prefixes(path):
            nonlocal fwd_prefix, bwd_prefix
            fwd_prefix = [0] * (N + 1)
            for i in range(N - 1):
                fwd_prefix[i + 1] = fwd_prefix[i] + ov[path[i]][path[i+1]]
            bwd_prefix = [0] * (N + 1)
            for i in range(N - 1):
                bwd_prefix[i + 1] = bwd_prefix[i] + ov[path[i+1]][path[i]]

        def get_fwd_sum(i, j):
            if i >= j:
                return 0
            return fwd_prefix[j] - fwd_prefix[i]
        
        def get_bwd_sum(i, j):
            if i >= j:
                return 0
            return bwd_prefix[j] - bwd_prefix[i]

        improved = True
        passes = 0
        max_passes = 500 # Limit to ensure time
        
        while improved and time.time() < deadline - 5 and passes < max_passes:
            improved = False
            passes += 1
            
            # 1. 2-Opt (Reverse segment) - Best Improvement
            best_gain = 0
            best_i = -1
            best_j = -1
            
            for i in range(N - 1):
                left_idx = i - 1
                for j in range(i + 1, N):
                    right_idx = j + 1
                    
                    old_cost = 0
                    if left_idx != -1:
                        old_cost += ov[current_path[left_idx]][current_path[i]]
                    old_cost += get_fwd_sum(i, j)
                    if right_idx != N:
                        old_cost += ov[current_path[j]][current_path[right_idx]]
                    
                    new_cost = 0
                    if left_idx != -1:
                        new_cost += ov[current_path[left_idx]][current_path[j]]
                    new_cost += get_bwd_sum(i, j)
                    if right_idx != N:
                        new_cost += ov[current_path[i]][current_path[right_idx]]
                    
                    gain = new_cost - old_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                        best_j = j
            
            if best_gain > 0:
                current_path[best_i:best_j+1] = reversed(current_path[best_i:best_j+1])
                current_overlap += best_gain
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = current_path[:]
                update_prefixes(current_path)
                improved = True
                continue # Restart pass to re-evaluate all moves

            # 2. 1-Opt (Relocate single element) - Best Improvement
            best_gain = 0
            best_i = -1
            best_j = -1
            for i in range(N):
                left = current_path[i-1] if i > 0 else -1
                right = current_path[i+1] if i < N - 1 else -1
                
                gain_remove = 0
                if left != -1 and right != -1:
                    gain_remove = ov[left][right] - ov[left][i] - ov[i][right]
                elif left != -1:
                    gain_remove = -ov[left][i]
                elif right != -1:
                    gain_remove = -ov[i][right]
                else:
                    gain_remove = 0
                    
                item = current_path[i]
                
                for j in range(N):
                    if j == i or j == i + 1:
                        continue
                    
                    if j < i:
                        left_orig = j - 1 if j > 0 else -1
                        right_orig = j
                    else:
                        left_orig = j
                        right_orig = j + 1 if j < N - 1 else -1

                    gain_insert = 0
                    if left_orig != -1 and right_orig != -1:
                        gain_insert = ov[left_orig][item] + ov[item][right_orig] - ov[left_orig][right_orig]
                    elif left_orig != -1:
                        gain_insert = ov[left_orig][item]
                    elif right_orig != -1:
                        gain_insert = ov[item][right_orig]
                    else:
                        gain_insert = 0
                    
                    total_gain = gain_remove + gain_insert
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_i = i
                        best_j = j
            
            if best_gain > 0:
                i = best_i
                j = best_j
                item = current_path[i]
                del current_path[i]
                if i < j:
                    j -= 1
                current_path.insert(j, item)
                current_overlap += best_gain
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = current_path[:]
                update_prefixes(current_path)
                improved = True
                continue

            # 3. 2-Opt-Relocate (Move block of 2)
            if not improved:
                best_gain = 0
                best_i = -1
                best_j = -1
                for i in range(N - 1):
                    block = current_path[i:i+2]
                    left = current_path[i-1] if i > 0 else -1
                    right = current_path[i+2] if i + 2 < N else -1
                    
                    gain_remove = 0
                    if left != -1 and right != -1:
                        gain_remove = ov[left][right] - ov[left][block[0]] - ov[block[0]][block[1]] - ov[block[1]][right]
                    elif left != -1:
                        gain_remove = -ov[left][block[0]] - ov[block[0]][block[1]]
                    elif right != -1:
                        gain_remove = -ov[block[0]][block[1]] - ov[block[1]][right]
                    else:
                        gain_remove = -ov[block[0]][block[1]]
                    
                    for j in range(N - 1):
                        if j == i or j == i + 1:
                            continue
                        
                        if j < i:
                            left_orig = j - 1 if j > 0 else -1
                            right_orig = j
                        else:
                            left_orig = j + 1
                            right_orig = j + 2 if j < N - 2 else -1
                            
                        gain_insert = 0
                        if left_orig != -1 and right_orig != -1:
                            gain_insert = ov[left_orig][block[0]] + ov[block[0]][block[1]] + ov[block[1]][right_orig] - ov[left_orig][right_orig]
                        elif left_orig != -1:
                            gain_insert = ov[left_orig][block[0]] + ov[block[0]][block[1]]
                        elif right_orig != -1:
                            gain_insert = ov[block[0]][block[1]] + ov[block[1]][right_orig]
                        else:
                            gain_insert = ov[block[0]][block[1]]
                        
                        total_gain = gain_remove + gain_insert
                        if total_gain > best_gain:
                            best_gain = total_gain
                            best_i = i
                            best_j = j
                
                if best_gain > 0:
                    i = best_i
                    j = best_j
                    block = current_path[i:i+2]
                    del current_path[i:i+2]
                    if i < j:
                        j -= 2
                    current_path.insert(j, block[0])
                    current_path.insert(j+1, block[1])
                    current_overlap += best_gain
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = current_path[:]
                    update_prefixes(current_path)
                    improved = True

        # Build final word
        final_word = build_word_from_path(best_path)
        return "".join(map(str, final_word))
    
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
