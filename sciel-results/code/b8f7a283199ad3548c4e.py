"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

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
        overlap_mat = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                for k in range(6, 0, -1):
                    if perms[i][6-k:] == perms[j][:k]:
                        overlap_mat[i][j] = k
                        break

        # Helper to calculate total overlap
        def calc_total_overlap(p):
            total = 0
            for i in range(len(p) - 1):
                total += overlap_mat[p[i]][p[i + 1]]
            return total

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

        # 2. Simulated Annealing Optimization with multi-start
        best_path = naive_path[:]
        best_overlap = current_overlap = best_overlap
        
        random.seed(42)
        
        # Run multiple SA restarts
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            if restart == 0:
                current_path = naive_path[:]
                current_overlap = best_overlap
            else:
                # Perturb best_path to create new starting point
                current_path = best_path[:]
                for _ in range(random.randint(10, 50)):
                    i = random.randint(0, N - 1)
                    j = random.randint(0, N - 1)
                    if i != j:
                        current_path[i], current_path[j] = current_path[j], current_path[i]
                current_overlap = calc_total_overlap(current_path)
            
            # SA Parameters
            T = 5.0
            cooling = 0.9997
            
            for step in range(500000):
                if time.time() > deadline - 20:
                    break
                
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue
                
                p_i = current_path[i]
                p_j = current_path[j]
                
                # Calculate Delta
                old_ov = 0
                new_ov = 0
                
                if abs(i - j) == 1:
                    # Adjacent swap
                    if i == 0:
                        old_ov += overlap_mat[p_i][p_j]
                        new_ov += overlap_mat[p_j][p_i]
                        if N > 2:
                            right = current_path[2]
                            old_ov += overlap_mat[p_j][right]
                            new_ov += overlap_mat[p_i][right]
                    elif i == N - 1:
                        left = current_path[N-2]
                        old_ov += overlap_mat[left][p_j]
                        new_ov += overlap_mat[left][p_i]
                        old_ov += overlap_mat[p_j][p_i]
                        new_ov += overlap_mat[p_i][p_j]
                    elif j == 0:
                        old_ov += overlap_mat[p_j][p_i]
                        new_ov += overlap_mat[p_i][p_j]
                        if N > 2:
                            right = current_path[2]
                            old_ov += overlap_mat[p_i][right]
                            new_ov += overlap_mat[p_j][right]
                    elif j == N - 1:
                        left = current_path[N-2]
                        old_ov += overlap_mat[left][p_i]
                        new_ov += overlap_mat[left][p_j]
                        old_ov += overlap_mat[p_i][p_j]
                        new_ov += overlap_mat[p_j][p_i]
                    else:
                        left = current_path[i-1]
                        right = current_path[i+1]
                        old_ov += overlap_mat[left][p_i]
                        old_ov += overlap_mat[p_i][p_j]
                        old_ov += overlap_mat[p_j][right]
                        new_ov += overlap_mat[left][p_j]
                        new_ov += overlap_mat[p_j][p_i]
                        new_ov += overlap_mat[p_i][right]
                else:
                    # Non-adjacent swap
                    if i > 0:
                        left_i = current_path[i-1]
                        old_ov += overlap_mat[left_i][p_i]
                        new_ov += overlap_mat[left_i][p_j]
                    if i < N - 1:
                        right_i = current_path[i+1]
                        old_ov += overlap_mat[p_i][right_i]
                        new_ov += overlap_mat[p_j][right_i]
                    
                    if j > 0:
                        left_j = current_path[j-1]
                        old_ov += overlap_mat[left_j][p_j]
                        new_ov += overlap_mat[left_j][p_i]
                    if j < N - 1:
                        right_j = current_path[j+1]
                        old_ov += overlap_mat[p_j][right_j]
                        new_ov += overlap_mat[p_i][right_j]
                
                delta = new_ov - old_ov
                
                if delta > 0:
                    current_path[i], current_path[j] = current_path[j], current_path[i]
                    current_overlap += delta
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_path = current_path[:]
                else:
                    if T > 1e-6:
                        if random.random() < math.exp(delta / T):
                            current_path[i], current_path[j] = current_path[j], current_path[i]
                            current_overlap += delta
                            if current_overlap > best_overlap:
                                best_overlap = current_overlap
                                best_path = current_path[:]
                
                T *= cooling

        # 3. Local search polish: 2-opt on best_path
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N - 1):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline - 5:
                        break
                    p_i = best_path[i]
                    p_j = best_path[j]
                    
                    old_ov = 0
                    new_ov = 0
                    
                    if j == i + 1:
                        # Adjacent
                        if i > 0:
                            left = best_path[i-1]
                            old_ov += overlap_mat[left][p_i]
                            new_ov += overlap_mat[left][p_j]
                        old_ov += overlap_mat[p_i][p_j]
                        new_ov += overlap_mat[p_j][p_i]
                        if j < N - 1:
                            right = best_path[j+1]
                            old_ov += overlap_mat[p_j][right]
                            new_ov += overlap_mat[p_i][right]
                    else:
                        # Non-adjacent
                        if i > 0:
                            left_i = best_path[i-1]
                            old_ov += overlap_mat[left_i][p_i]
                            new_ov += overlap_mat[left_i][p_j]
                        if i < N - 1:
                            right_i = best_path[i+1]
                            old_ov += overlap_mat[p_i][right_i]
                            new_ov += overlap_mat[p_j][right_i]
                        
                        if j > 0:
                            left_j = best_path[j-1]
                            old_ov += overlap_mat[left_j][p_j]
                            new_ov += overlap_mat[left_j][p_i]
                        if j < N - 1:
                            right_j = best_path[j+1]
                            old_ov += overlap_mat[p_j][right_j]
                            new_ov += overlap_mat[p_i][right_j]
                    
                    delta = new_ov - old_ov
                    if delta > 0:
                        best_path[i], best_path[j] = best_path[j], best_path[i]
                        best_overlap += delta
                        improved = True
        
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
