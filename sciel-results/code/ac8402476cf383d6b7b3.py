import time
import random
import sys
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit

    if n <= 6:
        # Standard recursive construction is optimal for n <= 6
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
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Step 1: Build S_6 standard construction to get the list of permutations
        word6 = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word6) - m + 2):
                window = tuple(word6[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word6 = new
        
        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word6) - 5):
            window = tuple(word6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms) # Should be 720
        
        # Precompute overlap matrix
        M = [[0]*N for _ in range(N)]
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j:
                    M[i][j] = 6
                    continue
                q = perms[j]
                best = 0
                for k in range(5, 0, -1):
                    if p[-k:] == q[:k]:
                        best = k
                        break
                M[i][j] = best
        
        import numpy as np
        M_np = np.array(M, dtype=np.int8)

        def calc_score(path):
            total = 0
            for i in range(len(path) - 1):
                total += M[path[i]][path[i+1]]
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

        # --- Improved Search Strategy ---
        
        # 1. Greedy Initialization
        used = [False] * N
        path = [random.randint(0, N-1)]
        used[path[0]] = True
        
        for _ in range(N - 1):
            last = path[-1]
            best_next = -1
            best_overlap = -1
            for j in range(N):
                if not used[j]:
                    overlap = M[last][j]
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_next = j
            path.append(best_next)
            used[best_next] = True
            
        current_path = path
        current_score = calc_score(current_path)
        best_path = current_path[:]
        best_score = current_score

        # 2. Simulated Annealing with NumPy-accelerated 2-opt
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < deadline - 2:
            if T < min_T:
                break
            
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            # Calculate delta
            delta = 0
            
            # Boundary edges
            if i > 0:
                delta -= M[current_path[i-1]][current_path[i]]
                delta += M[current_path[i-1]][current_path[j]]
            if j < N - 1:
                delta -= M[current_path[j]][current_path[j+1]]
                delta += M[current_path[i]][current_path[j+1]]
            
            # Internal edges
            if j - i < 20:
                sum_old = 0
                sum_new = 0
                for k in range(i, j):
                    sum_old += M[current_path[k]][current_path[k+1]]
                    sum_new += M[current_path[k+1]][current_path[k]]
                delta += sum_new - sum_old
            else:
                seg = np.array(current_path[i:j+1])
                idx1 = seg[:-1]
                idx2 = seg[1:]
                sum_old = np.sum(M_np[idx1, idx2])
                sum_new = np.sum(M_np[idx2, idx1])
                delta += sum_new - sum_old
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                # Accept move
                current_path[i:j+1] = reversed(current_path[i:j+1])
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_path = current_path[:]
            
            T *= cooling

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
            overlap = 0
            min_len = min(len(new), len(block))
            for o in range(min_len, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
