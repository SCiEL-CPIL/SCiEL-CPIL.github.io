import itertools
import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit
    
    # For n <= 6, use standard recursive construction
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
        
    # For n=7, use TSP heuristic on S6 permutations
    # Generate S6 standard superpermutation
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word6 = new
        
    # Extract all 720 permutations of 1..6 in order of appearance
    perms6, seen = [], set()
    for i in range(len(word6) - 5):
        window = tuple(word6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms6.append(window)
            
    N = len(perms6)  # Should be 720
    
    # Precompute overlap matrix using numpy for speed
    # overlap[i][j] = longest suffix of perms6[i] that is prefix of perms6[j]
    overlap_mat = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        pi = perms6[i]
        for j in range(N):
            if i == j:
                overlap_mat[i][j] = 6
                continue
            pj = perms6[j]
            ov = 0
            for k in range(5, -1, -1):
                if pi[-k:] == pj[:k]:
                    ov = k
                    break
            overlap_mat[i][j] = ov
            
    # Convert to list of lists for easier manipulation in path operations
    overlap_list = [list(row) for row in overlap_mat]
    
    def calc_total_overlap(path):
        total = 0
        for i in range(len(path) - 1):
            total += overlap_list[path[i]][path[i + 1]]
        return total
        
    def build_word_from_path(path):
        w = []
        for idx in path:
            perm = perms6[idx]
            block = list(perm) + [7] + list(perm)
            overlap = 0
            min_len = min(len(w), len(block))
            for o in range(min_len, -1, -1):
                if w[len(w) - o:] == block[:o]:
                    overlap = o
                    break
            w += block[overlap:]
        return w
        
    # Generate initial paths using different heuristics
    random.seed(42)
    best_path = None
    best_overlap = -1
    
    # 1. Naive path (standard order)
    naive_path = list(range(N))
    naive_overlap = calc_total_overlap(naive_path)
    if naive_overlap > best_overlap:
        best_overlap = naive_overlap
        best_path = naive_path[:]
        
    # 2. Multiple Nearest Neighbor restarts
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        unvisited = list(range(N))
        random.shuffle(unvisited)
        path = [unvisited.pop()]
        
        while unvisited:
            last = path[-1]
            best_idx = -1
            best_ov = -1
            for idx in unvisited:
                ov = overlap_list[last][idx]
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
    if time.time() < deadline - 20:
        path = best_path[:]
        current_overlap = best_overlap
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            
            for i in range(len(path) - 2):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, len(path) - 1):
                    if j == len(path) - 1:
                        old_ov = overlap_list[path[i]][path[i + 1]]
                        new_ov = overlap_list[path[i]][path[j]]
                        if new_ov > old_ov:
                            path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                    else:
                        old_ov1 = overlap_list[path[i]][path[i + 1]]
                        old_ov2 = overlap_list[path[j]][path[j + 1]]
                        new_ov1 = overlap_list[path[i]][path[j]]
                        new_ov2 = overlap_list[path[i + 1]][path[j + 1]]
                        if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                            path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                            current_overlap += (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
                            improved = True
                            break
                if improved:
                    break
                    
        if current_overlap > best_overlap:
            best_overlap = current_overlap
            best_path = path[:]
            
    # Build the word from the best path
    w = build_word_from_path(best_path)
    
    # 4. Greedy removal polish
    all_perms = set(itertools.permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Count occurrences
    perm_counts = {}
    for i in range(len(w) - n + 1):
        p = tuple(w[i:i+n])
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    if len(perm_counts) < total_perms:
        return "".join(map(str, w))
        
    def is_removable(w, i, perm_counts, n, all_perms):
        for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
            p = tuple(w[start:start+n])
            if p in all_perms:
                if perm_counts.get(p, 0) == 1:
                    return False
        return True
        
    def remove_char(w, i, perm_counts, n, all_perms):
        for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
            p = tuple(w[start:start+n])
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
        
        w.pop(i)
        
        for start in range(max(0, i-n+1), i):
            if start + n <= len(w):
                p = tuple(w[start:start+n])
                if p in all_perms:
                    perm_counts[p] = perm_counts.get(p, 0) + 1

    # Run multiple passes of greedy removal
    for pass_num in range(10):
        if time.time() > deadline - 5:
            break
        improved = False
        i = 0
        while i < len(w) - n + 1:
            if time.time() > deadline - 5:
                break
            if is_removable(w, i, perm_counts, n, all_perms):
                remove_char(w, i, perm_counts, n, all_perms)
                improved = True
            else:
                i += 1
        if not improved:
            break
            
    return "".join(map(str, w))
