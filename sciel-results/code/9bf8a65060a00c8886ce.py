import numpy as np
import random
import itertools
import time

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple greedy for other parameters
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    neighbors[i].append(word_to_idx[new_word])
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = sum(1 for nb in neighbors[c] if nb in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            for nb in neighbors[best_c]:
                if nb in uncovered:
                    uncovered.remove(nb)
        
        return [list(all_words[i]) for i in solution]

    # Specific optimization for q=3, n=7, radius=1 using Simulated Annealing
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words)  # 2187
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (distance <= 1)
    neighbors = [None] * N
    for i, w in enumerate(all_words):
        nbrs = [i]
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                new_word = list(w)
                new_word[pos] = val
                nbrs.append(word_to_idx[tuple(new_word)])
        neighbors[i] = nbrs
    
    start_time = time.time()
    timeout = 115.0
    
    random.seed(42)
    
    # Helper: Greedy set cover for uncovered set
    def greedy_cover(uncovered_set):
        if not uncovered_set:
            return []
        solution = []
        remaining = set(uncovered_set)
        while remaining:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = sum(1 for nb in neighbors[c] if nb in remaining)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            for nb in neighbors[best_c]:
                if nb in remaining:
                    remaining.remove(nb)
        return solution
    
    # Helper: Prune redundant codewords
    def prune(code_list):
        if not code_list:
            return []
        cover_count = np.zeros(N, dtype=int)
        for c in code_list:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        final_code = list(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(final_code)):
                c = final_code[i]
                redundant = True
                for nb in neighbors[c]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    final_code.pop(i)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    changed = True
                    break
        return final_code
    
    # Initial solution: Greedy
    init_code = greedy_cover(set(range(N)))
    init_code = prune(init_code)
    
    best_code = set(init_code)
    best_size = len(best_code)
    
    # Simulated Annealing state
    current_code = set(init_code)
    cover_count = np.zeros(N, dtype=int)
    for c in current_code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
    uncov_count = int(np.sum(cover_count == 0))
    
    T = 50.0
    T_min = 0.5
    cooling_rate = 0.9995
    
    while time.time() - start_time < timeout:
        move_type = random.random()
        
        if move_type < 0.25:
            # Remove a random codeword
            if not current_code:
                continue
            c = random.choice(list(current_code))
            current_code.discard(c)
            for nb in neighbors[c]:
                cover_count[nb] -= 1
            new_uncov = int(np.sum(cover_count == 0))
            delta = (new_uncov - uncov_count) * 1000 + (len(current_code) - len(current_code) + 1)
            # delta = (new_uncov - uncov_count) * 1000 - 1
            delta = (new_uncov - uncov_count) * 1000 - 1
            
            if delta <= 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                uncov_count = new_uncov
                if uncov_count == 0 and len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
            else:
                current_code.add(c)
                for nb in neighbors[c]:
                    cover_count[nb] += 1
        
        elif move_type < 0.5:
            # Add a random codeword
            c = random.randint(0, N - 1)
            if c in current_code:
                continue
            current_code.add(c)
            for nb in neighbors[c]:
                cover_count[nb] += 1
            new_uncov = int(np.sum(cover_count == 0))
            delta = (new_uncov - uncov_count) * 1000 + 1
            
            if delta <= 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                uncov_count = new_uncov
                if uncov_count == 0 and len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
            else:
                current_code.remove(c)
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
        
        elif move_type < 0.75:
            # Swap: remove one, add one
            if not current_code:
                continue
            c_rem = random.choice(list(current_code))
            c_add = random.randint(0, N - 1)
            if c_add in current_code:
                continue
            
            current_code.discard(c_rem)
            for nb in neighbors[c_rem]:
                cover_count[nb] -= 1
            current_code.add(c_add)
            for nb in neighbors[c_add]:
                cover_count[nb] += 1
            
            new_uncov = int(np.sum(cover_count == 0))
            delta = (new_uncov - uncov_count) * 1000  # size unchanged
            
            if delta <= 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                uncov_count = new_uncov
                if uncov_count == 0 and len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
            else:
                current_code.remove(c_add)
                for nb in neighbors[c_add]:
                    cover_count[nb] -= 1
                current_code.add(c_rem)
                for nb in neighbors[c_rem]:
                    cover_count[nb] += 1
        
        else:
            # Perturb: remove k random, then greedy repair
            k = random.randint(2, 5)
            if len(current_code) <= k:
                continue
            to_remove = random.sample(list(current_code), k)
            for c in to_remove:
                current_code.discard(c)
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
            
            uncov_set = set(np.where(cover_count == 0)[0].tolist())
            if uncov_set:
                repair = greedy_cover(uncov_set)
                for c in repair:
                    if c not in current_code:
                        current_code.add(c)
                        for nb in neighbors[c]:
                            cover_count[nb] += 1
            
            uncov_count = int(np.sum(cover_count == 0))
            if uncov_count == 0 and len(current_code) < best_size:
                best_size = len(current_code)
                best_code = set(current_code)
        
        T = max(T * cooling_rate, T_min)
    
    # Ensure validity
    covered = np.zeros(N, dtype=bool)
    for c in best_code:
        for nb in neighbors[c]:
            covered[nb] = True
    if not np.all(covered):
        uncov_set = set(np.where(~covered)[0].tolist())
        while uncov_set:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = sum(1 for nb in neighbors[c] if nb in uncov_set)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            best_code.add(best_c)
            for nb in neighbors[best_c]:
                if nb in uncov_set:
                    uncov_set.remove(nb)
    
    return [list(all_words[i]) for i in best_code]
