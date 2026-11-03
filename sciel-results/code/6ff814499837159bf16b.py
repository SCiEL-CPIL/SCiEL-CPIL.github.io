import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112  # 8s buffer for safety
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
        N = q ** n
        def word_to_idx(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx
        def idx_to_word(idx):
            word = [0] * n
            for i in range(n-1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word
        all_words = [idx_to_word(i) for i in range(N)]
        neighbor_masks = [0] * N
        for c_idx in range(N):
            c_word = all_words[c_idx]
            mask = 1 << c_idx
            if radius == 1:
                for i in range(n):
                    orig = c_word[i]
                    for new_val in range(q):
                        if new_val != orig:
                            new_word = c_word.copy()
                            new_word[i] = new_val
                            w_idx = word_to_idx(new_word)
                            mask |= (1 << w_idx)
            neighbor_masks[c_idx] = mask
        uncovered = (1 << N) - 1
        code = []
        while uncovered and time.time() < deadline:
            best_c = -1
            best_count = -1
            for c in range(N):
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            uncovered &= ~neighbor_masks[best_c]
        return [all_words[i] for i in code]
    
    # Main path: q=3, n=8, radius=1
    N = 3 ** 8  # 6561
    
    # Precompute neighbor lists and inverse mapping
    neighbor_lists = []
    word_to_coders = [[] for _ in range(N)]
    
    for c in range(N):
        word = []
        temp = c
        for i in range(8):
            word.append(temp % 3)
            temp //= 3
        word = word[::-1]  # word[0] is most significant
        
        neighbors = [c]
        for pos in range(8):
            orig = word[pos]
            for delta in [1, 2]:
                new_val = (orig + delta) % 3
                new_word = word[:]
                new_word[pos] = new_val
                idx = 0
                for j in range(8):
                    idx = idx * 3 + new_word[j]
                neighbors.append(idx)
        
        neighbor_lists.append(neighbors)
        for w in neighbors:
            word_to_coders[w].append(c)
    
    neighbor_arrays = [np.array(nl, dtype=np.int32) for nl in neighbor_lists]
    
    def time_left():
        return deadline - time.time()
    
    def build_cover_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
        return counts
    
    def idx_to_word(idx):
        word = [0] * 8
        for i in range(7, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word
    
    def word_to_idx(word):
        idx = 0
        for i in range(8):
            idx = idx * 3 + word[i]
        return idx
    
    def criticality_greedy():
        """
        Greedy that targets the hardest-to-cover uncovered word first.
        Uses inverse mapping to quickly find candidate codewords.
        """
        code = []
        code_set = set()
        covered = np.zeros(N, dtype=np.bool_)
        
        while True:
            if time_left() < 3:
                break
            
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            
            # Find the uncovered word with minimum criticality (fewest coders)
            min_crit = N + 1
            target_word = -1
            for w in uncovered:
                crit = len(word_to_coders[int(w)])
                if crit < min_crit:
                    min_crit = crit
                    target_word = int(w)
                    if min_crit == 1:  # Can't get better
                        break
            
            if target_word == -1:
                break
            
            # Find the best codeword that covers target_word and maximizes new coverage
            best_c = -1
            best_new_count = -1
            
            for c_idx in word_to_coders[target_word]:
                if c_idx in code_set:
                    continue
                new_count = int(np.sum(~covered[neighbor_arrays[c_idx]]))
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx
            
            if best_c == -1 or best_new_count == 0:
                # Fallback: pick any codeword that covers target_word
                for c_idx in word_to_coders[target_word]:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            covered[neighbor_arrays[best_c]] = True
        
        return code
    
    def greedy_complete(code):
        """Complete a partial covering with standard greedy."""
        code_set = set(code)
        counts = build_cover_counts(code)
        
        while True:
            if time_left() < 2:
                break
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            
            best_c = -1
            best_count = 0
            for c in range(N):
                if c in code_set:
                    continue
                covered = np.sum(counts[neighbor_arrays[c]] == 0)
                if covered > best_count:
                    best_count = covered
                    best_c = c
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            counts[neighbor_arrays[best_c]] += 1
        
        return code
    
    def local_search_remove(code):
        """
        Strict local search: repeatedly try to remove codewords.
        A codeword can be removed if no word is uniquely covered by it.
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        improved = True
        while improved and time_left() > 2:
            improved = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if np.any(counts[neighbor_arrays[c]] == 1):
                    continue
                # Safe to remove
                code.pop(i)
                code_set.discard(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                break
        
        return code
    
    def two_opt_local_search(code):
        """
        2-opt: try removing one codeword and adding one that covers all
        newly-uncovered words. Also try 2-removal + 1-addition.
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        improved = True
        while improved and time_left() > 3:
            improved = False
            
            # Try single removals first
            for i in range(len(code) - 1, -1, -1):
                if time_left() < 2:
                    break
                c = code[i]
                if np.any(counts[neighbor_arrays[c]] == 1):
                    continue
                code.pop(i)
                code_set.discard(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                break
            
            if improved:
                continue
            
            # Try single swaps: remove one, add one
            for i in range(len(code)):
                if time_left() < 2:
                    break
                
                c_remove = code[i]
                newly_uncovered = neighbor_arrays[c_remove][counts[neighbor_arrays[c_remove]] == 1]
                
                if len(newly_uncovered) == 0:
                    code.pop(i)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    improved = True
                    break
                
                # Find a replacement that covers all newly_uncovered
                best_add = -1
                best_gain = -1
                
                for c_add in range(N):
                    if c_add in code_set:
                        continue
                    if not np.all(np.isin(newly_uncovered, neighbor_arrays[c_add])):
                        continue
                    gain = np.sum(counts[neighbor_arrays[c_add]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = c_add
                
                if best_add != -1 and best_gain >= 0:
                    code.pop(i)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    code.append(best_add)
                    code_set.add(best_add)
                    counts[neighbor_arrays[best_add]] += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-removal + 1-addition
            for i in range(len(code) - 1):
                if time_left() < 2:
                    break
                for j in range(i + 1, len(code)):
                    if time_left() < 2:
                        break
                    
                    c1 = code[i]
                    c2 = code[j]
                    
                    # Words uniquely covered by c1 or c2
                    newly_uncovered = set()
                    for w in neighbor_arrays[c1]:
                        if counts[w] == 1:
                            newly_uncovered.add(int(w))
                    for w in neighbor_arrays[c2]:
                        if counts[w] == 1:
                            newly_uncovered.add(int(w))
                    
                    if len(newly_uncovered) == 0:
                        # Remove both
                        code.pop(j)
                        code.pop(i)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        counts[neighbor_arrays[c1]] -= 1
                        counts[neighbor_arrays[c2]] -= 1
                        improved = True
                        break
                    
                    if len(newly_uncovered) > 17:  # Can't be covered by one codeword
                        continue
                    
                    # Find a codeword that covers all newly_uncovered
                    best_add = -1
                    best_gain = -1
                    newly_uncovered_arr = np.array(list(newly_uncovered), dtype=np.int32)
                    
                    for c_add in range(N):
                        if c_add in code_set:
                            continue
                        if not np.all(np.isin(newly_uncovered_arr, neighbor_arrays[c_add])):
                            continue
                        gain = np.sum(counts[neighbor_arrays[c_add]] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_add = c_add
                    
                    if best_add != -1 and best_gain >= 1:
                        # Remove both, add one
                        code.pop(j)
                        code.pop(i)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        counts[neighbor_arrays[c1]] -= 1
                        counts[neighbor_arrays[c2]] -= 1
                        code.append(best_add)
                        code_set.add(best_add)
                        counts[neighbor_arrays[best_add]] += 1
                        improved = True
                        break
                
                if improved:
                    break
        
        return code
    
    def swap_search(code):
        """
        Targeted swap search: for each codeword, try to find a replacement
        that enables removal of another codeword.
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        improved = True
        while improved and time_left() > 5:
            improved = False
            
            # Try to find a swap that reduces total size
            for i in range(len(code)):
                if time_left() < 3:
                    break
                
                c_remove = code[i]
                newly_uncovered = neighbor_arrays[c_remove][counts[neighbor_arrays[c_remove]] == 1]
                
                if len(newly_uncovered) == 0:
                    code.pop(i)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    improved = True
                    break
                
                if len(newly_uncovered) > 17:
                    continue
                
                # Find a codeword that covers all newly_uncovered
                best_add = -1
                best_gain = -1
                
                for c_add in range(N):
                    if c_add in code_set:
                        continue
                    if not np.all(np.isin(newly_uncovered, neighbor_arrays[c_add])):
                        continue
                    gain = np.sum(counts[neighbor_arrays[c_add]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = c_add
                
                if best_add != -1 and best_gain >= 1:
                    code[i] = best_add
                    code_set.discard(c_remove)
                    code_set.add(best_add)
                    counts[neighbor_arrays[c_remove]] -= 1
                    counts[neighbor_arrays[best_add]] += 1
                    improved = True
                    break
        
        return code
    
    # Main algorithm
    # 1. Criticality-based greedy initialization
    code = criticality_greedy()
    code = greedy_complete(code)
    
    # 2. Strict local search: removals, then 2-opt
    code = local_search_remove(code)
    code = two_opt_local_search(code)
    code = local_search_remove(code)
    
    best_code = code
    best_size = len(code)
    
    # 3. Additional local search passes if time permits
    if time_left() > 30:
        code = two_opt_local_search(code)
        code = local_search_remove(code)
        code = swap_search(code)
        code = local_search_remove(code)
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
    
    # 4. Simulated annealing for further refinement
    def simulated_annealing(code, max_iters=300):
        """Simulated annealing for further refinement."""
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        uncovered_count = int(np.sum(counts == 0))
        penalty = max(3 * len(code), 20)
        
        current_obj = len(code) + penalty * uncovered_count
        best_code = list(code)
        best_obj = current_obj
        
        temp = 15.0
        cooling = 0.985
        
        for it in range(max_iters):
            if time_left() < 4:
                break
            if temp < 0.1:
                break
            
            move_type = random.random()
            
            if move_type < 0.3 and len(code) > 10:
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                newly_uncovered = int(np.sum(counts[neighbor_arrays[c_remove]] == 1))
                delta_obj = -1 + penalty * newly_uncovered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code.pop(idx)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    current_obj = new_obj
                    uncovered_count += newly_uncovered
                    
            elif move_type < 0.6:
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                newly_covered = int(np.sum(counts[neighbor_arrays[c_add]] == 0))
                delta_obj = 1 - penalty * newly_covered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code.append(c_add)
                    code_set.add(c_add)
                    counts[neighbor_arrays[c_add]] += 1
                    current_obj = new_obj
                    uncovered_count -= newly_covered
                    
            else:
                if len(code) == 0:
                    continue
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                rem_uncov = int(np.sum(counts[neighbor_arrays[c_remove]] == 1))
                add_cov = int(np.sum(counts[neighbor_arrays[c_add]] == 0))
                uncov_delta = rem_uncov - add_cov
                delta_obj = penalty * uncov_delta
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code[idx] = c_add
                    code_set.discard(c_remove)
                    code_set.add(c_add)
                    counts[neighbor_arrays[c_remove]] -= 1
                    counts[neighbor_arrays[c_add]] += 1
                    current_obj = new_obj
                    uncovered_count += uncov_delta
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_code = list(code)
            
            temp *= cooling
        
        # Ensure full coverage
        counts_best = build_cover_counts(best_code)
        if np.any(counts_best == 0):
            best_code = greedy_complete(best_code)
        
        return best_code
    
    if time_left() > 15:
        sa_code = simulated_annealing(best_code, max_iters=200)
        sa_code = local_search_remove(sa_code)
        sa_code = two_opt_local_search(sa_code)
        sa_code = local_search_remove(sa_code)
        if len(sa_code) < best_size:
            best_code = sa_code
            best_size = len(sa_code)
    
    # 5. Final pass
    if time_left() > 5:
        best_code = local_search_remove(best_code)
        best_code = two_opt_local_search(best_code)
        best_code = local_search_remove(best_code)
    
    # Verify full coverage
    final_counts = build_cover_counts(best_code)
    if np.any(final_counts == 0):
        best_code = greedy_complete(best_code)
        best_code = local_search_remove(best_code)
    
    # Final deduplication safety
    seen = set()
    deduped = []
    for c in best_code:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    best_code = deduped
    
    # Convert indices to words
    result = []
    for c in best_code:
        result.append(idx_to_word(c))
    
    return result
