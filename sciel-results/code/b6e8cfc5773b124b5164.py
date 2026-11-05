"""Model-and-optimize approach for ternary covering code K_3(6,1).

Strategy:
1. Generate initial codes using randomized greedy set cover.
2. Use NumPy-vectorized coverage counting for fast greedy removal.
3. Apply swap-based local search with simulated annealing to escape local minima.
4. Use time management to ensure we finish well within the 120s limit.
"""
import itertools
import time
import random
import sys
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin: finish by 110s, limit is 120s
    
    # For the specific problem: q=3, n=6, radius=1
    # Space size is q^n = 729.
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def index_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))
    
    total_words = q ** n
    
    # Precompute coverage for all words in the space
    # cover_indices[i] is a list of word indices covered by word i
    # Each word covers itself and all distance-1 neighbors
    cover_indices = [None] * total_words
    for i in range(total_words):
        w = index_to_word(i)
        covered = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w_list = list(w)
                w_list[pos] = val
                covered.append(word_to_index(tuple(w_list)))
        cover_indices[i] = covered
        
    # Function to perform local search on a given initial code
    def local_search(initial_code):
        # initial_code: list of word indices
        active = np.zeros(total_words, dtype=bool)
        for i in initial_code:
            active[i] = True
            
        cov_count = np.zeros(total_words, dtype=np.int32)
        for i in initial_code:
            cov_count[cover_indices[i]] += 1
            
        current_size = len(initial_code)
        best_size = current_size
        best_code = initial_code[:]
        
        # SA Parameters
        T = 1.0
        T_end = 0.05
        cooling = 0.999
        max_iterations = 5000
        iters = 0
        
        # Precompute sets for faster intersection checks?
        # cover_indices are lists. Converting to set once might help if we do many lookups.
        # But we only check "is w in cover_indices[j]".
        # Since size is 13, linear search or set is both fast. Set is O(1) lookup.
        cover_sets = [set(c) for c in cover_indices]
        
        while T > T_end and iters < max_iterations:
            if time.time() > deadline - 5.0:
                break
                
            # Randomly choose a move
            # 1. Remove: Pick random active i. If redundant, remove.
            # 2. Swap: Pick random active i, random inactive j. If swap is valid, do it.
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Try Remove
                active_indices = np.where(active)[0]
                if len(active_indices) == 0:
                    break
                i = random.choice(list(active_indices))
                covered_by_i = cover_indices[i]
                # Check if any word is covered only by i
                # i.e., cov_count[w] == 1 for any w in covered_by_i
                # If all cov_count[w] > 1, i is redundant.
                if np.all(cov_count[covered_by_i] > 1):
                    # Remove i
                    active[i] = False
                    cov_count[covered_by_i] -= 1
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_code = [int(x) for x in np.where(active)[0]]
                    T = max(T * cooling, T_end)
                    iters += 1
                else:
                    # Cannot remove, maybe try to find a swap that allows removal?
                    # Or just accept failure of this move
                    T = max(T * cooling, T_end)
                    iters += 1
            else:
                # Try Swap
                active_indices = np.where(active)[0]
                inactive_indices = np.where(~active)[0]
                if len(active_indices) == 0 or len(inactive_indices) == 0:
                    break
                i = random.choice(list(active_indices))
                j = random.choice(list(inactive_indices))
                
                # Check if swapping i with j is valid
                # 1. j must cover all words that i covers exclusively.
                # 2. (Optional) The swap should not break coverage.
                
                covered_by_i = cover_indices[i]
                # Find words covered exclusively by i
                exclusive_to_i = [w for w in covered_by_i if cov_count[w] == 1]
                
                if not exclusive_to_i:
                    # i is redundant, just remove it (handled by remove move, but let's do it here too)
                    active[i] = False
                    cov_count[covered_by_i] -= 1
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_code = [int(x) for x in np.where(active)[0]]
                    T = max(T * cooling, T_end)
                    iters += 1
                else:
                    # Check if j covers all exclusive_to_i
                    # exclusive_to_i is a list of word indices
                    # cover_sets[j] is a set of word indices
                    if all(w in cover_sets[j] for w in exclusive_to_i):
                        # Perform swap
                        active[i] = False
                        active[j] = True
                        cov_count[covered_by_i] -= 1
                        cov_count[cover_indices[j]] += 1
                        # Size remains same
                        # We don't update best_size unless we can remove more later
                        T = max(T * cooling, T_end)
                        iters += 1
                    else:
                        T = max(T * cooling, T_end)
                        iters += 1
                        
        return best_code, best_size

    def greedy_set_cover():
        # Randomized greedy set cover
        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            best_word = -1
            best_gain = 0
            # Randomize order of words to break ties
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for w in candidates:
                # Calculate gain: number of uncovered words covered by w
                # cover_indices[w] is list of words
                # Intersection with uncovered
                # Since cover_indices is small (13), we can just check
                gain = 0
                for u in cover_indices[w]:
                    if u in uncovered:
                        gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w
                            # Early exit if we covered all? No, we need max gain.
                            # But if gain == len(uncovered), we can break
                            if gain == len(uncovered):
                                break
                # Optimization: if best_gain is already high, we can skip some
                # But for n=6, 729 words, this is fast enough.
            
            if best_word == -1:
                break # Should not happen if uncovered is not empty
            
            code.append(best_word)
            # Update uncovered
            for u in cover_indices[best_word]:
                if u in uncovered:
                    uncovered.remove(u)
            
            if len(uncovered) == 0:
                break
        
        return code

    best_code_overall = None
    best_size_overall = float('inf')
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10.0:
        # Generate initial code
        init_code = greedy_set_cover()
        
        # Local search
        res_code, res_size = local_search(init_code)
        
        if res_size < best_size_overall:
            best_size_overall = res_size
            best_code_overall = res_code
            num_restarts += 1
            
        # If we reached the known lower bound or very close, we can stop or continue
        if best_size_overall <= 73:
            # We can try to polish further or just break if time is tight
            # Let's do a few more to see if we can go lower (unlikely)
            if best_size_overall <= 71:
                break

    # Convert indices back to words
    if best_code_overall is None:
        # Fallback to direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
        code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                code.append(tuple(h) + tuple(tail))
        return [list(c) for c in code]
        
    final_words = [list(index_to_word(i)) for i in best_code_overall]
    
    # Final verification
    # Ensure distinctness and coverage
    covered_set = set()
    for w in final_words:
        idx = word_to_index(tuple(w))
        for u in cover_indices[idx]:
            covered_set.add(u)
            
    if len(covered_set) < total_words:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
        code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                code.append(tuple(h) + tuple(tail))
        return [list(c) for c in code]

    return final_words
