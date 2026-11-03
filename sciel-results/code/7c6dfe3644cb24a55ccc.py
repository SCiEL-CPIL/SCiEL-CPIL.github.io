import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for safety
    
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
    
    # Precompute neighbor masks and lists
    neighbor_masks = [0] * N
    neighbor_lists = [[] for _ in range(N)]
    
    for c_idx in range(N):
        c_word = all_words[c_idx]
        mask = 1 << c_idx
        neighbor_lists[c_idx].append(c_idx)
        
        if radius == 1:
            for i in range(n):
                orig = c_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = c_word.copy()
                        new_word[i] = new_val
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
        else:
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
                        
        neighbor_masks[c_idx] = mask
    
    # Precompute: for each word, which codewords cover it (inverse mapping)
    word_to_coders = [[] for _ in range(N)]
    for c_idx in range(N):
        for w in neighbor_lists[c_idx]:
            word_to_coders[w].append(c_idx)
    
    # Precompute criticality
    criticality = np.array([len(word_to_coders[w]) for w in range(N)], dtype=np.int32)
    
    def greedy_cover_criticality(seed=None):
        """Greedy that prioritizes uncovered words with fewest potential coders (criticality)"""
        if seed is not None:
            random.seed(seed)
            
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Get list of uncovered word indices
            uncovered_words = []
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w_idx = lsb.bit_length() - 1
                uncovered_words.append(w_idx)
                temp ^= lsb
            
            if not uncovered_words:
                break
            
            # Find word with minimum criticality among uncovered
            min_crit = N + 1
            target_word = -1
            
            for w_idx in uncovered_words:
                crit = criticality[w_idx]
                if crit < min_crit:
                    min_crit = crit
                    target_word = w_idx
                if min_crit == 1:
                    break
            
            if target_word == -1:
                break
            
            # Find the best codeword that covers target_word and maximizes new coverage
            best_c = -1
            best_new_count = -1
            
            for c_idx in word_to_coders[target_word]:
                if c_idx in code_set:
                    continue
                
                new_count = (uncovered & neighbor_masks[c_idx]).bit_count()
                
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx
            
            if best_c == -1 or best_new_count == 0:
                for c_idx in word_to_coders[target_word]:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
            
        return code
    
    def local_search_remove(code):
        """Remove redundant codewords"""
        if not code:
            return code
            
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                has_unique = False
                for w in neighbor_lists[c]:
                    if cover_counts[w] == 1:
                        has_unique = True
                        break
                        
                if not has_unique:
                    code.pop(i)
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break
                    
        return code
    
    def two_opt_local_search(code):
        """2-opt: try to replace one codeword with another to enable removal of another"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                
                c_remove = code[i]
                
                newly_uncovered = []
                for w in neighbor_lists[c_remove]:
                    if cover_counts[w] == 1:
                        newly_uncovered.append(w)
                
                if not newly_uncovered:
                    code.pop(i)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    improved = True
                    break
                
                best_add = -1
                best_gain = -1
                
                candidate_codewords = set()
                for w in newly_uncovered:
                    for c in word_to_coders[w]:
                        if c not in code_set:
                            candidate_codewords.add(c)
                
                for c_add in candidate_codewords:
                    covers_all = True
                    for w in newly_uncovered:
                        if (neighbor_masks[c_add] >> w) & 1 == 0:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    new_coverage = 0
                    for w in neighbor_lists[c_add]:
                        if cover_counts[w] == 0:
                            new_coverage += 1
                    
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_add = c_add
                
                if best_add != -1:
                    code.pop(i)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    
                    code.append(best_add)
                    for w in neighbor_lists[best_add]:
                        cover_counts[w] += 1
                    code_set.add(best_add)
                    
                    improved = True
                    break
        
        return code
    
    def simulated_annealing_local_search(code, max_iters=200):
        """Simulated annealing on the hypergraph: moves are add/remove/swap codewords.
        Objective: minimize |code| subject to full coverage.
        Penalty for uncovered words."""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        # Current objective: size + large penalty * uncovered_count
        uncovered_count = np.sum(cover_counts == 0)
        penalty = len(code) * 2  # penalty per uncovered word
        current_obj = len(code) + penalty * uncovered_count
        
        best_code = list(code)
        best_obj = current_obj
        
        temp = 50.0
        cooling = 0.95
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if temp < 0.5:
                break
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.3 and code:
                # Remove a random codeword
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                
                # Check impact
                newly_uncovered = 0
                for w in neighbor_lists[c_remove]:
                    if cover_counts[w] == 1:
                        newly_uncovered += 1
                
                delta_obj = -1 + penalty * newly_uncovered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or random.random() < np.exp(-delta_obj / temp):
                    code.pop(idx)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    current_obj = new_obj
                    uncovered_count += newly_uncovered
                    
            elif move_type < 0.6:
                # Add a random codeword
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                
                newly_covered = 0
                for w in neighbor_lists[c_add]:
                    if cover_counts[w] == 0:
                        newly_covered += 1
                
                delta_obj = 1 - penalty * newly_covered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or random.random() < np.exp(-delta_obj / temp):
                    code.append(c_add)
                    for w in neighbor_lists[c_add]:
                        cover_counts[w] += 1
                    code_set.add(c_add)
                    current_obj = new_obj
                    uncovered_count -= newly_covered
                    
            else:
                # Swap: remove one, add another
                if not code:
                    continue
                    
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                
                # Compute delta
                delta = 0
                for w in neighbor_lists[c_remove]:
                    if cover_counts[w] == 1:
                        delta += penalty  # becomes uncovered
                for w in neighbor_lists[c_add]:
                    if cover_counts[w] == 0:
                        delta -= penalty  # becomes covered
                delta += 1  # net size change: -1 + 1 = 0, but we track separately
                
                # Actually size stays same (remove 1, add 1)
                delta_obj = 0  # size unchanged
                # But coverage changes
                uncovered_delta = 0
                for w in neighbor_lists[c_remove]:
                    if cover_counts[w] == 1:
                        uncovered_delta += 1
                for w in neighbor_lists[c_add]:
                    if cover_counts[w] == 0:
                        uncovered_delta -= 1
                
                delta_obj = penalty * uncovered_delta
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or random.random() < np.exp(-delta_obj / temp):
                    code[idx] = c_add
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    for w in neighbor_lists[c_add]:
                        cover_counts[w] += 1
                    code_set.discard(c_remove)
                    code_set.add(c_add)
                    current_obj = new_obj
                    uncovered_count += uncovered_delta
            
            # Track best
            if current_obj < best_obj:
                best_obj = current_obj
                best_code = list(code)
            
            temp *= cooling
        
        # Ensure full coverage in best_code
        cover_counts_best = np.zeros(N, dtype=np.int32)
        for c in best_code:
            for w in neighbor_lists[c]:
                cover_counts_best[w] += 1
        
        if np.any(cover_counts_best == 0):
            # Fix uncovered words
            uncovered_mask = (1 << N) - 1
            for c in best_code:
                uncovered_mask &= ~neighbor_masks[c]
            
            while uncovered_mask and time.time() < deadline:
                best_c = -1
                best_count = -1
                for c in range(N):
                    if c in best_code:
                        continue
                    count = (uncovered_mask & neighbor_masks[c]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_c = c
                if best_c == -1:
                    break
                best_code.append(best_c)
                uncovered_mask &= ~neighbor_masks[best_c]
        
        return best_code
    
    # Run criticality-based greedy
    best_code = greedy_cover_criticality()
    best_code = local_search_remove(best_code)
    best_code = two_opt_local_search(best_code)
    
    # Apply simulated annealing
    sa_code = simulated_annealing_local_search(best_code, max_iters=300)
    sa_code = local_search_remove(sa_code)
    sa_code = two_opt_local_search(sa_code)
    
    if len(sa_code) < len(best_code):
        best_code = sa_code
    
    best_size = len(best_code)
    
    # Try multiple random restarts if time permits
    for restart in range(1, 15):
        if time.time() > deadline - 15:
            break
        
        code = greedy_cover_criticality(seed=restart * 42)
        code = local_search_remove(code)
        code = two_opt_local_search(code)
        
        # Apply SA
        sa_code = simulated_annealing_local_search(code, max_iters=100)
        sa_code = local_search_remove(sa_code)
        sa_code = two_opt_local_search(sa_code)
        
        if len(sa_code) < best_size:
            best_code = sa_code
            best_size = len(sa_code)
    
    # Final verification and fix if needed
    covered_mask = 0
    for c in best_code:
        covered_mask |= neighbor_masks[c]
        
    if covered_mask != (1 << N) - 1:
        uncovered = ((1 << N) - 1) & ~covered_mask
        while uncovered and time.time() < deadline:
            best_c = None
            best_count = -1
            for c in range(N):
                if c in best_code:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None:
                break
            best_code.append(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
    
    result = [all_words[i] for i in best_code]
    return result
