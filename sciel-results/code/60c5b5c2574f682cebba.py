import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    return _fallback_construct(q, n, radius)

def _fallback_construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    if radius != 1:
        return [list(w) for w in all_words]
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    covered = [False] * total
    codewords = []
    
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        covers_i = set(nbrs)
        
        if all(covered[j] for j in covers_i):
            continue
        cnt = sum(1 for j in covers_i if not covered[j])
        if not codewords or cnt > 0:
            codewords.append(list(all_words[i]))
            for j in covers_i:
                covered[j] = True
    
    return codewords

def _construct_ternary_8_1():
    q = 3
    n = 8
    total = 6561
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    # Precompute all words and neighbors
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx
    
    # Precompute neighbors
    neighbors = [None] * total
    for idx in range(total):
        w = list(idx_to_word(idx))
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w[pos] = val
                    nbrs.append(word_to_idx(tuple(w)))
            w[pos] = orig
        neighbors[idx] = nbrs
    
    # Precompute neighbor sets
    neighbor_sets = [set(nbrs) for nbrs in neighbors]
    
    # Bit-parallel representation
    num_blocks = (total + 63) // 64
    neighbor_bits = [None] * total
    for i in range(total):
        blocks = [0] * num_blocks
        for nbr in neighbors[i]:
            blocks[nbr // 64] |= (1 << (nbr % 64))
        neighbor_bits[i] = blocks
    
    # Convert to numpy for fast operations
    neighbor_bits_np = np.array(neighbor_bits, dtype=np.uint64)
    
    def get_gain(idx, covered_mask_np):
        """Compute gain (uncovered count) for candidate idx"""
        bits = neighbor_bits_np[idx]
        uncovered = bits & (~covered_mask_np & np.uint64((1 << 64) - 1))
        return sum(bin(b).count('1') for b in uncovered)
    
    def is_fully_covered(covered_mask_np):
        """Check if all words are covered"""
        for b in range(num_blocks - 1):
            if covered_mask_np[b] != np.uint64((1 << 64) - 1):
                return False
        last_block = total % 64
        valid_mask = (1 << last_block) - 1 if last_block != 0 else (1 << 64) - 1
        return covered_mask_np[-1] == np.uint64(valid_mask)
    
    def greedy_construct(covered_mask_np, code_set, code_list, deadline):
        """Greedy phase to cover all words"""
        gains = np.array([len(neighbors[i]) for i in range(total)], dtype=np.int32)
        
        while time.time() < deadline - 30:
            if is_fully_covered(covered_mask_np):
                break
            
            # Find best candidate
            best_idx = -1
            best_gain = -1
            for i in range(total):
                if i in code_set:
                    continue
                if gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            code_set.add(best_idx)
            code_list.append(best_idx)
            
            # Update covered mask
            for nbr in neighbors[best_idx]:
                covered_mask_np[nbr // 64] |= np.uint64(1 << (nbr % 64))
            
            # Update gains
            for w in range(total):
                if w in code_set:
                    continue
                gains[w] = get_gain(w, covered_mask_np)
        
        return code_list
    
    def local_search(code_list, code_set, covered_mask_np, deadline, seed):
        """Local search to minimize code size"""
        random.seed(seed)
        
        # Build cover counts
        cover_count = [0] * total
        for c in code_list:
            for nbr in neighbors[c]:
                cover_count[nbr] += 1
        
        best_size = len(code_list)
        best_code = list(code_list)
        no_improvement = 0
        max_no_improve = 30
        
        while time.time() < deadline - 10:
            # Phase 1: Remove redundant codewords
            removed = False
            random.shuffle(code_list)
            for i in range(len(code_list) - 1, -1, -1):
                if time.time() > deadline - 15:
                    break
                c = code_list[i]
                removable = True
                for nbr in neighbors[c]:
                    if cover_count[nbr] <= 1:
                        removable = False
                        break
                if removable:
                    code_list.pop(i)
                    code_set.discard(c)
                    for nbr in neighbors[c]:
                        cover_count[nbr] -= 1
                    removed = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
            
            if removed:
                no_improvement = 0
                continue
            
            # Phase 2: Single swap
            if len(code_list) > 450:
                idx_c = random.randint(0, len(code_list) - 1)
                c = code_list[idx_c]
                
                # Temporarily remove
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                
                # Find uncovered
                uncovered = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
                
                if not uncovered:
                    code_list.pop(idx_c)
                    code_set.discard(c)
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    no_improvement = 0
                    continue
                
                # Find replacement
                candidates = neighbor_sets[uncovered[0]].copy()
                for u in uncovered[1:]:
                    candidates &= neighbor_sets[u]
                candidates.discard(c)
                candidates -= code_set
                
                if candidates:
                    idx_cp = random.choice(list(candidates))
                    code_list[idx_c] = idx_cp
                    code_set.discard(c)
                    code_set.add(idx_cp)
                    for nbr in neighbors[idx_cp]:
                        cover_count[nbr] += 1
                    no_improvement = 0
                else:
                    for nbr in neighbors[c]:
                        cover_count[nbr] += 1
                    no_improvement += 1
            
            # Phase 3: 2-opt escape
            if no_improvement > max_no_improve:
                if time.time() < deadline - 20 and len(code_list) >= 2:
                    i1, i2 = random.sample(range(len(code_list)), 2)
                    c1, c2 = code_list[i1], code_list[i2]
                    
                    for nbr in neighbors[c1]:
                        cover_count[nbr] -= 1
                    for nbr in neighbors[c2]:
                        cover_count[nbr] -= 1
                    
                    all_uncovered = set()
                    for nbr in neighbors[c1]:
                        if cover_count[nbr] == 0:
                            all_uncovered.add(nbr)
                    for nbr in neighbors[c2]:
                        if cover_count[nbr] == 0:
                            all_uncovered.add(nbr)
                    
                    if not all_uncovered:
                        code_list.remove(c2)
                        code_list.remove(c1)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        if len(code_list) < best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
                        no_improvement = 0
                        continue
                    
                    candidates = neighbor_sets[list(all_uncovered)[0]].copy()
                    for u in list(all_uncovered)[1:]:
                        candidates &= neighbor_sets[u]
                    candidates.discard(c1)
                    candidates.discard(c2)
                    candidates -= code_set
                    
                    if candidates:
                        idx_cp = random.choice(list(candidates))
                        code_list.remove(c2)
                        code_list.remove(c1)
                        code_list.append(idx_cp)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        code_set.add(idx_cp)
                        for nbr in neighbors[idx_cp]:
                            cover_count[nbr] += 1
                        no_improvement = 0
                    else:
                        for nbr in neighbors[c1]:
                            cover_count[nbr] += 1
                        for nbr in neighbors[c2]:
                            cover_count[nbr] += 1
                        no_improvement += 1
            
            if no_improvement > max_no_improve * 3:
                break
        
        return best_code, best_size
    
    # Main search with multiple restarts
    best_code_overall = None
    best_size_overall = total
    
    # Initial greedy construction
    covered_mask_np = np.zeros(num_blocks, dtype=np.uint64)
    code_set = set()
    code_list = []
    
    greedy_construct(covered_mask_np, code_set, code_list, deadline)
    
    if is_fully_covered(covered_mask_np):
        best_code_overall = list(code_list)
        best_size_overall = len(code_list)
    
    # Local search with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        
        code_list_copy = list(code_list)
        code_set_copy = set(code_set)
        covered_mask_copy = covered_mask_np.copy()
        
        best_code, best_size = local_search(
            code_list_copy, code_set_copy, covered_mask_copy, deadline, seed
        )
        
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code
    
    # Final polish
    if best_code_overall and time.time() < deadline - 5:
        code_set_final = set(best_code_overall)
        cover_count_final = [0] * total
        for c in best_code_overall:
            for nbr in neighbors[c]:
                cover_count_final[nbr] += 1
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(best_code_overall) - 1, -1, -1):
                c = best_code_overall[i]
                removable = True
                for nbr in neighbors[c]:
                    if cover_count_final[nbr] <= 1:
                        removable = False
                        break
                if removable:
                    best_code_overall.pop(i)
                    code_set_final.discard(c)
                    for nbr in neighbors[c]:
                        cover_count_final[nbr] -= 1
                    improved = True
    
    if best_code_overall:
        return [list(all_words[c]) for c in best_code_overall]
    return [list(all_words[c]) for c in code_list]
