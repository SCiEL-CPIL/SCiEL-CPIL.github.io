import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific known case with a strong algorithm
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    
    # Fallback for other cases: simple greedy covering
    return _fallback_construct(q, n, radius)

def _fallback_construct(q, n, radius):
    # Simple greedy set cover for general case
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    if radius != 1:
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
    covers = {}
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
        covers[i] = set(nbrs)
        
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(total):
            if all(covered[j] for j in covers[i]):
                continue
            cnt = sum(1 for j in covers[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in covers[best_idx]:
            covered[j] = True
            
    return codewords

def _construct_ternary_8_1():
    q = 3
    n = 8
    radius = 1
    total = q ** n  # 6561
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s margin for return/cleanup

    # Precompute neighbors for all 6561 words
    # Using base-3 encoding for O(1) lookup instead of dict of tuples
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)

    # Precompute neighbors as lists of ints
    neighbors = [None] * total
    for idx in range(total):
        w = idx_to_word(idx)
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = list(w)
                    w_new[pos] = val
                    nbrs.append(word_to_idx(tuple(w_new)))
        neighbors[idx] = nbrs
    
    # Convert neighbors to numpy arrays for faster vectorized operations in local search
    neighbors_np = np.array(neighbors, dtype=np.int32)
    
    # Precompute neighbor sets for O(1) membership and intersection (used in swap search)
    neighbor_sets = [set(nbrs) for nbrs in neighbors]

    # Bit-parallel representation for fast gain calculation in greedy phase
    # Each word's neighborhood is a set of 17 indices (0..6560)
    # We need 6561 bits = 103 uint64 words
    num_blocks = (total + 63) // 64  # 103
    neighbor_bits = [None] * total
    for i in range(total):
        blocks = [0] * num_blocks
        for nbr in neighbors[i]:
            blocks[nbr // 64] |= (1 << (nbr % 64))
        neighbor_bits[i] = blocks

    # Greedy construction with bit-parallel gain updates
    covered_mask = [0] * num_blocks
    
    def is_covered(idx):
        return (covered_mask[idx // 64] >> (idx % 64)) & 1

    def mark_covered(idx):
        covered_mask[idx // 64] |= (1 << (idx % 64))
    
    def count_uncovered_in_mask(blocks):
        cnt = 0
        for b in range(num_blocks):
            bits = blocks[b] & (~covered_mask[b] & ((1 << 64) - 1))
            cnt += bin(bits).count('1')
        return cnt

    code_list = []
    code_set = set()
    
    # Greedy phase: pick word with max uncovered coverage
    gains = [0] * total
    for i in range(total):
        gains[i] = len(neighbors[i])  # initially all 17
    
    while time.time() < deadline - 60:
        # Check if all covered
        all_covered = True
        for b in range(num_blocks):
            if covered_mask[b] != ((1 << 64) - 1):
                if b == num_blocks - 1:
                    valid_mask = (1 << (total % 64)) - 1 if total % 64 != 0 else ((1 << 64) - 1)
                    if covered_mask[b] != valid_mask:
                        all_covered = False
                        break
                else:
                    all_covered = False
                    break
        
        if all_covered:
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
        
        # Add best_idx
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        # Update covered_mask
        for nbr in neighbors[best_idx]:
            mark_covered(nbr)
        
        # Update gains for all non-code words
        for w in range(total):
            if w in code_set:
                continue
            cnt = 0
            for nbr in neighbors[w]:
                if not is_covered(nbr):
                    cnt += 1
            gains[w] = cnt

    # Check if we've covered everything
    # If not, continue greedy until covered or time runs out
    while time.time() < deadline - 30:
        all_covered = True
        for b in range(num_blocks):
            if b < num_blocks - 1:
                if covered_mask[b] != ((1 << 64) - 1):
                    all_covered = False
                    break
            else:
                valid_mask = (1 << (total % 64)) - 1 if total % 64 != 0 else ((1 << 64) - 1)
                if covered_mask[b] != valid_mask:
                    all_covered = False
                    break
        
        if all_covered:
            break
        
        best_idx = -1
        best_gain = -1
        for i in range(total):
            if i in code_set:
                continue
            cnt = count_uncovered_in_mask(neighbor_bits[i])
            if cnt > best_gain:
                best_gain = cnt
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
        
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        for nbr in neighbors[best_idx]:
            mark_covered(nbr)
        
        # Update gains
        for w in range(total):
            if w in code_set:
                continue
            gains[w] = count_uncovered_in_mask(neighbor_bits[w])

    # Local Search: Peeling Heuristic
    # We maintain cover_count for each word.
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors[c]] += 1
    
    best_size = len(code_list)
    best_code = list(code_list)
    
    no_improvement_count = 0
    max_no_improvement = 20
    
    # Precompute list of neighbors for each code word for faster access
    # We will iterate through code_list indices
    
    while time.time() < deadline - 5:
        # Phase 1: Peeling - Remove most redundant codewords
        # A codeword c is "removable" if for all v in N(c), cover_count[v] >= 2.
        # We prioritize removing codewords with the highest "redundancy score":
        # number of neighbors v where cover_count[v] >= 2.
        
        removed_any = False
        # Shuffle code_list to break ties and add randomness
        random.shuffle(code_list)
        
        # We need to check removability efficiently.
        # Instead of checking all codewords every time, we can just iterate.
        # Since code_list is small (~500), this is fast.
        
        # To make peeling more effective, we identify the "most redundant" one first.
        # But a simple pass that removes any removable one is often sufficient if repeated.
        # Let's try to find the best candidate to remove to maximize future removals.
        
        best_removable_idx = -1
        best_redundancy = -1
        
        for i, c in enumerate(code_list):
            # Calculate redundancy: count of neighbors with cover_count >= 2
            # Note: cover_count includes c itself, so if cover_count[v] >= 2, 
            # it means at least one other codeword covers v.
            nbrs = neighbors[c]
            # Vectorized check
            redundant_count = np.sum(cover_count[nbrs] >= 2)
            
            # Check if removable: all neighbors must have cover_count >= 2
            # This is equivalent to redundant_count == len(nbrs)
            if redundant_count == len(nbrs):
                if redundant_count > best_redundancy:
                    best_redundancy = redundant_count
                    best_removable_idx = i
        
        if best_removable_idx != -1:
            c = code_list[best_removable_idx]
            code_list.pop(best_removable_idx)
            code_set.discard(c)
            cover_count[neighbors[c]] -= 1
            removed_any = True
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
        
        if removed_any:
            no_improvement_count = 0
            continue
        
        # Phase 2: Swap to escape local minimum
        # If no codeword is removable, we try to swap one.
        # We pick a random codeword (or the one with lowest redundancy) to remove,
        # find uncovered words, and find a replacement.
        
        # Pick a candidate to remove. 
        # Strategy: Pick the one with the lowest "criticality" or just random.
        # Let's pick the one with the lowest redundancy score to minimize damage.
        min_redundancy = 100
        candidate_idx = -1
        for i, c in enumerate(code_list):
            nbrs = neighbors[c]
            redundant_count = np.sum(cover_count[nbrs] >= 2)
            if redundant_count < min_redundancy:
                min_redundancy = redundant_count
                candidate_idx = i
        
        if candidate_idx == -1:
            candidate_idx = random.randint(0, len(code_list) - 1)
            
        c = code_list[candidate_idx]
        
        # Temporarily remove
        cover_count[neighbors[c]] -= 1
        
        # Find uncovered words (those with cover_count == 0)
        uncovered_words = neighbors[c][cover_count[neighbors[c]] == 0]
        
        if len(uncovered_words) == 0:
            # Safe removal (shouldn't happen if Phase 1 failed, but just in case)
            code_list.pop(candidate_idx)
            code_set.discard(c)
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
            no_improvement_count = 0
            continue
        
        # Find replacement: must be in intersection of neighbor_sets of all uncovered_words
        # Start with neighbors of first uncovered word, intersect with others
        candidate_set = neighbor_sets[uncovered_words[0]].copy()
        for u in uncovered_words[1:]:
            candidate_set &= neighbor_sets[u]
        
        # Remove c and existing code from candidates
        candidate_set.discard(c)
        candidate_set -= code_set
        
        found_swap = False
        for idx_cp in candidate_set:
            # Check if idx_cp covers all uncovered words (it should by construction)
            # Perform swap
            code_list[candidate_idx] = idx_cp
            code_set.discard(c)
            code_set.add(idx_cp)
            cover_count[neighbors[idx_cp]] += 1
            found_swap = True
            no_improvement_count = 0
            break
        
        if not found_swap:
            # Revert
            cover_count[neighbors[c]] += 1
            no_improvement_count += 1
        
        # Phase 3: 2-opt escape (remove two, add one) when stuck
        if no_improvement_count > max_no_improvement:
            if time.time() < deadline - 15:
                # Pick two random codewords to remove
                if len(code_list) >= 2:
                    i1, i2 = random.sample(range(len(code_list)), 2)
                    c1 = code_list[i1]
                    c2 = code_list[i2]
                    
                    # Temporarily remove both
                    cover_count[neighbors[c1]] -= 1
                    cover_count[neighbors[c2]] -= 1
                    
                    # Find uncovered words
                    all_uncovered = set()
                    for nbr in neighbors[c1]:
                        if cover_count[nbr] == 0:
                            all_uncovered.add(nbr)
                    for nbr in neighbors[c2]:
                        if cover_count[nbr] == 0:
                            all_uncovered.add(nbr)
                    
                    if not all_uncovered:
                        # Safe removal of both
                        code_list.remove(c2)
                        code_list.remove(c1)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        if len(code_list) < best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
                        no_improvement_count = 0
                        continue
                    
                    # Find replacement that covers all uncovered words
                    ulist = list(all_uncovered)
                    candidate_set = neighbor_sets[ulist[0]].copy()
                    for u in ulist[1:]:
                        candidate_set &= neighbor_sets[u]
                    
                    candidate_set.discard(c1)
                    candidate_set.discard(c2)
                    candidate_set -= code_set
                    
                    found = False
                    for idx_cp in candidate_set:
                        # Swap: remove c1, c2, add idx_cp
                        code_list.remove(c2)
                        code_list.remove(c1)
                        code_list.append(idx_cp)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        code_set.add(idx_cp)
                        cover_count[neighbors[idx_cp]] += 1
                        found = True
                        no_improvement_count = 0
                        break
                    
                    if not found:
                        # Revert
                        cover_count[neighbors[c1]] += 1
                        cover_count[neighbors[c2]] += 1
                        no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement * 2:
            # Aggressive restart: remove 10% and rebuild
            if time.time() < deadline - 20:
                num_remove = max(1, len(code_list) // 10)
                to_remove = random.sample(code_list, min(num_remove, len(code_list)))
                for c in to_remove:
                    if c in code_set:
                        code_set.discard(c)
                        cover_count[neighbors[c]] -= 1
                        code_list.remove(c)
                
                # Rebuild greedily
                while time.time() < deadline - 10:
                    # Check if all covered
                    all_covered = True
                    for idx in range(total):
                        if cover_count[idx] == 0:
                            all_covered = False
                            break
                    if all_covered:
                        break
                    
                    best_idx = -1
                    best_count = -1
                    for idx in range(total):
                        if idx in code_set:
                            continue
                        cnt = np.sum(cover_count[neighbors[idx]] == 0)
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = idx
                    
                    if best_idx == -1 or best_count == 0:
                        break
                    
                    code_list.append(best_idx)
                    code_set.add(best_idx)
                    cover_count[neighbors[best_idx]] += 1
                
                no_improvement_count = 0
    
    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(idx_to_word(c)) for c in best_code]
    return [list(idx_to_word(c)) for c in code_list]
