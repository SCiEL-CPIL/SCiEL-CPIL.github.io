"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    total_words = q ** n

    # Precompute index mapping
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute covered indices for each possible codeword
    # covered_set[idx] is a set of word indices covered by codeword idx
    covered_sets = [set() for _ in range(total_words)]
    for idx in range(total_words):
        cw = idx_to_word(idx)
        covered_sets[idx].add(idx)
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = cw.copy()
                    lst[pos] = new_val
                    covered_sets[idx].add(word_to_idx(lst))

    # Build initial code using direct sum of tetracode if possible, else greedy
    # For q=3, n=6, tetracode sum gives 81 words.
    # A better start might be a random greedy or a known structure.
    # Let's use the tetracode sum as it's deterministic and valid.
    if q == 3 and n == 6:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        codewords = [word_to_idx(h + list(tail)) for h in tetracode 
                    for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Generic greedy
        codewords = []
        remaining = set(range(total_words))
        while remaining and time.time() < deadline - 5:
            best_cw = -1
            best_count = -1
            for idx in remaining:
                count = len(covered_sets[idx] & remaining)
                if count > best_count:
                    best_count = count
                    best_cw = idx
                if time.time() > deadline - 5:
                    break
            if best_cw == -1:
                break
            codewords.append(best_cw)
            remaining -= covered_sets[best_cw]

    # Helper to build coverage counts
    def build_coverage_count(cw_list):
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in cw_list:
            cov = covered_sets[cw]
            for w in cov:
                counts[w] += 1
        return counts

    # Helper to check if codeword is redundant
    def is_redundant(cw, counts):
        cov = covered_sets[cw]
        for w in cov:
            if counts[w] <= 1:
                return False
        return True

    # Phase 1: Greedy Removal
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        counts = build_coverage_count(codewords)
        
        # Try removing in random order
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 10:
                break
            if is_redundant(codewords[i], counts):
                codewords.pop(i)
                improved = True
                break

    # Phase 2: Swap-based Local Search
    # Goal: Escape local minima by swapping a codeword to enable removal of another.
    # If we remove c_i, and some words become uncovered, we try to find c_new that covers them.
    # If adding c_new allows removal of c_j, we reduce size by 1.
    
    swap_improved = True
    while swap_improved and time.time() < deadline - 5:
        swap_improved = False
        counts = build_coverage_count(codewords)
        
        # Try each codeword as a candidate for removal/swap
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            
            c_i = codewords[i]
            
            # Temporarily remove c_i
            # We don't actually pop it yet, we simulate.
            # Decrement counts for covered words
            cov_i = covered_sets[c_i]
            for w in cov_i:
                counts[w] -= 1
            
            # Find uncovered words
            uncovered = [w for w in cov_i if counts[w] == 0]
            
            if not uncovered:
                # c_i was redundant, remove it
                codewords.pop(i)
                swap_improved = True
                # Rebuild counts for next iteration of outer loop
                break
            
            # If there are uncovered words, try to find a replacement
            # We want a c_new (not in codewords) that covers all words in 'uncovered'
            # and potentially helps remove another codeword.
            
            # Optimization: Only consider c_new that covers at least one uncovered word.
            # Actually, it MUST cover ALL uncovered words to be a valid replacement for c_i alone.
            # If c_new doesn't cover all, we'd have a gap.
            # So we look for c_new such that uncovered <= covered_sets[c_new].
            
            best_c_new = -1
            best_score = -1 # Score: number of other codewords that become redundant
            
            # Iterate through all possible words
            for c_new in range(total_words):
                if c_new in codewords:
                    continue
                if c_new == c_i:
                    continue
                
                cov_new = covered_sets[c_new]
                
                # Check if c_new covers all uncovered words
                # Fast check: if any w in uncovered is not in cov_new, skip
                valid = True
                for w in uncovered:
                    if w not in cov_new:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # If valid, simulate adding c_new
                # The words in cov_new that were previously covered by others will have increased count.
                # The words in uncovered (subset of cov_new) will have count 1 (from c_new).
                # Other words in cov_new \ uncovered will have count increased by 1.
                
                # We need to check if any OTHER codeword c_j (j != i) becomes redundant.
                # A codeword c_j is redundant if all words in cov_j have count >= 2 
                # in the new configuration (with c_i removed and c_new added).
                
                # Let's calculate the "potential redundancy" of other codewords.
                # For each c_j, check if min count in cov_j is >= 2.
                
                # To do this efficiently, we can pre-compute which words are "critical" for each c_j.
                # But counts change.
                
                # Let's just iterate through other codewords.
                # This is O(N^2) per swap attempt. N is ~73. 73^2 * 729 * 729 is too slow?
                # 73 * 729 * 73 checks... ~4 million operations per swap iteration.
                # With 100 iterations, 400 million. Might be tight in Python.
                
                # Optimization: Only check c_j that overlap with c_new or c_i?
                # If c_j is far from c_new and c_i, its counts didn't change.
                # Counts changed for:
                # 1. Words in cov_i (decremented)
                # 2. Words in cov_new (incremented)
                # So only c_j that cover words in cov_i or cov_new are affected.
                
                # Let's identify affected codewords.
                # affected_indices = [j for j, c_j in enumerate(codewords) if j != i and (covered_sets[c_j] & (cov_i | cov_new))]
                
                # Simpler: Just check all j != i.
                # But we can break early if we find one redundant.
                
                # We want to maximize the number of removable codewords?
                # Or just find *any* removable codeword to reduce size.
                # If we find one, we can perform the swap: remove c_i, add c_new, remove c_j.
                
                removable_j = -1
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    c_j = codewords[j]
                    cov_j = covered_sets[c_j]
                    
                    # Check if c_j is redundant in new state
                    # New count for w:
                    # old_count[w] - (1 if w in cov_i else 0) + (1 if w in cov_new else 0)
                    # We need new_count[w] >= 2 for all w in cov_j.
                    
                    redundant = True
                    for w in cov_j:
                        new_count = counts[w] # This is the count AFTER removing c_i
                        if w in cov_new:
                            new_count += 1
                        if new_count < 2:
                            redundant = False
                            break
                    
                    if redundant:
                        removable_j = j
                        break
                
                if removable_j != -1:
                    # We found a valid swap that reduces size by 1
                    # Execute swap:
                    # 1. Remove c_i
                    # 2. Add c_new
                    # 3. Remove c_j (if j > i, index shifts, handle carefully)
                    
                    # Since we are in a simulation, let's just mark this as the best move
                    # and break out of c_new loop to execute it.
                    best_c_new = c_new
                    best_removable_j = removable_j
                    break
            
            if best_c_new != -1:
                # Execute the swap
                # 1. Remove c_i
                codewords.pop(i)
                # 2. Add c_new
                codewords.append(best_c_new)
                # 3. Remove c_j
                # Note: indices might have shifted if we appended to end.
                # c_j was at index 'removable_j' in the original list.
                # If we removed i, indices > i shifted left by 1.
                # If we appended c_new, it's at the end.
                # We need to remove the codeword that was at 'removable_j' in the ORIGINAL list.
                
                # Let's find the codeword value first
                target_cj = codewords[removable_j] if removable_j < i else codewords[removable_j - 1]
                # Wait, if removable_j == i, we skipped it.
                # If removable_j < i, it is still at removable_j.
                # If removable_j > i, it is now at removable_j - 1.
                
                # Actually, simpler: just find the value and remove it.
                # But we must be careful if c_new == target_cj? No, c_new was not in codewords.
                # And target_cj is in codewords.
                
                # Remove target_cj
                try:
                    codewords.remove(target_cj)
                except ValueError:
                    # Should not happen
                    pass
                
                swap_improved = True
                # Rebuild counts for next iteration
                break
            
            # If no swap found for this c_i, restore counts
            for w in cov_i:
                counts[w] += 1

    # Final pass: try all removals one more time
    counts = build_coverage_count(codewords)
    for i in range(len(codewords) - 1, -1, -1):
        if time.time() > deadline - 2:
            break
        if is_redundant(codewords[i], counts):
            codewords.pop(i)
            counts = build_coverage_count(codewords)

    # Convert indices back to words
    return [idx_to_word(cw) for cw in codewords]
# EVOLVE-BLOCK-END
