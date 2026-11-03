"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    total_space = q ** n

    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    all_words = [idx_to_word(i) for i in range(total_space)]

    # Precompute cover indices: for each word, list of words it covers (distance <= radius)
    # radius is 1
    cover_indices = []
    for w in all_words:
        covered = [word_to_idx(w)]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = w[:]
                    nw[pos] = val
                    covered.append(word_to_idx(nw))
        cover_indices.append(covered)
        
    # Precompute cover masks for fast union if needed, but we'll use counts
    # Actually, for greedy, set intersection with uncovered is good.

    def greedy_random(seed):
        """Greedy set cover using random tie-breaking."""
        rng = random.Random(seed)
        uncovered = set(range(total_space))
        code = []
        
        while uncovered:
            best_gain = -1
            best_cands = []
            # Iterate through all possible words
            # Optimization: we can skip words already in code? 
            # But we don't know which are in code easily without a set.
            code_set = set(code)
            
            for w in range(total_space):
                if w in code_set:
                    continue
                # Calculate gain: how many currently uncovered words does w cover?
                # cover_indices[w] is a list.
                # Intersection with uncovered (set)
                gain = 0
                for u in cover_indices[w]:
                    if u in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early break not possible easily, but gain is small (max 15)
                            pass
                
                # This inner loop is slow. 2187 * 15 * 186 ~ 6M ops. OK.
                # But `if u in uncovered` is O(1) average.
                
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [w]
                elif gain == best_gain:
                    best_cands.append(w)
            
            if not best_cands:
                break
                
            w = rng.choice(best_cands)
            code.append(w)
            # Update uncovered
            for u in cover_indices[w]:
                if u in uncovered:
                    uncovered.discard(u)
                    
        return code

    def local_search_counts(code):
        """Local search using coverage counts to enable 2-for-1 swaps."""
        code = code[:]
        code_set = set(code)
        
        # Initialize counts
        counts = [0] * total_space
        for w in code:
            for u in cover_indices[w]:
                counts[u] += 1
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            n_cw = len(code)
            
            # Phase 1: Remove single redundant codewords
            for i in range(n_cw):
                if time.time() > deadline:
                    break
                w = code[i]
                # Check if removing w leaves all counts >= 1
                # We need to check words covered by w
                # If any count becomes 0, it's not removable
                removable = True
                for u in cover_indices[w]:
                    if counts[u] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove w
                    for u in cover_indices[w]:
                        counts[u] -= 1
                    code.pop(i)
                    code_set.discard(w)
                    improved = True
                    break # Restart loop
            
            if improved:
                continue
                
            # Phase 2: 2-for-1 swap
            # Try to remove i and j, and add k if necessary
            # To be efficient, we only try pairs if n_cw is small or we limit attempts
            # Given time, we might just do single removals and rely on greedy restarts
            # But 2-for-1 is powerful.
            
            # Limit pairs to avoid O(N^2) blowup if N is large
            # N is ~190. N^2 = 36100.
            # For each pair, check validity.
            # If valid, remove.
            # If invalid, find k.
            
            found_swap = False
            # To keep it fast, we can randomize the order of pairs or just iterate
            for i in range(n_cw):
                if found_swap or time.time() > deadline:
                    break
                w_i = code[i]
                for j in range(i + 1, n_cw):
                    if time.time() > deadline:
                        break
                    w_j = code[j]
                    
                    # Tentatively remove i and j
                    # We need to know if counts drop to 0
                    # And which words drop to 0
                    deficient = []
                    valid = True
                    for u in cover_indices[w_i]:
                        counts[u] -= 1
                        if counts[u] == 0:
                            deficient.append(u)
                    for u in cover_indices[w_j]:
                        counts[u] -= 1
                        if counts[u] == 0:
                            deficient.append(u)
                        elif counts[u] < 0:
                             # This shouldn't happen if counts were accurate
                             pass
                    
                    # Check if any count is 0
                    # We tracked deficient, but need to ensure no other count is 0?
                    # No, only words covered by i or j can become 0.
                    # So if deficient is empty, it's a valid removal.
                    
                    if not deficient:
                        # Valid removal
                        code.pop(j)
                        code.pop(i)
                        code_set.discard(w_i)
                        code_set.discard(w_j)
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Try to find a k that covers all deficient words
                        # deficient is a list of indices
                        # We need k such that set(deficient) is subset of cover_indices[k]
                        # And k is not in code (or if it is, we are just removing i,j and keeping k? 
                        # If k is in code, we are removing i,j and k stays. Net change -2.
                        # If k is not in code, we add k, remove i,j. Net change -1.
                        
                        # To be safe and simple, let's look for ANY k.
                        # If k is in code, we just remove i,j.
                        # If k is not in code, we replace i,j with k.
                        
                        # Optimization: check candidates in cover_indices of deficient words?
                        # Or just iterate all k?
                        # 2187 candidates.
                        # Check: all(d in cover_indices[k] for d in deficient)
                        
                        best_k = -1
                        for k in range(total_space):
                            # Skip if k is i or j (they are removed)
                            if k == w_i or k == w_j:
                                continue
                            
                            # Check if k covers all deficient
                            # Use set for speed?
                            # cover_indices[k] is a list.
                            # deficient is a list.
                            # Convert deficient to set for fast lookup?
                            # Or just loop.
                            
                            # Since deficient is small, loop is fine.
                            covers_all = True
                            for d in deficient:
                                if d not in cover_indices[k]:
                                    covers_all = False
                                    break
                            
                            if covers_all:
                                best_k = k
                                break
                                
                        if best_k != -1:
                            # Apply swap
                            # Restore counts for i and j? No, we already decremented.
                            # We need to increment counts for best_k
                            for u in cover_indices[best_k]:
                                counts[u] += 1
                            
                            # Remove i and j from code
                            code.pop(j)
                            code.pop(i)
                            code_set.discard(w_i)
                            code_set.discard(w_j)
                            
                            # Add best_k to code
                            code.append(best_k)
                            code_set.add(best_k)
                            
                            improved = True
                            found_swap = True
                            break
                        else:
                            # Restore counts
                            for u in cover_indices[w_i]:
                                counts[u] += 1
                            for u in cover_indices[w_j]:
                                counts[u] += 1

        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy restarts
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        code = greedy_random(42 + r)
        code = local_search_counts(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
    
    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
