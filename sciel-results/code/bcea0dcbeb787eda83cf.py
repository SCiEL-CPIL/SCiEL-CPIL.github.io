import time
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Precompute powers of 3
    pow3 = np.array([3**i for i in range(n)], dtype=np.int64)
    total = 3**n  # 6561
    
    # Convert word integer to vector: w = sum c[i]*3^i
    # Precompute digit vectors for all words
    digits = np.zeros((total, n), dtype=np.int64)
    tmp = all_words = np.arange(total, dtype=np.int64)
    tmp = all_words.copy()
    for i in range(n):
        digits[:, i] = tmp % 3
        tmp //= 3
    
    # Build cover_list: for each codeword c, the 17 words it covers (itself + radius-1 neighbors)
    cover_list = np.zeros((total, 1 + n*(q-1)), dtype=np.int64)
    for c in range(total):
        d = digits[c]
        idx = 0
        cover_list[c, idx] = c
        idx += 1
        for i in range(n):
            for v in range(q):
                if v != d[i]:
                    cover_list[c, idx] = c + (v - d[i]) * pow3[i]
                    idx += 1
    
    # === Phase 1: Deterministic structured initialization using tetracode product ===
    # Tetracode is a perfect (4,3,1) code with 9 codewords
    # Tetracode codewords (length 4 over {0,1,2}, radius 1, 9 codewords)
    tetracode = [
        [0,0,0,0], [0,1,1,2], [0,2,2,1],
        [1,0,2,1], [1,1,0,2], [1,2,1,0],
        [2,0,1,2], [2,1,2,0], [2,2,0,1]
    ]
    
    # Product code: combine two tetracode blocks of length 4 to get length 8
    # The product of two perfect (4,3,1) codes gives a (8,3,1) code with 81 codewords
    # Each codeword is [t1[0..3], t2[0..3]] where t1,t2 are tetracode codewords
    # This covers all words within Hamming distance 1 of the product code
    
    # Actually, let's verify: the product of two (4,3,1) perfect codes
    # A word w = [w1, w2] where w1 is length 4, w2 is length 4
    # dist(w, [t1,t2]) = dist(w1,t1) + dist(w2,t2)
    # For this to be <= 1, we need dist(w1,t1)+dist(w2,t2) <= 1
    # This means either (dist(w1,t1)=0 and dist(w2,t2)<=1) or (dist(w1,t1)<=1 and dist(w2,t2)=0)
    # Since tetracode is perfect with radius 1, every w1 is within dist 1 of some t1,
    # and every w2 is within dist 1 of some t2.
    # But we need BOTH to be within dist 1 simultaneously for the product to work with radius 1.
    # Actually no - the product code has radius 2, not 1.
    # dist([t1,t2], [w1,w2]) = dist(t1,w1) + dist(t2,w2) which can be up to 2.
    
    # So the product code doesn't directly give a radius-1 covering code.
    # Let me use a different approach: just use greedy with a good initial set.
    
    # === Phase 1: Greedy set cover with criticality-based ordering ===
    uncovered = np.ones(total, dtype=bool)
    covered_count = np.zeros(total, dtype=np.int32)
    selected = []
    
    # For each word w, precompute which codewords cover it (reverse mapping)
    # codewords_covering[w] = list of codeword indices c such that w is in cover_list[c]
    # Each word w is covered by: w itself, and all words at distance 1 from w
    # That's 1 + n*(q-1) = 17 codewords
    
    # Build reverse mapping: for each word w, which codewords cover it
    # A codeword c covers w if dist(c,w) <= 1
    # So for word w, the codewords that cover it are: w, and w with one coordinate changed
    codewords_covering = np.zeros((total, 1 + n*(q-1)), dtype=np.int64)
    for w in range(total):
        d = digits[w]
        idx = 0
        codewords_covering[w, idx] = w
        idx += 1
        for i in range(n):
            for v in range(q):
                if v != d[i]:
                    codewords_covering[w, idx] = w + (v - d[i]) * pow3[i]
                    idx += 1
    
    # Greedy: repeatedly pick the codeword that covers the most uncovered words
    for step in range(total):
        if time.time() > deadline - 40:  # leave time for local search
            break
        
        # For each candidate codeword c, count how many uncovered words it covers
        scores = uncovered[cover_list].sum(axis=1)
        
        # Pick the codeword with maximum score
        best_c = np.argmax(scores)
        best_score = scores[best_c]
        
        if best_score == 0:
            break  # all covered
        
        # Select best_c
        selected.append(best_c)
        
        # Update covered and uncovered
        for w in cover_list[best_c]:
            w = int(w)
            covered_count[w] += 1
            if covered_count[w] == 1:
                uncovered[w] = False
    
    # Check if all covered; if not, add more codewords
    if np.any(uncovered):
        # Add codewords for remaining uncovered words
        for w in range(total):
            if uncovered[w]:
                if time.time() > deadline - 40:
                    break
                # Pick the codeword that covers w and covers the most other uncovered words
                candidates = codewords_covering[w]
                best_c = -1
                best_score = -1
                for c in candidates:
                    c = int(c)
                    if c in selected:
                        continue
                    score = uncovered[cover_list[c]].sum()
                    if score > best_score:
                        best_score = score
                        best_c = c
                if best_c >= 0:
                    selected.append(best_c)
                    for w2 in cover_list[best_c]:
                        w2 = int(w2)
                        covered_count[w2] += 1
                        if covered_count[w2] == 1:
                            uncovered[w2] = False
    
    # === Phase 2: Local search - remove redundant codewords ===
    # Rebuild covered count from scratch
    covered_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        for w in cover_list[c]:
            covered_count[int(w)] += 1
    
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            c = selected[i]
            # Check if removing c leaves all words it covers still covered
            cl = cover_list[c]
            if np.all(covered_count[cl] > 1):
                # Remove c
                for w in cl:
                    covered_count[int(w)] -= 1
                selected.pop(i)
                changed = True
    
    # === Phase 3: 2-opt local search - try swapping pairs ===
    # Try removing two codewords and adding one or zero to maintain coverage
    # More aggressive: try replacing one codeword with another
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try 1-for-1 swaps: remove codeword i, add codeword j if coverage maintained
        for i in range(len(selected)):
            if time.time() > deadline - 5:
                break
            
            c_remove = selected[i]
            cl_remove = cover_list[c_remove]
            
            # Words that would become uncovered if we remove c_remove
            # These are words w in cl_remove where covered_count[w] == 1
            critical_words = []
            for w in cl_remove:
                w = int(w)
                if covered_count[w] == 1:
                    critical_words.append(w)
            
            if not critical_words:
                # Can just remove
                for w in cl_remove:
                    covered_count[int(w)] -= 1
                selected.pop(i)
                improved = True
                break
            
            # Try to find a codeword j that covers all critical_words
            # and doesn't introduce new uncovered words
            # For efficiency, check candidates that cover the first critical word
            first_critical = critical_words[0]
            candidates = codewords_covering[first_critical]
            
            best_j = -1
            best_new_uncovered = 0
            
            for j in candidates:
                j = int(j)
                if j == c_remove:
                    continue
                if j in selected:
                    continue
                
                cl_j = cover_list[j]
                
                # Check if j covers all critical words
                covers_all = True
                for cw in critical_words:
                    if cw not in cl_j:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Check how many new uncovered words j would create
                # Words in cl_j that are currently uncovered (covered_count == 0 after removing c_remove)
                # But we haven't removed c_remove yet, so we need to simulate
                # After removing c_remove, words in cl_remove have count decreased by 1
                # Words in cl_j \ cl_remove have same count
                # Words in cl_j ∩ cl_remove have count decreased by 1
                
                # A word w in cl_j is uncovered after swap if:
                # - w is in cl_remove and covered_count[w] == 1 (becomes 0)
                #   but j covers it, so it becomes 1 again
                # - w is NOT in cl_remove and covered_count[w] == 0
                #   but covered_count[w] == 0 means it's already uncovered, which shouldn't happen
                #   if our current solution is valid
                
                # Actually, if the current solution is valid, all words have covered_count >= 1
                # After removing c_remove, words in cl_remove with count==1 become 0
                # After adding j, words in cl_j with count==0 become 1
                # Net effect: words in (cl_remove \ cl_j) with count==1 become uncovered (bad)
                # Words in (cl_j \ cl_remove) with count==1 become covered (good, but they were already covered)
                # Words in (cl_remove ∩ cl_j) with count==1: removed then added, net 0 change
                
                # So the swap is valid if: for all w in (cl_remove \ cl_j), covered_count[w] > 1
                # i.e., no word becomes uncovered
                
                # Check: for w in cl_remove, if w not in cl_j, need covered_count[w] > 1
                # We already know critical_words = {w in cl_remove : covered_count[w] == 1}
                # So we need: for all w in critical_words, w in cl_j
                # Which we already checked above!
                
                # So the swap is always valid if j covers all critical words!
                # The size stays the same, so we only do this if it helps future reductions.
                # For now, skip 1-for-1 swaps that don't reduce size.
                pass
        
        # Try 2-for-1: remove two codewords, add one
        if time.time() < deadline - 5:
            for i in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, len(selected)):
                    if time.time() > deadline - 5:
                        break
                    
                    c1 = selected[i]
                    c2 = selected[j]
                    cl1 = cover_list[c1]
                    cl2 = cover_list[c2]
                    
                    # Words that would become uncovered if we remove both c1 and c2
                    # A word w is affected if it's in cl1 or cl2
                    # After removal, covered_count[w] decreases by:
                    #   1 if w in cl1 only
                    #   1 if w in cl2 only  
                    #   2 if w in both
                    # A word becomes uncovered if its new count is 0
                    
                    # Find words that would become uncovered
                    would_be_uncovered = set()
                    for w in cl1:
                        w = int(w)
                        new_count = covered_count[w]
                        if w in cl2:
                            new_count -= 2
                        else:
                            new_count -= 1
                        if new_count == 0:
                            would_be_uncovered.add(w)
                    
                    for w in cl2:
                        w = int(w)
                        if w not in cl1:
                            new_count = covered_count[w] - 1
                            if new_count == 0:
                                would_be_uncovered.add(w)
                    
                    if not would_be_uncovered:
                        # Can remove both
                        for w in cl1:
                            covered_count[int(w)] -= 1
                        for w in cl2:
                            covered_count[int(w)] -= 1
                        selected.pop(j)
                        selected.pop(i)
                        improved = True
                        break
                    elif len(would_be_uncovered) <= 17:
                        # Try to find a codeword that covers all would_be_uncovered words
                        # Check candidates covering the first would-be-uncovered word
                        first_w = list(would_be_uncovered)[0]
                        candidates = codewords_covering[first_w]
                        
                        for k in candidates:
                            k = int(k)
                            if k == c1 or k == c2:
                                continue
                            if k in selected:
                                continue
                            
                            cl_k = cover_list[k]
                            # Check if k covers all would_be_uncovered words
                            covers_all = all(w in cl_k for w in would_be_uncovered)
                            if covers_all:
                                # Check if adding k creates any new uncovered words
                                # Words in cl_k that are not in cl1 or cl2
                                # Their count doesn't change, so they're still covered
                                # Words in cl_k ∩ (cl1 ∪ cl2) - their count goes up, fine
                                # So the swap is valid!
                                
                                # Remove c1 and c2, add k
                                for w in cl1:
                                    covered_count[int(w)] -= 1
                                for w in cl2:
                                    covered_count[int(w)] -= 1
                                for w in cl_k:
                                    covered_count[int(w)] += 1
                                
                                selected.pop(j)
                                selected.pop(i)
                                selected.append(k)
                                improved = True
                                break
                        
                        if improved:
                            break
                if improved:
                    break
    
    # Final removal pass
    if time.time() < deadline - 2:
        covered_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            for w in cover_list[c]:
                covered_count[int(w)] += 1
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                c = selected[i]
                cl = cover_list[c]
                if np.all(covered_count[cl] > 1):
                    for w in cl:
                        covered_count[int(w)] -= 1
                    selected.pop(i)
                    changed = True
    
    # Convert selected indices to codeword vectors
    if not selected:
        # Fallback: return all words
        return [digits[i].tolist() for i in range(total)]
    
    selected_codes = [digits[c].tolist() for c in selected]
    
    return selected_codes
