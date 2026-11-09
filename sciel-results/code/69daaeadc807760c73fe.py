import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Return a trivial cover (all words) or a simple greedy
        # For robustness, if not the specific case, just return a valid cover
        # Greedy set cover for general case
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Calculate neighbors for each word
        neighbors = [[] for _ in range(len(all_words))]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors[i].append(word_to_idx[tuple(new_w)])
                    
        covers = [set(neighbors[i]) for i in range(len(all_words))]
        
        uncovered = set(range(len(all_words)))
        code = []
        while uncovered:
            best_c = -1
            best_gain = 0
            for c in range(len(all_words)):
                gain = len(covers[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1: break
            code.append(best_c)
            for w in covers[best_c]:
                uncovered.discard(w)
                
        return [list(all_words[c]) for c in code]

    # Specific case q=3, n=7, radius=1
    total_words = 3 ** 7
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(7):
            for val in range(3):
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                neighbors[i].append(word_to_idx[tuple(new_w)])
                
    covers = [set(neighbors[i]) for i in range(total_words)]
    covered_by = [set() for _ in range(total_words)]
    for c in range(total_words):
        for w in covers[c]:
            covered_by[w].add(c)
            
    # 1. Greedy Initialization
    uncovered = set(range(total_words))
    code = []
    counts = [0] * total_words
    
    while uncovered:
        best_c = -1
        best_gain = 0
        # Optimization: Skip codewords that are already in code? 
        # No, we can't add duplicates. But greedy doesn't add duplicates.
        for c in range(total_words):
            # Fast check: if c is already in code, gain is 0 (or rather, we don't add it again)
            # Actually, we just pick the best one not in code.
            # Since we update 'uncovered', a codeword already in code will have 0 gain 
            # because all its words are covered (count >= 1). 
            # So we don't need to explicitly check 'in_code' for gain calculation, 
            # but we should avoid re-adding. 
            # len(covers[c] & uncovered) will be 0 for existing codewords.
            gain = len(covers[c] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c == -1:
            break
            
        code.append(best_c)
        for w in covers[best_c]:
            counts[w] += 1
            if counts[w] == 1:
                uncovered.discard(w)
                
    # 2. Local Search
    in_code = [False] * total_words
    for c in code:
        in_code[c] = True
        
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # Deletion Pass
        for c in list(code):
            if not in_code[c]:
                continue
            unique = False
            for w in covers[c]:
                if counts[w] == 1:
                    unique = True
                    break
            if not unique:
                in_code[c] = False
                code.remove(c)
                for w in covers[c]:
                    counts[w] -= 1
                improved = True
                
        # Swap Pass
        for c in list(code):
            if not in_code[c]:
                continue
            U_c = [w for w in covers[c] if counts[w] == 1]
            if not U_c:
                continue
            
            candidates = covered_by[U_c[0]].copy()
            for w in U_c[1:]:
                candidates &= covered_by[w]
            
            candidates -= set(code)
            
            if not candidates:
                continue
            
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            for c_prime in cand_list[:3]:
                # Perform swap
                in_code[c] = False
                for w in covers[c]:
                    counts[w] -= 1
                    
                in_code[c_prime] = True
                for w in covers[c_prime]:
                    counts[w] += 1
                
                code.remove(c)
                code.append(c_prime)
                improved = True
                break

    return [list(all_words[c]) for c in code]
