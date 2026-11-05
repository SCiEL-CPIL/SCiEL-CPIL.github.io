import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    words = list(itertools.product(range(q), repeat=n))
    num_words = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    neighbors_list = [[] for _ in range(num_words)]
    neighbors_set = [set() for _ in range(num_words)]
    for i, w in enumerate(words):
        neighbors_list[i].append(i)
        neighbors_set[i].add(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors_list[i].append(idx)
                neighbors_set[i].add(idx)
                
    uncovered = set(range(num_words))
    code_indices = []
    
    # Greedy Set Cover
    while uncovered:
        if time.time() > deadline:
            break
        best_word_idx = -1
        best_gain = 0
        # Optimization: only check candidates in uncovered? 
        # No, a covered word can be a better center.
        # But we can prune: if a word is not in uncovered, its gain is at most len(neighbors).
        # Actually, just scan all.
        for i in range(num_words):
            # Count intersection with uncovered
            count = 0
            for nb in neighbors_list[i]:
                if nb in uncovered:
                    count += 1
            if count > best_gain:
                best_gain = count
                best_word_idx = i
        if best_gain == 0:
            break
        code_indices.append(best_word_idx)
        for nb in neighbors_list[best_word_idx]:
            uncovered.discard(nb)
            
    # Post-processing: Peeling
    current_code = list(code_indices)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_code = []
        for c_idx in current_code:
            essential = False
            for w in neighbors_list[c_idx]:
                # Check if w is covered by any other codeword in current_code
                covered_by_other = False
                for other_idx in current_code:
                    if other_idx == c_idx:
                        continue
                    if w in neighbors_set[other_idx]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    essential = True
                    break
            if essential:
                new_code.append(c_idx)
            else:
                improved = True
        current_code = new_code
        
    return [list(words[i]) for i in current_code]
