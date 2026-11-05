import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        total_words = q ** n
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        spheres = [[] for _ in range(total_words)]
        for i, word in enumerate(all_words):
            spheres[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val != word[pos]:
                        new_word = list(word)
                        new_word[pos] = val
                        j = word_to_idx[tuple(new_word)]
                        spheres[i].append(j)
        cc = np.zeros(total_words, dtype=np.int8)
        code = []
        while True:
            best_idx = -1
            best_gain = -1
            for i in range(total_words):
                gain = sum(1 for w in spheres[i] if cc[w] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0: break
            code.append(best_idx)
            for w in spheres[best_idx]: cc[w] += 1
        return [list(all_words[idx]) for idx in code]

    total_words = 3 ** 8
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    spheres = [[] for _ in range(total_words)]
    for i, word in enumerate(all_words):
        spheres[i].append(i)
        for pos in range(8):
            for val in range(3):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    j = word_to_idx[tuple(new_word)]
                    spheres[i].append(j)

    def greedy_init():
        cc = np.zeros(total_words, dtype=np.int16)
        code = []
        uncovered = total_words
        while uncovered > 0:
            best_idx = -1
            best_gain = -1
            for i in range(total_words):
                gain = 0
                for w in spheres[i]:
                    if cc[w] == 0: gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0: break
            code.append(best_idx)
            for w in spheres[best_idx]:
                if cc[w] == 0: uncovered -= 1
                cc[w] += 1
        return code, cc

    def reduce_code(code, cc):
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                c = code[i]
                redundant = True
                for w in spheres[c]:
                    if cc[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in spheres[c]: cc[w] -= 1
                    code.pop(i)
                    changed = True
                    break
        return code

    def local_search(code, cc):
        improved = True
        while improved:
            improved = False
            for i in range(len(code)):
                c_old = code[i]
                # Remove c_old
                for w in spheres[c_old]: cc[w] -= 1
                
                # Find best c_new
                best_new = -1
                best_gain = -1
                for c_new in range(total_words):
                    if c_new in code: continue
                    gain = 0
                    for w in spheres[c_new]:
                        if cc[w] == 0: gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_new = c_new
                
                if best_new != -1:
                    # Add best_new
                    for w in spheres[best_new]: cc[w] += 1
                    code[i] = best_new
                    
                    old_size = len(code)
                    code = reduce_code(code, cc)
                    
                    if len(code) < old_size:
                        improved = True
                        break
                    else:
                        # Revert if no improvement, to avoid wandering
                        # But maybe accept sideways moves? 
                        # For now, revert to ensure we only keep improvements.
                        for w in spheres[best_new]: cc[w] -= 1
                        code[i] = c_old
                        for w in spheres[c_old]: cc[w] += 1
                else:
                    for w in spheres[c_old]: cc[w] += 1
        return code, cc

    start_time = time.time()
    code, cc = greedy_init()
    code = reduce_code(code, cc)
    best_code = code.copy()
    best_cc = cc.copy()
    best_size = len(code)
    
    # Run Local Search
    code, cc = local_search(code, cc)
    if len(code) < best_size:
        best_code = code
        best_cc = cc
        best_size = len(code)

    # Iterated Local Search
    while (time.time() - start_time) < 100:
        # Perturb
        curr_code = best_code.copy()
        curr_cc = best_cc.copy()
        
        # Randomly swap 3 codewords
        for _ in range(3):
            if len(curr_code) < 2: break
            i, j = random.sample(range(len(curr_code)), 2)
            c_i = curr_code[i]
            c_j = curr_code[j]
            
            # Replace c_i with random c_new
            c_new = random.randint(0, total_words - 1)
            if c_new in curr_code: continue
            
            # Remove c_i
            for w in spheres[c_i]: curr_cc[w] -= 1
            # Add c_new
            for w in spheres[c_new]: curr_cc[w] += 1
            curr_code[i] = c_new
            
            # Reduce
            curr_code = reduce_code(curr_code, curr_cc)
            
            # If size increased, revert? Or just let it be?
            # Let's revert if size > best_size + 2 to keep it close
            if len(curr_code) > best_size + 5:
                # Revert perturbation
                for w in spheres[c_new]: curr_cc[w] -= 1
                for w in spheres[c_i]: curr_cc[w] += 1
                curr_code[i] = c_i
                break

        # Local Search on perturbed code
        curr_code, curr_cc = local_search(curr_code, curr_cc)
        
        if len(curr_code) < best_size:
            best_code = curr_code
            best_cc = curr_cc
            best_size = len(curr_code)

    return [list(all_words[idx]) for idx in best_code]
