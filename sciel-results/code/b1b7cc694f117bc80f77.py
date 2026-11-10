import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            import itertools
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    all_words = [idx_to_word(i) for i in range(total_words)]
    
    spheres = [set() for _ in range(total_words)]
    for i in range(total_words):
        w = all_words[i]
        spheres[i].add(i)
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    new_word = w.copy()
                    new_word[pos] = new_val
                    j = word_to_idx(new_word)
                    spheres[i].add(j)
                    
    sphere_bits = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in spheres[i]:
            mask |= (1 << j)
        sphere_bits[i] = mask
        
    # Precompute coverers: coverers[j] = list of i such that j in spheres[i]
    coverers = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in spheres[i]:
            coverers[j].append(i)
            
    best_code = None
    best_size = float('inf')
    
    num_runs = 100 # Increase runs since local search is fast?
    
    def local_search(code_set, deadline):
        current_code = set(code_set)
        count = [0] * total_words
        for c in current_code:
            for j in spheres[c]:
                count[j] += 1
                
        changed = True
        while changed:
            changed = False
            # Prune
            for c in list(current_code):
                safe = True
                for j in spheres[c]:
                    if count[j] <= 1:
                        safe = False
                        break
                if safe:
                    current_code.remove(c)
                    for j in spheres[c]:
                        count[j] -= 1
                    changed = True
            
            if time.time() > deadline:
                break
                
            # Swap
            candidates = list(current_code)
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > deadline:
                    break
                if c not in current_code:
                    continue
                    
                missing = []
                for j in spheres[c]:
                    if count[j] == 1:
                        missing.append(j)
                        
                if not missing:
                    current_code.remove(c)
                    for j in spheres[c]:
                        count[j] -= 1
                    changed = True
                    continue
                    
                # Find best c_prime
                target = missing[0]
                cands = coverers[target]
                random.shuffle(cands)
                
                best_c_prime = -1
                best_score = float('inf')
                
                for c_prime in cands:
                    if c_prime in current_code:
                        continue
                    
                    # Check if covers all missing
                    covers_all = True
                    for j in missing:
                        if not (sphere_bits[c_prime] >> j) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Calculate score: number of words j in spheres[c_prime] 
                    # that will have count > 1 after swap
                    score = 0
                    for j in spheres[c_prime]:
                        new_count = count[j]
                        if j in spheres[c]:
                            new_count -= 1
                        new_count += 1
                        if new_count > 1:
                            score += 1
                            
                    if score < best_score:
                        best_score = score
                        best_c_prime = c_prime
                        
                if best_c_prime != -1:
                    # Perform swap
                    current_code.remove(c)
                    for j in spheres[c]:
                        count[j] -= 1
                    current_code.add(best_c_prime)
                    for j in spheres[best_c_prime]:
                        count[j] += 1
                    changed = True
                    
        return current_code

    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        # Greedy
        while uncovered:
            best_gain = -1
            best_c = -1
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for c in candidates:
                gain = bin(sphere_bits[c] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                if best_gain == 15:
                    break
                    
            if best_c == -1:
                break
                
            code_indices.append(best_c)
            uncovered &= ~sphere_bits[best_c]
            
        # Local Search
        code_set = local_search(set(code_indices), deadline)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = code_set
            if best_size <= 186:
                break

    if best_code is None:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = [all_words[i] for i in best_code]
    return result
