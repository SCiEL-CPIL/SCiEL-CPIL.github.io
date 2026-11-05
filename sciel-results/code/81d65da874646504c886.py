import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if q != 3 or n != 6 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    total_words = 3 ** 6
    
    def idx_to_word(idx):
        return [idx % 3, (idx // 3) % 3, (idx // 9) % 3, (idx // 27) % 3, (idx // 81) % 3, (idx // 243) % 3]
        
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx

    balls = []
    ball_sets = []
    for i in range(total_words):
        w = idx_to_word(i)
        neighbors = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                w[pos] = val
                neighbors.append(word_to_idx(w))
            w[pos] = orig
        balls.append(neighbors)
        ball_sets.append(set(neighbors))

    # Greedy Set Cover
    uncovered = set(range(total_words))
    current_code = set()
    coverage = np.zeros(total_words, dtype=np.int32)
    
    while uncovered:
        if time.time() > deadline - 10: # Leave time for local search
            break
        best_w = -1
        best_gain = 0
        # Iterate through all words? 729 words.
        # For each, count intersection with uncovered.
        # This is O(729 * 16) per step.
        # Steps ~ 73. Total ~ 73 * 729 * 16 ~ 850k ops. Fast.
        for w in range(total_words):
            if w in current_code:
                continue
            gain = 0
            for nb in balls[w]:
                if nb in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_w = w
                if gain == len(uncovered): # Perfect
                    break
        
        if best_w == -1:
            break
            
        current_code.add(best_w)
        for nb in balls[best_w]:
            if nb in uncovered:
                uncovered.remove(nb)
            coverage[nb] += 1
            
    # Local Search
    best_size = len(current_code)
    best_code = set(current_code)
    
    while time.time() < deadline:
        removed_any = False
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c in code_list:
            if c not in current_code:
                continue
            if np.min(coverage[balls[c]]) > 1:
                current_code.remove(c)
                for nb in balls[c]:
                    coverage[nb] -= 1
                removed_any = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
        
        if removed_any:
            continue
        
        if not current_code:
            break
            
        c = random.choice(list(current_code))
        unique_c = [v for v in balls[c] if coverage[v] == 1]
        
        if not unique_c:
            continue
            
        swapped = False
        for _ in range(100):
            if time.time() > deadline:
                break
            w = random.randint(0, total_words - 1)
            if w in current_code:
                continue
            valid = True
            for v in unique_c:
                if v not in ball_sets[w]:
                    valid = False
                    break
            if valid:
                current_code.remove(c)
                for nb in balls[c]:
                    coverage[nb] -= 1
                
                current_code.add(w)
                for nb in balls[w]:
                    coverage[nb] += 1
                
                swapped = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
                break
        
    return [idx_to_word(idx) for idx in best_code]
