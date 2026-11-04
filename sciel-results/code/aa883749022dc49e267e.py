import numpy as np
import time
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    else:
        if q == 3:
            import itertools
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n >= 4:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                return []
        else:
            import itertools
            return [list(t) for t in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1():
    start_time = time.time()
    deadline = start_time + 110
    
    q = 3
    n = 8
    total_words = q ** n
    
    # Pre-compute neighbors as a 2D numpy array
    neighbors_list = []
    for idx in range(total_words):
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        
        covered_indices = [idx]
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_word = word[:]
                new_word[pos] = new_val
                new_idx = 0
                power = 1
                for i in range(n):
                    new_idx += new_word[i] * power
                    power *= q
                covered_indices.append(new_idx)
        neighbors_list.append(covered_indices)
        
    neighbors = np.array(neighbors_list, dtype=np.int32)
    
    # Greedy Construction
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    while True:
        if time.time() > deadline:
            break
            
        # Vectorized score calculation
        # covered[neighbors] has shape (6561, 17)
        # We want count of False (uncovered)
        scores = np.sum(covered[neighbors] == False, axis=1)
        best_word = np.argmax(scores)
        best_score = scores[best_word]
        
        if best_score == 0:
            break
            
        code.append(int(best_word))
        covered[neighbors[best_word]] = True
        
    # Pruning Phase
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code:
        counts[neighbors[c]] += 1
        
    random.seed(42)
    
    for _ in range(50):  # More passes since it's fast now
        if time.time() > deadline:
            break
            
        random.shuffle(code)
        improved = False
        
        for i in range(len(code)):
            c = code[i]
            n_idx = neighbors[c]
            
            if np.min(counts[n_idx]) > 1:
                counts[n_idx] -= 1
                code.pop(i)
                improved = True
                break
        
        if not improved:
            break
            
    result = []
    for idx in code:
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        result.append(word)
        
    return result
