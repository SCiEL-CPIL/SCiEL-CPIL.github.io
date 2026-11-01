import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110

    # Handle non-standard parameters with a simple valid code
    if q != 3 or n != 7 or radius != 1:
        # Simple greedy set cover for general case
        space_size = q ** n
        if space_size > 10000:
             return [list(t) for t in itertools.product(range(q), repeat=n)]
        
        def word_to_idx(word):
            idx = 0
            for val in word:
                idx = idx * q + val
            return idx

        all_words = [tuple(t) for t in itertools.product(range(q), repeat=n)]
        uncovered = set(all_words)
        code = []
        while uncovered:
            best_word = None
            max_covered = -1
            for w in all_words:
                covered = 0
                for u in uncovered:
                    dist = sum(1 for a, b in zip(w, u) if a != b)
                    if dist <= radius:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_word = w
            code.append(list(best_word))
            new_uncovered = set()
            for u in uncovered:
                dist = sum(1 for a, b in zip(best_word, u) if a != b)
                if dist > radius:
                    new_uncovered.add(u)
            uncovered = new_uncovered
            if time.time() > deadline:
                break
        return code

    # Specific solution for q=3, n=7, radius=1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    initial_code = [tuple(c) for c in initial_code]
    
    space_size = 3 ** 7
    
    def word_to_idx(word):
        idx = 0
        for val in word:
            idx = idx * 3 + val
        return idx

    # Precompute neighbors
    neighbors = {}
    for i in range(space_size):
        word = []
        tmp = i
        for _ in range(n):
            word.append(tmp % 3)
            tmp //= 3
        word.reverse()
        
        nb = [i]
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_word = word[:]
                    new_word[pos] = new_val
                    nb.append(word_to_idx(tuple(new_word)))
        neighbors[i] = nb
        
    best_code_size = len(initial_code)
    best_code_set = set(word_to_idx(c) for c in initial_code)
    
    # Run multiple greedy removals with random tie-breaking
    num_runs = 100
    for run in range(num_runs):
        code_indices = [word_to_idx(c) for c in initial_code]
        current_code = set(code_indices)
        counts = [0] * space_size
        
        for idx in current_code:
            for nb in neighbors[idx]:
                counts[nb] += 1
                
        while time.time() < deadline:
            redundant = []
            for idx in current_code:
                is_redundant = True
                risk = 0
                for nb in neighbors[idx]:
                    if counts[nb] == 1:
                        is_redundant = False
                        break
                    elif counts[nb] == 2:
                        risk += 1
                if is_redundant:
                    redundant.append((risk, idx))
                    
            if not redundant:
                break
                
            min_risk = redundant[0][0]
            candidates = [idx for r, idx in redundant if r == min_risk]
            best_idx = random.choice(candidates)
            
            current_code.remove(best_idx)
            for nb in neighbors[best_idx]:
                counts[nb] -= 1
                
        if len(current_code) < best_code_size:
            best_code_size = len(current_code)
            best_code_set = set(current_code)
            
    # Convert best_code_set to list of lists
    final_code = []
    for idx in best_code_set:
        word = []
        tmp = idx
        for _ in range(n):
            word.append(tmp % 3)
            tmp //= 3
        word.reverse()
        final_code.append(word)
        
    return final_code
