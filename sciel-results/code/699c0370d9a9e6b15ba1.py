import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius, deadline)

    total_words = 3 ** 6
    all_ones = (1 << total_words) - 1
    
    # Precompute indices
    indices = []
    for i in range(total_words):
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % 3)
            x //= 3
        indices.append(digits)
        
    # Precompute balls
    balls = [0] * total_words
    for i in range(total_words):
        balls[i] |= (1 << i)
        for pos in range(n):
            for delta in [-1, 1]:
                neighbor = list(indices[i])
                neighbor[pos] = (neighbor[pos] + delta) % 3
                idx = 0
                for k in range(n):
                    idx = idx * 3 + neighbor[k]
                balls[i] |= (1 << idx)

    best_code = None
    best_size = total_words
    
    # Try a few randomized greedy runs to find a smaller code
    # The deletion phase is deterministic given the code.
    # Greedy is deterministic if we don't randomize ties.
    # Let's randomize tie-breaking.
    
    for run in range(5): # 5 runs is cheap
        if time.time() > deadline:
            break
            
        uncovered = all_ones
        code = []
        
        while uncovered:
            best_count = -1
            candidates = []
            
            for w in range(total_words):
                count = (balls[w] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    candidates = [w]
                elif count == best_count and count > 0:
                    candidates.append(w)
            
            if not candidates:
                break
                
            # Randomly pick from candidates to break ties
            best_word = random.choice(candidates)
            code.append(best_word)
            uncovered &= ~balls[best_word]
            
        # Deletion Phase
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                coverage = 0
                for j in range(len(code)):
                    if i != j:
                        coverage |= balls[code[j]]
                
                if coverage == all_ones:
                    code.pop(i)
                    changed = True
                    break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if best_size == 73: # Optimal found
            break
            
    if best_code is None:
        # Fallback to first run result if something went wrong
        # But best_code should be set in first run
        best_code = code if 'code' in locals() else []

    return [indices[c] for c in best_code]

def _generic_construct(q, n, radius, deadline):
    # Same as before, simplified for fallback
    total_words = q ** n
    if total_words > 10000:
        return []
        
    all_ones = (1 << total_words) - 1
    indices = []
    for i in range(total_words):
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % q)
            x //= q
        indices.append(digits)
        
    balls = [0] * total_words
    for i in range(total_words):
        balls[i] |= (1 << i)
        for pos in range(n):
            for val in range(q):
                if val != indices[i][pos]:
                    neighbor = list(indices[i])
                    neighbor[pos] = val
                    idx = 0
                    for k in range(n):
                        idx = idx * q + neighbor[k]
                    balls[i] |= (1 << idx)
                    
    uncovered = all_ones
    code = []
    while uncovered and time.time() < deadline:
        best_word = -1
        best_count = -1
        for w in range(total_words):
            count = (balls[w] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_word = w
        if best_word == -1:
            break
        code.append(best_word)
        uncovered &= ~balls[best_word]
        
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            coverage = 0
            for j in range(len(code)):
                if i != j:
                    coverage |= balls[code[j]]
            if coverage == all_ones:
                code.pop(i)
                changed = True
                break
                
    return [indices[c] for c in code]
