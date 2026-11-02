import time
import random
import math
import itertools

def _precompute_neighbors(q, n, radius):
    total = q ** n
    # For radius 1, neighbors are the word itself and words differing in 1 position
    neigh = [[] for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
    return neigh, total

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, _ = _precompute_neighbors(q, n, radius)
    
    # Helper to convert index to word and vice versa
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        return word
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # 2. Initialize with Tetracode Direct Sum
    # Tetracode is a perfect code for q=3, n=4, r=1
    # Size: 9 * 3^(n-4)
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                w = h + list(tail)
                init_code.append(word_to_idx(w))
    else:
        # Fallback: Greedy initialization for other parameters
        init_code = []
        remaining = [True] * total
        rem_count = total
        while rem_count > 0:
            best_c = -1
            best_gain = -1
            for c in range(total):
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            init_code.append(best_c)
            for u in neigh[best_c]:
                if remaining[u]:
                    remaining[u] = False
                    rem_count -= 1
    
    best_code = list(init_code)
    best_size = len(best_code)

    # Helper: Compute coverage counts
    def get_counts(code):
        counts = [0] * total
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        return counts

    # Helper: Local Search - Remove redundant codewords
    def local_search(code):
        code = list(code)
        changed = True
        while changed:
            changed = False
            counts = get_counts(code)
            new_code = []
            for c in code:
                # Check if c is redundant: all neighbors must have count > 1
                redundant = True
                for u in neigh[c]:
                    if counts[u] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
        return code

    # Helper: Repair coverage by adding greedy codewords
    def repair(code):
        code = list(code)
        counts = get_counts(code)
        uncovered = [i for i in range(total) if counts[i] == 0]
        
        while uncovered:
            best_c = -1
            best_gain = -1
            # To speed up, we might only check a subset, but for n=6 (729 words), full scan is fast
            for c in range(total):
                if c in code:
                    continue
                gain = 0
                for u in neigh[c]:
                    if counts[u] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain <= 0:
                break # Should not happen if code is valid initially
            
            code.append(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
            
            # Update uncovered list efficiently? 
            # For simplicity and since n is small, we can just re-check or maintain a set.
            # Let's use a set for uncovered to make removal O(1)
            # Actually, the loop above iterates over all c, which is 729. 
            # The inner loop is over neighbors (1 + 2*n = 13). 
            # 729 * 13 ~ 9000 ops per step. 
            # Number of steps is at most 729.
            # Total ~ 6.5M ops per repair. This might be slow if done many times.
            
            # Optimization: Use a set of uncovered indices
            # Let's refactor repair to use a set for uncovered
            
        return code

    # Optimized Repair using set for uncovered
    def repair_fast(code):
        code = list(code)
        counts = [0] * total
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        
        uncovered = set()
        for i in range(total):
            if counts[i] == 0:
                uncovered.add(i)
        
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in code:
                    continue
                gain = 0
                for u in neigh[c]:
                    if u in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain <= 0:
                break
            
            code.append(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1 and u in uncovered:
                    uncovered.remove(u)
        
        return code

    # Initial Local Search
    best_code = local_search(best_code)
    best_size = len(best_code)

    # 3. Perturbation Loop
    # Randomly remove some codewords, repair, and local search
    while time.monotonic() < deadline:
        # Choose number of codewords to drop (1 to 3)
        k_drop = rng.randint(1, 3)
        if len(best_code) <= k_drop:
            k_drop = max(1, len(best_code) - 1)
        
        # Create a copy and remove k_drop random codewords
        current_code = list(best_code)
        rng.shuffle(current_code)
        dropped = current_code[:k_drop]
        current_code = current_code[k_drop:]
        
        # Repair
        repaired_code = repair_fast(current_code)
        
        # Local Search
        improved_code = local_search(repaired_code)
        
        if len(improved_code) < best_size:
            best_size = len(improved_code)
            best_code = improved_code
            
            # If we hit the theoretical lower bound or a very good number, we can exit early
            # Lower bound is 71. If we get 73, it's optimal known.
            if best_size <= 73:
                break

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = idx_to_word(idx)
        result.append(word)
        
    return result
