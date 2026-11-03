import time
import random
import math
from itertools import product

def construct(q, n, radius):
    deadline = time.monotonic() + 118.0
    total = q ** n
    rng = random.Random(42)
    
    # Precompute neighbors for radius 1 (the specific case we care about)
    # For general radius, we'd need a different approach, but the task is specifically K_3(6,1)
    # We'll handle radius 1 efficiently and fall back for other cases
    
    if radius == 1:
        # Efficient neighbor precomputation for radius 1
        neigh = [[] for _ in range(total)]
        neigh_set = [set() for _ in range(total)]
        for idx in range(total):
            neigh[idx].append(idx)
            neigh_set[idx].add(idx)
            for i in range(n):
                pow_q_i = q ** i
                digit = (idx // pow_q_i) % q
                for new_digit in range(q):
                    if new_digit != digit:
                        new_idx = idx + (new_digit - digit) * pow_q_i
                        neigh[idx].append(new_idx)
                        neigh_set[idx].add(new_idx)
        
        # Precompute covers: for each word, which codewords cover it
        covers = [[] for _ in range(total)]
        for c in range(total):
            for u in neigh[c]:
                covers[u].append(c)
        
        # Greedy set cover initialization
        covered = [0] * total
        code = []
        code_set = set()
        rem_count = total
        
        while rem_count > 0:
            if time.monotonic() > deadline - 60.0:
                break
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for u in neigh[c]:
                    if covered[u] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            code_set.add(best_c)
            for u in neigh[best_c]:
                covered[u] += 1
                if covered[u] == 1:
                    rem_count -= 1
        
        if rem_count > 0:
            while rem_count > 0:
                c = rng.randint(0, total - 1)
                if c not in code_set:
                    code.append(c)
                    code_set.add(c)
                    for u in neigh[c]:
                        covered[u] += 1
                        if covered[u] == 1:
                            rem_count -= 1
        
        best_code = code[:]
        best_size = len(code)
        
        # Local search with simulated annealing
        temp = 1.0
        cool_rate = 0.9998
        min_temp = 0.01
        
        def is_removable(c):
            for u in neigh[c]:
                if covered[u] == 1:
                    return False
            return True
        
        def try_remove(c):
            if not is_removable(c):
                return False
            idx = code.index(c)
            code.pop(idx)
            code_set.discard(c)
            for u in neigh[c]:
                covered[u] -= 1
            return True
        
        def try_swap(old_c, new_c):
            if new_c in code_set or new_c == old_c:
                return False
            for u in neigh[old_c]:
                if covered[u] == 1 and u not in neigh_set[new_c]:
                    return False
            idx = code.index(old_c)
            code[idx] = new_c
            code_set.discard(old_c)
            code_set.add(new_c)
            for u in neigh[old_c]:
                covered[u] -= 1
            for u in neigh[new_c]:
                covered[u] += 1
            return True
        
        best_size = [len(code)]
        best_code = [code[:]]
        
        while time.monotonic() < deadline - 15.0:
            if temp < min_temp:
                temp = min_temp
                
            move_type = rng.random()
            
            if move_type < 0.4:
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if try_remove(c):
                        if len(code) < best_size[0]:
                            best_size[0] = len(code)
                            best_code[0] = code[:]
                            
            elif move_type < 0.75:
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    unique_words = [u for u in neigh[old_c] if covered[u] == 1]
                    
                    if unique_words:
                        best_new_c = -1
                        best_score = -1
                        for target in unique_words[:5]:
                            for new_c in covers[target]:
                                if new_c in code_set or new_c == old_c:
                                    continue
                                score = 0
                                for u in unique_words:
                                    if u in neigh_set[new_c]:
                                        score += 1
                                if score > best_score:
                                    best_score = score
                                    best_new_c = new_c
                        if best_new_c != -1 and try_swap(old_c, best_new_c):
                            if len(code) < best_size[0]:
                                best_size[0] = len(code)
                                best_code[0] = code[:]
                    else:
                        new_c = rng.randint(0, total - 1)
                        if try_swap(old_c, new_c):
                            if len(code) < best_size[0]:
                                best_size[0] = len(code)
                                best_code[0] = code[:]
            else:
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    gain = sum(1 for u in neigh[new_c] if covered[u] == 0)
                    if gain > 0 or rng.random() < math.exp(-2.0 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        for u in neigh[new_c]:
                            covered[u] += 1
                    
            temp *= cool_rate
        
        # Final polish
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if is_removable(c):
                    code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        covered[u] -= 1
                    changed = True
                    if len(code) < best_size[0]:
                        best_size[0] = len(code)
                        best_code[0] = code[:]
        
        for i in range(len(code)):
            if time.monotonic() > deadline - 2.0:
                break
            old_c = code[i]
            unique_words = [u for u in neigh[old_c] if covered[u] == 1]
            if unique_words:
                best_new_c = -1
                best_score = -1
                for target in unique_words[:10]:
                    for new_c in covers[target]:
                        if new_c in code_set or new_c == old_c:
                            continue
                        score = sum(1 for u in unique_words if u in neigh_set[new_c])
                        if score > best_score:
                            best_score = score
                            best_new_c = new_c
                if best_new_c != -1 and try_swap(old_c, best_new_c):
                    if len(code) < best_size[0]:
                        best_size[0] = len(code)
                        best_code[0] = code[:]
            else:
                for new_c in range(total):
                    if new_c in code_set or new_c == old_c:
                        continue
                    if try_swap(old_c, new_c):
                        if len(code) < best_size[0]:
                            best_size[0] = len(code)
                            best_code[0] = code[:]
                        break
        
        result = []
        for idx in best_code[0]:
            word = [0] * n
            temp_idx = idx
            for i in range(n):
                word[i] = temp_idx % q
                temp_idx //= q
            result.append(word)
        
        return result
    else:
        # Fallback for other radii: simple greedy
        neigh = [[] for _ in range(total)]
        for idx in range(total):
            # Compute all words within Hamming distance radius
            word = []
            temp_idx = idx
            for i in range(n):
                word.append(temp_idx % q)
                temp_idx //= q
            word = tuple(word)
            
            for offset in range(radius + 1):
                # Choose offset positions to change
                from itertools import combinations
                for positions in combinations(range(n), offset):
                    # For each position, try all other digits
                    for new_digits in product(range(q), repeat=offset):
                        new_word = list(word)
                        for pos, nd in zip(positions, new_digits):
                            new_word[pos] = nd
                        new_word = tuple(new_word)
                        new_idx = 0
                        for i in range(n - 1, -1, -1):
                            new_idx = new_idx * q + new_word[i]
                        if new_idx not in neigh[idx]:
                            neigh[idx].append(new_idx)
        
        covered = [0] * total
        code = []
        code_set = set()
        rem_count = total
        
        while rem_count > 0:
            if time.monotonic() > deadline - 10.0:
                break
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in code_set:
                    continue
                gain = sum(1 for u in neigh[c] if covered[u] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            code_set.add(best_c)
            for u in neigh[best_c]:
                covered[u] += 1
                if covered[u] == 1:
                    rem_count -= 1
        
        result = []
        for idx in code:
            word = [0] * n
            temp_idx = idx
            for i in range(n):
                word[i] = temp_idx % q
                temp_idx //= q
            result.append(word)
        
        return result
