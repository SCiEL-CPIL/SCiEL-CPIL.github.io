import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
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
    return neigh, neigh_set, total

def construct(q, n, radius):
    deadline = time.monotonic() + 118.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_set, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy set cover initialization
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
        # Fallback: add random words until covered
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
    
    # 3. Local search with simulated annealing
    temp = 1.0
    cool_rate = 0.9998
    min_temp = 0.01
    
    def is_removable(c):
        for u in neigh[c]:
            if covered[u] == 1:
                return False
        return True
    
    def try_remove(c):
        """Remove c if possible, return True if removed"""
        if not is_removable(c):
            return False
        idx = code.index(c)
        code.pop(idx)
        code_set.discard(c)
        for u in neigh[c]:
            covered[u] -= 1
        return True
    
    def try_swap(old_c, new_c):
        """Swap old_c with new_c if valid, return True if swapped"""
        if new_c in code_set or new_c == old_c:
            return False
        # Check validity: all words uniquely covered by old_c must also be covered by new_c
        for u in neigh[old_c]:
            if covered[u] == 1 and u not in neigh_set[new_c]:
                return False
        # Perform swap
        idx = code.index(old_c)
        code[idx] = new_c
        code_set.discard(old_c)
        code_set.add(new_c)
        for u in neigh[old_c]:
            covered[u] -= 1
        for u in neigh[new_c]:
            covered[u] += 1
        return True
    
    while time.monotonic() < deadline - 5.0:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try Remove
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                if try_remove(c):
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                        
        elif move_type < 0.85:
            # Try Swap: pick old_c randomly, pick new_c that covers some uncovered or uniquely-covered words
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                
                # Find candidate new_c values that could help
                # Prefer candidates that cover words uniquely covered by old_c
                candidates = []
                for u in neigh[old_c]:
                    if covered[u] == 1:
                        # u is uniquely covered by old_c, need new_c to also cover u
                        candidates.append(u)
                
                if candidates:
                    # Pick a random uniquely-covered word, find candidates covering it
                    target = rng.choice(candidates)
                    # Find codeword candidates that cover target
                    for new_c in range(total):
                        if new_c in code_set:
                            continue
                        if target in neigh_set[new_c]:
                            if try_swap(old_c, new_c):
                                if len(code) < best_size:
                                    best_size = len(code)
                                    best_code = code[:]
                                break
                else:
                    # No uniquely covered words, try random swap
                    new_c = rng.randint(0, total - 1)
                    if try_swap(old_c, new_c):
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]

        else:
            # Try Add (Perturbation) - add a word that covers many uncovered words
            new_c = rng.randint(0, total - 1)
            if new_c not in code_set:
                gain = sum(1 for u in neigh[new_c] if covered[u] == 0)
                if gain > 0 or rng.random() < math.exp(-2.0 / temp):
                    code.append(new_c)
                    code_set.add(new_c)
                    for u in neigh[new_c]:
                        covered[u] += 1
                    
        temp *= cool_rate
    
    # 4. Final polish: try all single-word removals and swaps systematically
    # First, try removing all removable words
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
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
    
    # Try systematic swaps: for each codeword, try replacing with each candidate
    for i in range(len(code)):
        if time.monotonic() > deadline - 2.0:
            break
        old_c = code[i]
        for new_c in range(total):
            if new_c in code_set or new_c == old_c:
                continue
            if try_swap(old_c, new_c):
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                break  # Only one swap per old_c in this pass
    
    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
