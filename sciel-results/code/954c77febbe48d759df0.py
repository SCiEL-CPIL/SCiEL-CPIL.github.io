import time
import random
import math
import sys

def construct(q, n, radius):
    start_time = time.monotonic()
    deadline = start_time + 115.0
    total = q ** n
    
    # Precompute neighbors for radius 1
    # For each index, neighbors are itself + all words at Hamming distance 1
    neigh = [[] for _ in range(total)]
    pow_q = [1]
    for i in range(1, n):
        pow_q.append(pow_q[-1] * q)
    
    for idx in range(total):
        neigh[idx].append(idx)
        for i in range(n):
            p = pow_q[i]
            digit = (idx // p) % q
            for new_digit in range(q):
                if new_digit != digit:
                    neigh[idx].append(idx + (new_digit - digit) * p)
    
    neigh_sets = [set(lst) for lst in neigh]
    
    best_code_global = None
    best_size_global = total + 1

    def greedy_init(rng):
        """Randomized greedy construction."""
        counts = [0] * total
        code = []
        code_set = set()
        remaining = [True] * total
        rem_count = total
        
        def add_to_code(c):
            nonlocal rem_count
            code.append(c)
            code_set.add(c)
            for u in neigh[c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1

        while rem_count > 0:
            best_c = -1
            best_gain = -1
            candidates = []
            for c in range(total):
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    candidates = [c]
                elif gain == best_gain:
                    candidates.append(c)
            if best_gain == 0:
                break
            best_c = rng.choice(candidates)
            add_to_code(best_c)
        
        return code, code_set, counts

    def local_search(code, code_set, counts, rng, time_limit):
        """Local search with remove, swap, and add moves."""
        ls_deadline = time.monotonic() + time_limit
        temp = 1.5
        cool_rate = 0.997
        min_temp = 0.05
        no_improve = 0
        max_no_improve = 800
        nonlocal best_code_global, best_size_global
        
        while time.monotonic() < ls_deadline and no_improve < max_no_improve:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.55:
                # Try Remove
                if len(code) > 0:
                    # Find removable candidates
                    removable = []
                    for idx, c in enumerate(code):
                        min_count = min(counts[u] for u in neigh[c])
                        if min_count > 1:
                            removable.append((min_count, idx))
                    
                    if removable:
                        removable.sort(reverse=True)
                        if rng.random() < 0.6 and len(removable) > 3:
                            idx = removable[rng.randint(0, 2)][1]
                        else:
                            idx = removable[rng.randint(0, len(removable) - 1)][1]
                        
                        c = code[idx]
                        can_remove = True
                        for u in neigh[c]:
                            if counts[u] == 1:
                                can_remove = False
                                break
                        if can_remove:
                            code.pop(idx)
                            code_set.remove(c)
                            for u in neigh[c]:
                                counts[u] -= 1
                            no_improve = 0
                            if len(code) < best_size_global:
                                best_size_global = len(code)
                                best_code_global = code[:]
                            continue
                    no_improve += 1
            elif move_type < 0.9:
                # Try Swap
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        no_improve += 1
                        continue
                        
                    valid = True
                    ns = neigh_sets[new_c]
                    for u in neigh[old_c]:
                        if counts[u] == 1 and u not in ns:
                            valid = False
                            break
                            
                    if valid:
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        code[idx] = new_c
                        code_set.remove(old_c)
                        code_set.add(new_c)
                        no_improve = 0
                        if len(code) < best_size_global:
                            best_size_global = len(code)
                            best_code_global = code[:]
                    else:
                        no_improve += 1
            else:
                # Try Add (perturbation)
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if rng.random() < math.exp(-1.0 / max(temp, 0.01)):
                        code.append(new_c)
                        code_set.add(new_c)
                        for u in neigh[new_c]:
                            counts[u] += 1
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            temp *= cool_rate
        
        return code, code_set, counts

    # Multiple restarts with diverse greedy starts
    restart_count = 0
    while time.monotonic() < deadline - 3.0:
        if time.monotonic() > deadline - 2.0:
            break
        restart_count += 1
        rng = random.Random(restart_count * 7919 + 12345)
        
        # Greedy init
        code, code_set, counts = greedy_init(rng)
        current_size = len(code)
        
        if current_size < best_size_global:
            best_size_global = current_size
            best_code_global = code[:]
        
        # Local search
        ls_time = min(1.8, max(0.5, deadline - time.monotonic() - 3.0))
        code, code_set, counts = local_search(code, code_set, counts, rng, ls_time)
        
        if best_size_global <= 73:
            break

    # Final polish if close to optimal
    if best_code_global is not None and best_size_global <= 76 and time.monotonic() < deadline - 2.0:
        counts = [0] * total
        code = best_code_global[:]
        code_set = set(code)
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        
        rng = random.Random(99999)
        polish_time = min(12.0, deadline - time.monotonic() - 1.0)
        code, code_set, counts = local_search(code, code_set, counts, rng, polish_time)

    # Fallback
    if best_code_global is None:
        counts = [0] * total
        code = []
        code_set = set()
        remaining = [True] * total
        rem_count = total
        
        def add_to_code(c):
            nonlocal rem_count
            code.append(c)
            code_set.add(c)
            for u in neigh[c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1

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
            add_to_code(best_c)
        best_code_global = code

    result = []
    for idx in best_code_global:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result
