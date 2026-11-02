import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
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
    deadline = time.monotonic() + 110.0
    total = q ** n
    rng = random.Random(42)
    
    neigh, _ = _precompute_neighbors(q, n, radius)
    neigh_sets = _get_neigh_sets(q, n, radius)
    
    # Precompute: for each codeword, which words does it uniquely cover?
    # We'll track counts dynamically
    
    def greedy_init(seed):
        """Greedy set cover with random tie-breaking for diversity."""
        r = random.Random(seed)
        counts = [0] * total
        code = []
        remaining = [True] * total
        rem_count = total
        
        while rem_count > 0:
            if time.monotonic() > deadline:
                break
            best_c = -1
            best_gain = -1
            best_second = -1  # for tie-breaking
            for c in range(total):
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                if gain > best_gain:
                    best_second = best_gain
                    best_gain = gain
                    best_c = c
                elif gain == best_gain and gain > 0:
                    # Random tie-break
                    if r.random() < 0.5:
                        best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1
        
        # Fallback if not fully covered
        if rem_count > 0:
            code_set = set(code)
            while rem_count > 0:
                for u in range(total):
                    if remaining[u]:
                        for nc in neigh[u]:
                            if nc not in code_set:
                                code.append(nc)
                                code_set.add(nc)
                                for v in neigh[nc]:
                                    counts[v] += 1
                                    if counts[v] == 1:
                                        remaining[v] = False
                                        rem_count -= 1
                                break
                        break
        
        return code, counts

    def polish(code, counts, time_budget):
        """Local search to minimize code size."""
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        end_time = time.monotonic() + time_budget
        
        def is_removable(c):
            for u in neigh[c]:
                if counts[u] == 1:
                    return False
            return True
        
        # First pass: remove all removable codewords
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        # SA phase with targeted swaps
        temp = 3.0
        cool_rate = 0.999
        min_temp = 0.05
        
        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.35:
                # Try Remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c):
                        code.pop(idx)
                        code_set.discard(c)
                        for u in neigh[c]:
                            counts[u] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        
            elif move_type < 0.85:
                # Try Swap - use targeted candidate (neighbor of existing codeword)
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    # Choose new_c from neighbors of old_c or random
                    if rng.random() < 0.7:
                        # Targeted: pick a neighbor of old_c
                        new_c = neigh[old_c][rng.randint(1, len(neigh[old_c])-1)]
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    # Check validity
                    valid = True
                    new_c_set = neigh_sets[new_c]
                    for u in neigh[old_c]:
                        if counts[u] == 1 and u not in new_c_set:
                            valid = False
                            break
                    
                    if valid:
                        # Accept
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                    else:
                        # SA acceptance for invalid swap (perturbation)
                        # Compute "cost": how many words would be uncovered
                        uncovered = 0
                        new_c_set = neigh_sets[new_c]
                        for u in neigh[old_c]:
                            if counts[u] == 1 and u not in new_c_set:
                                uncovered += 1
                        if uncovered <= 2 and rng.random() < math.exp(-uncovered / temp):
                            # Accept perturbation
                            for u in neigh[old_c]:
                                counts[u] -= 1
                            for u in neigh[new_c]:
                                counts[u] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            # Immediately try to fix by adding back uncovered words
                            for u in neigh[old_c]:
                                if counts[u] == 0:
                                    # Find a codeword that covers u and add it
                                    for nc in neigh[u]:
                                        if nc not in code_set:
                                            code.append(nc)
                                            code_set.add(nc)
                                            for v in neigh[nc]:
                                                counts[v] += 1
                                            break
                            
            else:
                # Try Add (perturbation)
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if rng.random() < math.exp(-1.0 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        for u in neigh[new_c]:
                            counts[u] += 1
            
            temp *= cool_rate
        
        # Final removal pass
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        return best_code, best_size

    # Multi-restart strategy
    overall_best_code = None
    overall_best_size = total + 1
    
    # Determine number of restarts based on time
    # Each restart: greedy (~0.1s) + polish (~remaining)
    num_restarts = 5
    time_per_restart = (deadline - time.monotonic()) / num_restarts
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 5:
            break
        
        # Greedy initialization with different seeds
        code, counts = greedy_init(42 + restart * 7)
        
        if not code:
            continue
        
        # Polish
        polished_code, polished_size = polish(code, counts, time_per_restart - 0.5)
        
        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]
    
    # If we haven't found anything, do one more greedy
    if overall_best_code is None:
        code, counts = greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result

# Precompute neighbor sets at module level for reuse across calls
_neigh_sets_cache = {}

def _get_neigh_sets(q, n, radius):
    key = (q, n, radius)
    if key not in _neigh_sets_cache:
        total = q ** n
        neigh, _ = _precompute_neighbors(q, n, radius)
        _neigh_sets_cache[key] = [set(neigh[i]) for i in range(total)]
    return _neigh_sets_cache[key]
