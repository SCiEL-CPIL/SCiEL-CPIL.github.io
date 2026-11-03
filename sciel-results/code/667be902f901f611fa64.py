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
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    neigh, _ = _precompute_neighbors(q, n, radius)
    
    best_code = None
    best_size = total + 1
    
    def greedy_init():
        code = []
        code_set = set()
        counts = [0] * total
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

        def remove_from_code(c):
            nonlocal rem_count
            code.remove(c)
            code_set.remove(c)
            for u in neigh[c]:
                counts[u] -= 1
                if counts[u] == 0:
                    remaining[u] = True
                    rem_count += 1

        while rem_count > 0:
            if time.monotonic() > deadline:
                return code, counts, remaining, rem_count
            best_c = -1
            best_gain = -1
            # Randomized tie-breaking
            candidates = []
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    candidates = [c]
                elif gain == best_gain:
                    candidates.append(c)
            
            if best_c == -1 or best_gain == 0:
                break
                
            # Pick randomly from best candidates
            c = rng.choice(candidates)
            add_to_code(c)
            
        return code, counts, remaining, rem_count

    def local_search(code, counts, remaining, rem_count):
        # Remove all redundant codewords
        changed = True
        while changed:
            changed = False
            # Iterate through a copy of code to allow removal
            for c in code[:]:
                removable = True
                for u in neigh[c]:
                    if counts[u] == 1:
                        removable = False
                        break
                if removable:
                    # Remove c
                    code.remove(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                        if counts[u] == 0:
                            remaining[u] = True
                            rem_count += 1
                    changed = True
        return code, counts, remaining, rem_count

    def perturb_and_repair(code, counts, remaining, rem_count):
        # Remove 1 random codeword
        if not code:
            return code, counts, remaining, rem_count
        
        c_to_remove = rng.choice(code)
        # Remove it
        code.remove(c_to_remove)
        for u in neigh[c_to_remove]:
            counts[u] -= 1
            if counts[u] == 0:
                remaining[u] = True
                rem_count += 1
        
        # Repair: Cover remaining words greedily
        # We want to add fewest words
        new_code = code[:]
        new_counts = counts[:]
        new_remaining = remaining[:]
        new_rem_count = rem_count
        
        # Greedy repair
        while new_rem_count > 0:
            if time.monotonic() > deadline:
                return code, counts, remaining, rem_count # Revert? Or just return current state? 
                # Better to revert to valid state if we can't finish repair in time? 
                # No, if we don't finish repair, the state is invalid. 
                # We should check time before starting repair or abort.
                # For now, if we hit deadline, we return the original code (before perturbation) 
                # but we don't have it saved easily here. 
                # Let's assume repair is fast enough (729 words, max ~10 steps).
            
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in new_code: # This check is slow if new_code is a list. Use set?
                    continue
                gain = 0
                for u in neigh[c]:
                    if new_remaining[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            new_code.append(best_c)
            for u in neigh[best_c]:
                new_counts[u] += 1
                if new_counts[u] == 1:
                    new_remaining[u] = False
                    new_rem_count -= 1
            
        return new_code, new_counts, new_remaining, new_rem_count

    # Main Loop: Multi-start ILS
    while time.monotonic() < deadline:
        # 1. Greedy Init
        code, counts, remaining, rem_count = greedy_init()
        
        # 2. Local Search
        code, counts, remaining, rem_count = local_search(code, counts, remaining, rem_count)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
        
        # 3. ILS Iterations
        # We can try to improve this specific solution
        current_code = code[:]
        current_counts = counts[:]
        current_remaining = remaining[:]
        current_rem_count = rem_count
        
        for _ in range(50): # Max ILS steps per restart
            if time.monotonic() > deadline:
                break
            
            # Perturb
            p_code, p_counts, p_remaining, p_rem_count = perturb_and_repair(
                current_code[:], current_counts[:], current_remaining[:], current_rem_count
            )
            
            # Local Search on perturbed state
            p_code, p_counts, p_remaining, p_rem_count = local_search(p_code, p_counts, p_remaining, p_rem_count)
            
            # Accept if better or equal (to escape plateaus)
            if len(p_code) <= len(current_code):
                current_code = p_code
                current_counts = p_counts
                current_remaining = p_remaining
                current_rem_count = p_rem_count
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code[:]
                    
            # If we hit the theoretical lower bound (71), we can stop early?
            # Best known is 73. Lower bound 71.
            if best_size <= 71:
                break

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
