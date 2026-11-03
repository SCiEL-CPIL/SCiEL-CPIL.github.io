import time
import random
import math
import heapq

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

def _randomized_greedy_cover(q, n, radius, neigh, rng, deadline):
    """Randomized greedy set cover with tie-breaking randomness."""
    total = q ** n
    covered = [0] * total
    code = []
    code_set = set()
    rem_count = total
    
    while rem_count > 0:
        if time.monotonic() > deadline:
            break
        # Compute gains with random tie-breaking
        best_gain = -1
        best_candidates = []
        for c in range(total):
            if c in code_set:
                continue
            gain = 0
            for u in neigh[c]:
                if covered[u] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidates = [c]
            elif gain == best_gain and gain > 0:
                best_candidates.append(c)
        
        if not best_candidates:
            break
        
        # Random selection among tied candidates
        best_c = rng.choice(best_candidates)
        code.append(best_c)
        code_set.add(best_c)
        for u in neigh[best_c]:
            covered[u] += 1
            if covered[u] == 1:
                rem_count -= 1
    
    # Fallback
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
    
    return code, code_set, covered, rem_count

def _aggressive_local_search(code, code_set, covered, neigh, neigh_set, rng, deadline):
    """Aggressive local search: repeatedly remove redundant words, then try swaps."""
    total = len(code)
    best_code = code[:]
    best_size = len(code)
    
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
    
    # Phase 1: Iterative removal until no more removable
    changed = True
    while changed and time.monotonic() < deadline:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            if time.monotonic() > deadline:
                break
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
    
    # Phase 2: Try swaps systematically
    for i in range(len(code)):
        if time.monotonic() > deadline:
            break
        old_c = code[i]
        # Find candidates that cover uniquely-covered words
        unique_words = [u for u in neigh[old_c] if covered[u] == 1]
        if not unique_words:
            continue
        for target in unique_words:
            if time.monotonic() > deadline:
                break
            for new_c in range(len(neigh_set)):
                if new_c in code_set or new_c == old_c:
                    continue
                if target in neigh_set[new_c]:
                    if try_swap(old_c, new_c):
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        break
    
    # Phase 3: Another removal pass after swaps
    changed = True
    while changed and time.monotonic() < deadline:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            if time.monotonic() > deadline:
                break
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
    
    return best_code, best_size

def _k_swap_perturbation(code, code_set, covered, neigh, neigh_set, rng, k):
    """Remove k random codewords and add k new ones."""
    total = len(neigh)
    n_remove = min(k, len(code))
    
    # Remove k random codewords
    removed = []
    for _ in range(n_remove):
        if not code:
            break
        idx = rng.randint(0, len(code) - 1)
        c = code.pop(idx)
        code_set.discard(c)
        for u in neigh[c]:
            covered[u] -= 1
        removed.append(c)
    
    # Add k new random codewords
    added = 0
    attempts = 0
    while added < n_remove and attempts < 100:
        attempts += 1
        c = rng.randint(0, total - 1)
        if c not in code_set:
            code.append(c)
            code_set.add(c)
            for u in neigh[c]:
                covered[u] += 1
            added += 1
    
    return code, code_set, covered

def construct(q, n, radius):
    deadline = time.monotonic() + 118.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_set, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Multi-start ILS
    best_code = None
    best_size = float('inf')
    
    # Determine number of restarts based on time budget
    # Each restart: greedy cover (~2s) + local search (~5s) = ~7s
    # Budget: 118s -> ~15 restarts max, use 10 to be safe
    num_restarts = 10
    restart_deadline = deadline - 10.0  # Leave 10s for final polish
    
    for restart in range(num_restarts):
        if time.monotonic() > restart_deadline:
            break
        
        # Randomized greedy initialization
        code, code_set, covered, rem_count = _randomized_greedy_cover(
            q, n, radius, neigh, rng, restart_deadline
        )
        
        if rem_count > 0:
            # Fallback
            while rem_count > 0:
                c = rng.randint(0, total - 1)
                if c not in code_set:
                    code.append(c)
                    code_set.add(c)
                    for u in neigh[c]:
                        covered[u] += 1
                        if covered[u] == 1:
                            rem_count -= 1
        
        # Local search
        ls_deadline = min(restart_deadline, time.monotonic() + 8.0)
        local_code, local_size = _aggressive_local_search(
            code, code_set, covered, neigh, neigh_set, rng, ls_deadline
        )
        
        if local_size < best_size:
            best_size = local_size
            best_code = local_code[:]
        
        # ILS perturbation: k-swap
        if restart < num_restarts - 1 and time.monotonic() < restart_deadline - 5.0:
            k = rng.randint(1, 3)
            code, code_set, covered = _k_swap_perturbation(
                local_code[:], set(local_code), 
                [0] * total,  # Rebuild covered
                neigh, neigh_set, rng, k
            )
            # Rebuild covered properly
            covered = [0] * total
            for c in code:
                for u in neigh[c]:
                    covered[u] += 1
    
    # 3. Final polish on best_code
    if best_code is not None:
        code_set = set(best_code)
        covered = [0] * total
        for c in best_code:
            for u in neigh[c]:
                covered[u] += 1
        
        final_deadline = deadline - 2.0
        final_code, final_size = _aggressive_local_search(
            best_code[:], code_set, covered, neigh, neigh_set, rng, final_deadline
        )
        
        if final_size < best_size:
            best_size = final_size
            best_code = final_code[:]
    
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
