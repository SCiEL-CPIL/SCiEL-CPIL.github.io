import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: The evaluator kills after 150s. We aim for 110-120s max.
    deadline = start_time + 115.0
    
    total_words = q ** n
    
    # For the specific case q=3, n=6, radius=1, we can use a more efficient approach
    # But we need a general solution that works for any q, n, radius
    
    # 1. Precompute neighbors (Hamming distance <= radius) using integer indices for speed
    # This is the key speedup from Inspiration 1
    neigh = [[] for _ in range(total_words)]
    neigh_set = [set() for _ in range(total_words)]
    
    for idx in range(total_words):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        
        if radius == 1:
            # Efficient radius 1 neighbor generation
            temp = idx
            for i in range(n):
                pow_q_i = q ** i
                digit = (idx // pow_q_i) % q
                for new_digit in range(q):
                    if new_digit != digit:
                        new_idx = idx + (new_digit - digit) * pow_q_i
                        neigh[idx].append(new_idx)
                        neigh_set[idx].add(new_idx)
        else:
            # General case: check Hamming distance (slower, but necessary for generality)
            # Since this is a specific task (q=3, n=6, r=1), the above branch is the critical path.
            # For other cases, we fall back to a simpler but potentially slower method if needed.
            # However, given the constraints, we assume the primary case is optimized.
            # If radius > 1, we need a more general neighbor generation.
            # For now, we stick to the optimized radius=1 path as it's the target task.
            # If radius != 1, we can use a brute force check for neighbors if total_words is small.
            # But since the task is fixed to K_3(6,1), we optimize for that.
            pass

    # If radius != 1, we need a fallback. But the task is specific.
    # Let's add a generic fallback just in case, though it might be slow for large n.
    if radius != 1:
        # Rebuild neighbors for general radius
        neigh = [[] for _ in range(total_words)]
        neigh_set = [set() for _ in range(total_words)]
        for i in range(total_words):
            for j in range(total_words):
                # Calculate Hamming distance
                dist = 0
                ti = i
                tj = j
                for _ in range(n):
                    di = ti % q
                    dj = tj % q
                    if di != dj:
                        dist += 1
                    ti //= q
                    tj //= q
                    if dist > radius:
                        break
                if dist <= radius:
                    neigh[i].append(j)
                    neigh_set[i].add(j)

    # 2. Build Distance-2 Graph for Independent Set Initialization (from Inspiration 1)
    # Two words are connected if their Hamming distance is exactly 2.
    # A maximal independent set in this graph gives a set of words where no two are at distance 2.
    # This is a strong structural property for covering codes.
    
    # We only build this if radius == 1 and n is small enough, or specifically for the target case.
    # For q=3, n=6, total=729. Building this graph is fast.
    dist2_adj = [set() for _ in range(total_words)]
    
    # Optimization: Only build the graph if we have time and it's the target case.
    # For 729 nodes, this is fast.
    if radius == 1:
        for idx in range(total_words):
            # To avoid duplicates, we add to both adj lists.
            for i in range(n):
                pow_q_i = q ** i
                digit_i = (idx // pow_q_i) % q
                for j in range(i + 1, n):
                    pow_q_j = q ** j
                    digit_j = (idx // pow_q_j) % q
                    # Change digit i and j
                    for new_i in range(q):
                        if new_i == digit_i:
                            continue
                        for new_j in range(q):
                            if new_j == digit_j:
                                continue
                            new_idx = idx + (new_i - digit_i) * pow_q_i + (new_j - digit_j) * pow_q_j
                            if new_idx != idx:
                                dist2_adj[idx].add(new_idx)
                                dist2_adj[new_idx].add(idx)

    rng = random.Random(42)
    
    # 3. Greedy Independent Set Initialization (from Inspiration 1)
    # We want a maximal independent set in the distance-2 graph.
    # Strategy: Randomly shuffle vertices, then greedily pick a vertex if it doesn't conflict.
    
    candidates = list(range(total_words))
    rng.shuffle(candidates)
    
    code_set = set()
    in_set = [False] * total_words
    
    # Greedy Independent Set
    for v in candidates:
        if in_set[v]:
            continue
        # Check if v conflicts with any node in code_set
        # Conflict if u in code_set and u is in dist2_adj[v]
        is_conflict = False
        for u in dist2_adj[v]:
            if in_set[u]:
                is_conflict = True
                break
        if not is_conflict:
            code_set.add(v)
            in_set[v] = True
    
    # Convert to list for easier manipulation
    code = list(code_set)
    
    # 4. Repair to Cover
    # The independent set might not cover the whole space.
    # We need to add words until it's a valid cover.
    counts = [0] * total_words
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
            
    # Find uncovered words
    uncovered = [i for i in range(total_words) if counts[i] == 0]
    
    # Greedily add words to cover uncovered space
    while uncovered and time.time() < deadline - 10.0:
        best_c = -1
        best_gain = 0
        # Iterate all candidates to find the one covering most uncovered nodes
        for c in range(total_words):
            if c in code_set:
                continue
            gain = 0
            for u in neigh[c]:
                if counts[u] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c == -1:
            break
            
        code_set.add(best_c)
        code.append(best_c)
        for u in neigh[best_c]:
            counts[u] += 1
        
        # Recalculate uncovered
        uncovered = [i for i in range(total_words) if counts[i] == 0]
        
    # If still uncovered (shouldn't happen with greedy), force add
    if uncovered:
        for u in uncovered:
            if u not in code_set:
                code_set.add(u)
                code.append(u)
                for v in neigh[u]:
                    counts[v] += 1

    best_code = code[:]
    best_size = len(code)
    
    # 5. Local Search (Combination of Current Program's Minimize and Inspiration's SA)
    # We use the "minimize" approach from the current program which is very effective at removing redundancy.
    # Then we use swaps to explore the space.
    
    def get_coverage(code_indices):
        cov = [0] * total_words
        for idx in code_indices:
            for nb in neigh[idx]:
                cov[nb] += 1
        return cov

    def is_valid_cov(cov):
        return all(c > 0 for c in cov)

    def minimize(code_indices):
        # Remove redundant codewords
        changed = True
        while changed:
            changed = False
            cov = get_coverage(code_indices)
            for i in range(len(code_indices)):
                # Try removing code_indices[i]
                test_cov = cov[:]
                for nb in neigh[code_indices[i]]:
                    test_cov[nb] -= 1
                if is_valid_cov(test_cov):
                    code_indices = code_indices[:i] + code_indices[i+1:]
                    cov = test_cov
                    changed = True
                    break
        return code_indices

    # Initial minimize
    code = minimize(code)
    best_code = code[:]
    best_size = len(code)
    
    # Simulated Annealing / Random Local Search
    # Moves:
    # 1. Remove: If a codeword is redundant, remove it. (Handled by minimize, but we can try random removals)
    # 2. Swap: Replace a codeword with another if the new one covers all unique neighbors of the old one.
    
    temp = 1.0
    cool_rate = 0.9995
    min_temp = 0.01
    
    def is_removable(c, cov):
        for u in neigh[c]:
            if cov[u] == 1:
                return False
        return True

    def is_valid_swap(c, new_c, cov):
        # Check if removing c leaves any neighbor uncovered by new_c
        for u in neigh[c]:
            if cov[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

    current_code = code[:]
    current_set = set(current_code)
    current_cov = get_coverage(current_code)
    
    while time.time() < deadline - 5.0:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try Remove
            if len(current_code) > 0:
                idx = rng.randint(0, len(current_code) - 1)
                c = current_code[idx]
                if is_removable(c, current_cov):
                    # Perform removal
                    current_code.pop(idx)
                    current_set.discard(c)
                    for u in neigh[c]:
                        current_cov[u] -= 1
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                        
        elif move_type < 0.8:
            # Try Swap
            if len(current_code) > 0:
                idx = rng.randint(0, len(current_code) - 1)
                old_c = current_code[idx]
                new_c = rng.randint(0, total_words - 1)
                
                if new_c == old_c or new_c in current_set:
                    continue
                    
                if is_valid_swap(old_c, new_c, current_cov):
                    # Perform swap
                    current_code[idx] = new_c
                    current_set.discard(old_c)
                    current_set.add(new_c)
                    
                    # Update coverage
                    for u in neigh[old_c]:
                        current_cov[u] -= 1
                    for u in neigh[new_c]:
                        current_cov[u] += 1
                        
                    # After a valid swap, try to minimize
                    # This is the key improvement: aggressive pruning after moves
                    current_code = minimize(current_code)
                    current_set = set(current_code)
                    current_cov = get_coverage(current_code)
                    
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                        
        else:
            # Try Add (less likely, but can help escape local minima if we get stuck)
            # Actually, adding increases size, so we only do this if we suspect we lost coverage?
            # No, we maintain coverage invariant. So adding is only useful if we want to swap later.
            # We skip pure adds to keep size down.
            pass
            
        # Cooling
        temp *= cool_rate
        
    # Final minimize on the best found code
    best_code = minimize(best_code)
    
    # Convert to words
    # We need to map integer indices back to tuples/lists
    # Assuming base-q representation where index i corresponds to digits d0...dn-1
    # i = d0 * q^(n-1) + d1 * q^(n-2) + ... + dn-1
    # Or the way we generated neighbors:
    # pow_q_i = q ** i, digit = (idx // pow_q_i) % q
    # This implies index = sum(digit_i * q^i)
    
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
