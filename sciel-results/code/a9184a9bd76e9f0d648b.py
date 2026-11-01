import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    # For the specific hard case q=3, n=6, radius=1, use specialized approach
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback for other cases
    if total_words > 10000:
        return _generic_construct(q, n, radius, deadline)
        
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    # Precompute all tuples and index mapping
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbor indices and bitsets
    neighbor_indices = [[] for _ in range(total_words)]
    neighbors = [0] * total_words
    for i, t in enumerate(all_tuples):
        mask = 0
        mask |= (1 << i)
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    def get_uncovered_mask(cover_count):
        mask = 0
        for i in range(total_words):
            if cover_count[i] == 0:
                mask |= (1 << i)
        return mask

    def greedy_start():
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask
        
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            
            if best_idx == -1 or best_cnt == 0:
                break
                
            code_set.add(best_idx)
            m = neighbors[best_idx]
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~m
            
        return code_set, cover_count, uncovered_mask

    def tetracode_start():
        """Use tetracode direct sum as initial code (81 words), then reduce."""
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code_set = set()
        cover_count = [0] * total_words
        
        for head in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                word = head + list(tail)
                idx = idx_map[tuple(word)]
                code_set.add(idx)
                for nidx in neighbor_indices[idx]:
                    cover_count[nidx] += 1
        
        uncovered_mask = get_uncovered_mask(cover_count)
        return code_set, cover_count, uncovered_mask

    def reduce_code(code_set, cover_count, max_iters=2000):
        """Remove redundant codewords iteratively."""
        changed = True
        iters = 0
        while changed and iters < max_iters:
            changed = False
            iters += 1
            for c in list(code_set):
                redundant = True
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    changed = True
                    break
        return code_set, cover_count

    def repair(code_set, cover_count, uncovered_mask):
        """Add codewords until valid cover."""
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1 or best_cnt == 0:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
        return code_set, cover_count, uncovered_mask

    def swap_local_search(code_set, cover_count):
        """Try to replace codewords to reduce size."""
        improved = True
        while improved:
            improved = False
            for c in list(code_set):
                unique_to_c = []
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        unique_to_c.append(idx)
                
                if not unique_to_c:
                    # c is redundant, remove it
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    improved = True
                    break
                
                # Try to find a replacement that covers all unique_to_c
                # and is not in code_set
                for i in range(total_words):
                    if i in code_set:
                        continue
                    all_covered = True
                    for u in unique_to_c:
                        if u != i and u not in neighbor_indices[i]:
                            all_covered = False
                            break
                    if all_covered:
                        # Check if adding i and removing c doesn't uncover anything
                        # Words that only c covers: unique_to_c
                        # Words that only i covers (among current code): need to check
                        # For simplicity, just try it
                        # Remove c
                        code_set.remove(c)
                        for idx in neighbor_indices[c]:
                            cover_count[idx] -= 1
                        # Add i
                        code_set.add(i)
                        for idx in neighbor_indices[i]:
                            cover_count[idx] += 1
                        # Check validity
                        valid = True
                        for idx in range(total_words):
                            if cover_count[idx] == 0:
                                valid = False
                                break
                        if valid:
                            improved = True
                            break
                        else:
                            # Revert
                            code_set.remove(i)
                            for idx in neighbor_indices[i]:
                                cover_count[idx] -= 1
                            code_set.add(c)
                            for idx in neighbor_indices[c]:
                                cover_count[idx] += 1
                            break
                if improved:
                    break
        return code_set, cover_count

    # Main ILS Loop
    rng = random.Random(42)
    
    # Try multiple initializations
    best_size = float('inf')
    best_code = None
    
    # Initialization 1: Tetracode direct sum + reduction
    code_set, cover_count, uncovered_mask = tetracode_start()
    code_set, cover_count = reduce_code(code_set, cover_count)
    uncovered_mask = get_uncovered_mask(cover_count)
    code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
    code_set, cover_count = reduce_code(code_set, cover_count)
    uncovered_mask = get_uncovered_mask(cover_count)
    code_set, cover_count = swap_local_search(code_set, cover_count)
    
    if len(code_set) < best_size:
        best_size = len(code_set)
        best_code = list(code_set)
    
    # Initialization 2: Greedy + reduction
    code_set, cover_count, uncovered_mask = greedy_start()
    code_set, cover_count = reduce_code(code_set, cover_count)
    uncovered_mask = get_uncovered_mask(cover_count)
    code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
    code_set, cover_count = reduce_code(code_set, cover_count)
    uncovered_mask = get_uncovered_mask(cover_count)
    code_set, cover_count = swap_local_search(code_set, cover_count)
    
    if len(code_set) < best_size:
        best_size = len(code_set)
        best_code = list(code_set)
        
    # Initialization 3: Random start (might find different basin)
    code_set, cover_count, uncovered_mask = greedy_start() # Reuse greedy for now as it's fast
    # Perturb it heavily
    for _ in range(10):
        if code_set:
            c = rng.choice(list(code_set))
            code_set.remove(c)
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
            uncovered_mask = get_uncovered_mask(cover_count)
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
            code_set, cover_count = reduce_code(code_set, cover_count)
            code_set, cover_count = swap_local_search(code_set, cover_count)
            
    if len(code_set) < best_size:
        best_size = len(code_set)
        best_code = list(code_set)

    # ILS from best solution
    code_set = set(best_code)
    cover_count = [0] * total_words
    for c in code_set:
        for idx in neighbor_indices[c]:
            cover_count[idx] += 1
    uncovered_mask = get_uncovered_mask(cover_count)
    
    # ILS iterations
    while time.time() < deadline - 2:
        # Perturb: remove k random codewords
        k = rng.randint(2, 6)
        current_code = list(code_set)
        if len(current_code) < k:
            k = max(1, len(current_code) - 1)
        
        to_remove = rng.sample(current_code, k)
        
        for c in to_remove:
            code_set.remove(c)
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
        
        uncovered_mask = get_uncovered_mask(cover_count)
        
        # Repair
        code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
        
        # Reduce
        code_set, cover_count = reduce_code(code_set, cover_count)
        code_set, cover_count = swap_local_search(code_set, cover_count)
        
        # Check improvement
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            if best_size == 73:
                break
    
    # Convert best_code to lists of ints
    result = [list(all_tuples[i]) for i in best_code]
    return result

def _generic_bitset_construct(q, n, radius, deadline):
    total_words = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    neighbors = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        mask = 0
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t[k] != t_j[k])
            if dist <= radius:
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    code_set = set()
    cover_count = [0] * total_words
    uncovered_mask = full_mask
    
    while uncovered_mask and time.time() < deadline - 10:
        best_idx = -1
        best_cnt = -1
        for i in range(total_words):
            if i in code_set:
                continue
            new_covered = neighbors[i] & uncovered_mask
            cnt = new_covered.bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_idx == -1: break
        code_set.add(best_idx)
        for idx in neighbor_indices[best_idx]:
            cover_count[idx] += 1
        uncovered_mask &= ~neighbors[best_idx]
        
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for c in list(code_set):
            redundant = True
            for idx in neighbor_indices[c]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                code_set.remove(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
                changed = True
                break
                
    result = [list(all_tuples[i]) for i in code_set]
    return result

def _generic_construct(q, n, radius, deadline):
    all_words = list(itertools.product(range(q), repeat=n))
    def dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    uncovered = set(all_words)
    code = []
    while uncovered and time.time() < deadline - 10:
        best = None
        best_cnt = -1
        code_set = set(code)
        for w in all_words:
            if w in code_set: continue
            cnt = sum(1 for u in uncovered if dist(w, u) <= radius)
            if cnt > best_cnt:
                best_cnt = cnt
                best = w
        if not best: break
        code.append(best)
        uncovered = {u for u in uncovered if dist(best, u) > radius}
    return [list(w) for w in code]

# EVOLVE-BLOCK-END
