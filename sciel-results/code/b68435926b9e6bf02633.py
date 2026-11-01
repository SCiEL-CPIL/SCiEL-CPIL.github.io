# EVOLVE-BLOCK-START
import time
import random
import itertools

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
    
    # For the specific hard case q=3, n=6, radius=1, use specialized group-structured ILS
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback for other cases
    if total_words > 10000:
        return _generic_construct(q, n, radius, deadline)
        
    # Generic bitset-based solver for small spaces
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    # Precompute neighbors as bitsets and indices
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbor lists and bitset masks
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
    
    # Precompute perturbation vectors in Z_3^6 for structured swaps
    perturbations = []
    for pos in range(n):
        for delta in (1, 2):  # +1 and -1 mod 3
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations.append(p)

    def is_valid(cover_count):
        # Check if all words are covered (count > 0)
        # Using min is fast in Python for lists of ints
        return min(cover_count) > 0

    def lagrangian_greedy(deadline):
        """
        Greedy construction using Lagrangian weights.
        """
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask
        weights = [1.0] * total_words
        
        while uncovered_mask and time.time() < deadline - 10:
            best_idx = -1
            best_score = -1.0
            
            # Iterate through candidates not in code
            # Optimization: only consider candidates that cover at least one uncovered word
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                if new_covered == 0:
                    continue
                
                # Calculate weighted score
                # Instead of iterating bits, we can just sum weights of newly covered
                # Since we only have 729 words, this is fast enough
                score = 0.0
                mask = new_covered
                while mask:
                    low_bit = mask & (-mask)
                    idx = low_bit.bit_length() - 1
                    score += weights[idx]
                    mask ^= low_bit
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx == -1:
                break
                
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            
            # Update weights: increase for uncovered, decrease for covered
            # This biases future choices towards covering "hard" spots
            for idx in range(total_words):
                if cover_count[idx] == 0:
                    weights[idx] *= 1.5
                else:
                    weights[idx] *= 0.9
            # Normalize to prevent overflow/underflow
            max_w = max(weights)
            if max_w > 0:
                weights = [w / max_w for w in weights]
                
            uncovered_mask &= ~neighbors[best_idx]
            
        return code_set, cover_count

    def reduce_code(code_set, cover_count):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Try to remove codewords in random order to avoid bias
            for c in list(code_set):
                # Check if c is redundant: if all its neighbors are covered by others
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
        while uncovered_mask and time.time() < deadline - 5:
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

    def local_search_1exchange(code_set, cover_count, rng, max_iters=100):
        """
        Try 1-exchange moves: replace codeword c with c+d.
        Accept if valid and size doesn't increase.
        """
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]
            
            if new_idx in code_set:
                continue
            
            # Try swap
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
            code_set.remove(c)
            
            code_set.add(new_idx)
            for idx in neighbor_indices[new_idx]:
                cover_count[idx] += 1
            
            # Check validity
            if is_valid(cover_count):
                improved = True
            else:
                # Revert
                code_set.remove(new_idx)
                for idx in neighbor_indices[new_idx]:
                    cover_count[idx] -= 1
                code_set.add(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] += 1
        
        return code_set, cover_count, improved

    def local_search_2exchange(code_set, cover_count, rng, max_iters=50):
        """
        Try 2-exchange moves: remove c1, c2 and add n1, n2.
        This can escape deeper local minima.
        """
        improved = False
        code_list = list(code_set)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if len(code_list) < 2:
                break
            c1, c2 = rng.sample(code_list, 2)
            if c1 == c2: continue
            
            d1 = rng.choice(perturbations)
            d2 = rng.choice(perturbations)
            
            c1_tuple = all_tuples[c1]
            c2_tuple = all_tuples[c2]
            
            n1_tuple = tuple((c1_tuple[i] + d1[i]) % q for i in range(n))
            n2_tuple = tuple((c2_tuple[i] + d2[i]) % q for i in range(n))
            
            n1_idx = idx_map[n1_tuple]
            n2_idx = idx_map[n2_tuple]
            
            # Check for conflicts
            if n1_idx in code_set or n2_idx in code_set:
                continue
            if n1_idx == n2_idx:
                continue
            
            # Perform swap
            for idx in neighbor_indices[c1]:
                cover_count[idx] -= 1
            for idx in neighbor_indices[c2]:
                cover_count[idx] -= 1
            code_set.remove(c1)
            code_set.remove(c2)
            
            code_set.add(n1_idx)
            code_set.add(n2_idx)
            for idx in neighbor_indices[n1_idx]:
                cover_count[idx] += 1
            for idx in neighbor_indices[n2_idx]:
                cover_count[idx] += 1
            
            if is_valid(cover_count):
                improved = True
                # Update local list if we want to be fancy, but re-listing next time is fine
            else:
                # Revert
                code_set.remove(n1_idx)
                code_set.remove(n2_idx)
                for idx in neighbor_indices[n1_idx]:
                    cover_count[idx] -= 1
                for idx in neighbor_indices[n2_idx]:
                    cover_count[idx] -= 1
                code_set.add(c1)
                code_set.add(c2)
                for idx in neighbor_indices[c1]:
                    cover_count[idx] += 1
                for idx in neighbor_indices[c2]:
                    cover_count[idx] += 1
        
        return code_set, cover_count, improved

    # Main ILS Loop
    rng = random.Random(42)
    
    # Initial solution using Lagrangian Greedy
    code_set, cover_count = lagrangian_greedy(deadline)
    code_set, cover_count = reduce_code(code_set, cover_count)
    
    best_size = len(code_set)
    best_code = list(code_set)
    
    # ILS
    iteration = 0
    while time.time() < deadline - 2:
        iteration += 1
        
        # Perturbation
        if rng.random() < 0.7:
            # 1-exchange perturbation
            code_set, cover_count, _ = local_search_1exchange(code_set, cover_count, rng, max_iters=20)
        else:
            # 2-exchange perturbation
            code_set, cover_count, _ = local_search_2exchange(code_set, cover_count, rng, max_iters=10)
            
            # Repair if invalid (though local search reverts invalid moves, 
            # sometimes we might want to accept invalid moves and repair, 
            # but standard ILS usually keeps validity. 
            # Let's stick to valid moves for stability, but maybe allow 
            # "simulated annealing" style acceptance? 
            # For now, let's just do a strong local search after perturbation)
        
        # Local Search (Refinement)
        # Alternate between 1-ex and 2-ex to ensure thoroughness
        if iteration % 2 == 0:
            code_set, cover_count, _ = local_search_1exchange(code_set, cover_count, rng, max_iters=50)
        else:
            code_set, cover_count, _ = local_search_2exchange(code_set, cover_count, rng, max_iters=20)
            
        # Reduce
        code_set, cover_count = reduce_code(code_set, cover_count)
        
        # Check improvement
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            if best_size == 73:
                break

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
        for w in all_words:
            if w in set(code): continue
            cnt = sum(1 for u in uncovered if dist(w, u) <= radius)
            if cnt > best_cnt:
                best_cnt = cnt
                best = w
        if not best: break
        code.append(best)
        uncovered = {u for u in uncovered if dist(best, u) > radius}
    return [list(w) for w in code]

# EVOLVE-BLOCK-END
