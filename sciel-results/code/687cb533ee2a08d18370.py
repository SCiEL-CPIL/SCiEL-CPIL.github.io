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
    
    # For the specific hard case q=3, n=6, radius=1, use specialized ILS
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback for other cases
    if total_words > 10000:
        # Too large for bitset approach in pure python quickly, use generic greedy
        return _generic_construct(q, n, radius, deadline)
        
    # Generic bitset-based solver for small spaces
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    # Precompute neighbors as bitsets
    # Map tuple to index
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbor lists (indices) for faster iteration in some contexts
    # But for bitset operations, we need the mask.
    # Let's precompute the list of neighbor indices for each word, 
    # and also the bitset mask.
    neighbor_indices = [[] for _ in range(total_words)]
    neighbors = [0] * total_words
    for i, t in enumerate(all_tuples):
        mask = 0
        # Self
        mask |= (1 << i)
        neighbor_indices[i].append(i)
        # Neighbors at distance 1
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                # Create new tuple
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    # Helper to calculate uncovered mask from cover_count
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
            # Optimization: only consider words not in code? 
            # Actually, adding a word already in code is useless.
            # Iterate all words, skip if in code_set
            # For speed, we can just iterate all. 729 is small.
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
                
            # Add best_idx
            code_set.add(best_idx)
            m = neighbors[best_idx]
            # Update cover_count
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            
            uncovered_mask &= ~m
            
        return code_set, cover_count, uncovered_mask

    def reduce_code(code_set, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            # Iterate over a copy because we might remove
            for c in list(code_set):
                # Check if c is redundant
                # c is redundant if for all w in neighbors[c], cover_count[w] > 1
                redundant = True
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove c
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    changed = True
                    break # Restart loop to check new state
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
                # Should not happen if space is connected
                break
                
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
        return code_set, cover_count, uncovered_mask

    # Main ILS Loop
    rng = random.Random(42)
    
    # Initial solution
    code_set, cover_count, uncovered_mask = greedy_start()
    code_set, cover_count = reduce_code(code_set, cover_count)
    
    # Calculate initial uncovered (should be 0)
    # Re-calculate uncovered_mask from cover_count for safety
    uncovered_mask = get_uncovered_mask(cover_count)
    
    best_size = len(code_set)
    best_code = list(code_set)
    
    # ILS
    while time.time() < deadline - 2:
        # Perturb
        # Remove k random codewords
        k = rng.randint(2, 5)
        current_code = list(code_set)
        if len(current_code) < k:
            k = len(current_code)
        
        to_remove = rng.sample(current_code, k)
        
        # Apply removal
        for c in to_remove:
            code_set.remove(c)
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
        
        # Recalculate uncovered
        uncovered_mask = get_uncovered_mask(cover_count)
        
        # Repair
        code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
        
        # Reduce
        code_set, cover_count = reduce_code(code_set, cover_count)
        
        # Check improvement
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            
            # Optional: If we hit 73, we can stop early?
            # 73 is the known optimum.
            if best_size == 73:
                break

    # Convert best_code to lists of ints
    result = [list(all_tuples[i]) for i in best_code]
    return result

def _generic_bitset_construct(q, n, radius, deadline):
    total_words = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbors
    neighbors = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        mask = 0
        # Generate all words within radius
        # For small n, we can just iterate all j and check distance
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t[k] != t_j[k])
            if dist <= radius:
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    # Greedy
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
        
    # Simple local search: remove redundant
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
