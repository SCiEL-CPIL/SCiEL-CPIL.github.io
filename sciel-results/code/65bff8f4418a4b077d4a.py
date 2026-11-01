import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors: neighbors[w] = list of 17 indices (w itself + 16 neighbors)
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1

    best_size = total
    best_set = None
    
    # Helper: greedy cover from a starting set
    def greedy_cover_from(start_selected, cover_count, uncovered_mask, uncovered_count):
        selected = list(start_selected)
        while uncovered_count > 0:
            # Count how many uncovered words each candidate covers
            counts = uncovered_mask[neighbors].sum(axis=1)
            
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            if time.time() > deadline - 5.0:
                break
        
        return selected, cover_count, uncovered_mask, uncovered_count

    # Helper: local search to improve
    def local_search(selected, cover_count):
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # Find removable codewords
            for i in range(len(selected)):
                cw = selected[i]
                if min_counts[cw] > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    break
            
            if not changed:
                # Try swaps
                if time.time() < deadline - 10.0 and len(selected) > 486:
                    for _ in range(10):  # Try a few swaps
                        if time.time() > deadline - 5.0:
                            break
                        
                        idx_to_swap = np.random.randint(len(selected))
                        cw_removed = selected[idx_to_swap]
                        
                        # Find words that would become uncovered
                        U = []
                        for w in neighbors[cw_removed]:
                            if cover_count[w] == 1:
                                U.append(w)
                        
                        U = np.array(U, dtype=np.int32)
                        
                        if len(U) == 0:
                            selected.pop(idx_to_swap)
                            cover_count[neighbors[cw_removed]] -= 1
                            changed = True
                            break
                        
                        selected_set = set(selected)
                        replacement_found = False
                        
                        # Find a replacement codeword that covers all words in U
                        for cw2 in neighbors[U[0]]:
                            if cw2 == cw_removed:
                                continue
                            if cw2 in selected_set:
                                continue
                            
                            covers_all = True
                            for w in U:
                                if w not in neighbors[cw2]:
                                    covers_all = False
                                    break
                            
                            if covers_all:
                                selected[idx_to_swap] = cw2
                                cover_count[neighbors[cw_removed]] -= 1
                                cover_count[neighbors[cw2]] += 1
                                replacement_found = True
                                changed = True
                                break
                        
                        if replacement_found:
                            break
        
        return selected, cover_count

    # Strategy: Latin hypercube-inspired initialization
    # Create a diverse initial set with good spread
    def create_latin_hypercube_init():
        # Use a structured approach: pick codewords that are spread out
        # We'll use a grid-based approach in the 8-dimensional space
        # Divide each dimension into 3 parts, pick one representative per hypercube cell
        # But we want fewer than 3^8 = 6561, so we'll subsample
        
        # Start with a smaller structured set
        # Use the tetracode structure as a base, then extend
        base = []
        for a in range(3):
            for b in range(3):
                word = [a, b, (a + b) % 3, (a + 2 * b) % 3, 0, 0, 0, 0]
                base.append(word)
        
        # Convert to indices
        base_indices = []
        for word in base:
            idx = 0
            for digit in word:
                idx = idx * 3 + digit
            base_indices.append(idx)
        
        return base_indices

    # Main loop
    while time.time() < deadline - 2.0:
        # 1. Latin hypercube initialization
        init_selected = create_latin_hypercube_init()
        
        # Compute initial coverage
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        for cw in init_selected:
            ball = neighbors[cw]
            cover_count[ball] += 1
            uncovered_mask[ball] = False
        
        uncovered_count = np.sum(uncovered_mask)
        
        # 2. Greedy completion
        selected, cover_count, uncovered_mask, uncovered_count = greedy_cover_from(
            init_selected, cover_count, uncovered_mask, uncovered_count
        )
        
        if uncovered_count > 0:
            # Greedy didn't finish, try plain greedy from scratch
            cover_count = np.zeros(total, dtype=np.int32)
            uncovered_mask = np.ones(total, dtype=bool)
            selected = []
            uncovered_count = total
            
            while uncovered_count > 0 and time.time() < deadline - 5.0:
                counts = uncovered_mask[neighbors].sum(axis=1)
                max_count = np.max(counts)
                if max_count == 0:
                    break
                candidates = np.where(counts == max_count)[0]
                best_idx = candidates[np.random.randint(len(candidates))]
                selected.append(best_idx)
                ball = neighbors[best_idx]
                was_uncovered = uncovered_mask[ball]
                num_newly_covered = np.sum(was_uncovered)
                uncovered_count -= num_newly_covered
                cover_count[ball] += 1
                uncovered_mask[ball] = False
            
            if uncovered_count > 0:
                continue
        
        # 3. Local search
        selected, cover_count = local_search(selected, cover_count)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486:
                break
    
    # Convert best_set to list of digit lists
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n).ljust(n, '0')) for w in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
