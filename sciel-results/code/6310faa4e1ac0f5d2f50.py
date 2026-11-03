import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces or general radius
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based approach for small spaces (e.g., 3^7 = 2187)
    
    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Precompute balls as bitmasks (Python ints)
    balls = [0] * total_words
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            # Self
            mask |= (1 << i)
            # Neighbors
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        # Calculate neighbor index directly
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        # General radius (slower)
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask

    all_ones_mask = (1 << total_words) - 1
    
    def greedy_cover(uncovered_mask, random_seed=None):
        """
        Greedily select codewords to cover the words in uncovered_mask.
        Returns a list of indices.
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        selected = []
        remaining = uncovered_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            candidates = []
            for i in range(total_words):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                    candidates = [i]
                elif c == max_cover and c > 0:
                    candidates.append(i)
            
            if best_idx == -1:
                break
            
            # Randomly pick from candidates to diversify
            if len(candidates) > 1:
                best_idx = random.choice(candidates)
                
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
        return selected

    def remove_redundants(selected):
        """
        Removes any codeword from selected that is redundant.
        Returns the new list of selected indices.
        """
        if not selected:
            return []
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break
        return selected

    def local_search(selected, time_limit):
        """
        Intensive local search: 2-1 moves and redundancy removal.
        """
        current_selected = selected[:]
        best_size = len(current_selected)
        best_code = current_selected[:]
        
        # Remove redundants first
        current_selected = remove_redundants(current_selected)
        best_size = len(current_selected)
        best_code = current_selected[:]
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try 2-1 Local Search
            # We try a few random pairs
            attempts = 10
            for _ in range(attempts):
                if len(current_selected) < 3:
                    break
                i1, i2 = random.sample(range(len(current_selected)), 2)
                c1 = current_selected[i1]
                c2 = current_selected[i2]
                
                # Calculate coverage of the rest
                rest_mask = 0
                for i in range(len(current_selected)):
                    if i != i1 and i != i2:
                        rest_mask |= balls[current_selected[i]]
                    
                uncovered = all_ones_mask & ~rest_mask
                
                if uncovered == 0:
                    # Both redundant?
                    new_selected = [x for x in current_selected if x != c1 and x != c2]
                    new_selected = remove_redundants(new_selected)
                    if len(new_selected) < len(current_selected):
                        current_selected = new_selected
                        improved = True
                        if len(current_selected) < best_size:
                            best_size = len(current_selected)
                            best_code = current_selected[:]
                        break
                    continue

                # Find the best single word to cover uncovered
                best_idx = -1
                max_cover = 0
                candidates = []
                for k in range(total_words):
                    if k == c1 or k == c2:
                        continue
                    c = (balls[k] & uncovered).bit_count()
                    if c > max_cover:
                        max_cover = c
                        best_idx = k
                        candidates = [k]
                    elif c == max_cover and c > 0:
                        candidates.append(k)
                
                if best_idx != -1 and (balls[best_idx] & uncovered) == uncovered:
                    new_selected = [x for x in current_selected if x != c1 and x != c2]
                    if len(candidates) > 1:
                        best_idx = random.choice(candidates)
                    new_selected.append(best_idx)
                    new_selected = remove_redundants(new_selected)
                    if len(new_selected) < len(current_selected):
                        current_selected = new_selected
                        improved = True
                        if len(current_selected) < best_size:
                            best_size = len(current_selected)
                            best_code = current_selected[:]
                        break
            
            if improved:
                continue

            # 2. Perturb and Repair (Simulated Annealing style)
            # If stuck, perturb
            perturb_amount = random.randint(2, 5)
            if len(current_selected) <= perturb_amount:
                # Reset to a fresh greedy if too small
                current_selected = greedy_cover(all_ones_mask, random_seed=random.randint(0, 10000))
                current_selected = remove_redundants(current_selected)
            else:
                indices_to_remove = random.sample(range(len(current_selected)), perturb_amount)
                indices_to_remove_set = set(indices_to_remove)
                
                covered_mask = 0
                for i in range(len(current_selected)):
                    if i not in indices_to_remove_set:
                        covered_mask |= balls[current_selected[i]]
                
                uncovered_mask = all_ones_mask & ~covered_mask
                
                repair_selected = greedy_cover(uncovered_mask, random_seed=random.randint(0, 10000))
                
                new_selected = []
                for i in range(len(current_selected)):
                    if i not in indices_to_remove_set:
                        new_selected.append(current_selected[i])
                new_selected.extend(repair_selected)
                
                new_selected = remove_redundants(new_selected)
                
                # Accept if better or with some probability if worse (annealing)
                # Simple: accept if better, or 10% if worse but not too much worse
                if len(new_selected) < len(current_selected):
                    current_selected = new_selected
                elif random.random() < 0.1 and len(new_selected) < len(current_selected) + 3:
                    current_selected = new_selected

        return best_code

    best_code = None
    best_size = total_words + 1
    start_time = time.time()
    time_limit = 115.0
    
    # Multi-start Greedy + Local Search
    # Run several greedy starts, keep the best, then refine
    candidates = []
    
    # Run 20 greedy starts
    num_starts = 20
    for i in range(num_starts):
        if time.time() - start_time > time_limit - 20:
            break
        sel = greedy_cover(all_ones_mask, random_seed=i)
        sel = remove_redundants(sel)
        if len(sel) < best_size:
            best_size = len(sel)
            best_code = [all_words[j] for j in sel]
        candidates.append(sel)
        
    # Sort candidates by size
    candidates.sort(key=len)
    
    # Refine the top candidates
    # We have ~90 seconds left for local search
    # Try to refine the best one, then second best, etc.
    for sel in candidates[:5]: # Top 5
        if time.time() - start_time > time_limit - 5:
            break
            
        # Run local search with a time slice
        slice_end = time.time() + 15.0
        if time_limit - time.time() < 15.0:
            slice_end = time_limit - 1.0
            
        refined = local_search(sel, slice_end)
        
        if len(refined) < best_size:
            best_size = len(refined)
            best_code = [all_words[j] for j in refined]

    return best_code
