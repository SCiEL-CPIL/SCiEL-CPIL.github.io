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
    
    def greedy_cover(uncovered_mask):
        """
        Greedily select codewords to cover the words in uncovered_mask.
        Returns a list of indices.
        """
        selected = []
        remaining = uncovered_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            # Optimization: Skip words that are already selected? 
            # Actually, we just need to find the best candidate from ALL words.
            # If a word is already selected, its ball is already in the union, 
            # so adding it again doesn't help cover 'remaining'. 
            # But 'remaining' is updated by removing covered bits.
            # So we just pick the best from all words.
            for i in range(total_words):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
                elif c == max_cover and c > 0:
                    if random.random() < 0.5:
                        best_idx = i
            if best_idx == -1:
                break
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

    def local_search_2_1(selected):
        """
        Try to remove 2 codewords and add 1 to reduce size by 1.
        Returns new selected list if improved, else None.
        """
        if len(selected) < 3:
            return None
            
        # Try more random pairs
        for _ in range(20):
            i1, i2 = random.sample(range(len(selected)), 2)
            c1 = selected[i1]
            c2 = selected[i2]
            
            # Calculate coverage of the rest
            rest_mask = 0
            for i in range(len(selected)):
                if i != i1 and i != i2:
                    rest_mask |= balls[selected[i]]
                    
            uncovered = all_ones_mask & ~rest_mask
            
            if uncovered == 0:
                # Both redundant?
                new_selected = [x for x in selected if x != c1 and x != c2]
                new_selected = remove_redundants(new_selected)
                if len(new_selected) < len(selected):
                    return new_selected
                continue

            # Find the best single word to cover uncovered
            best_idx = -1
            max_cover = 0
            for k in range(total_words):
                if k == c1 or k == c2:
                    continue
                c = (balls[k] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = k
                elif c == max_cover and c > 0:
                    if random.random() < 0.5:
                        best_idx = k
            
            if best_idx != -1 and (balls[best_idx] & uncovered) == uncovered:
                new_selected = [x for x in selected if x != c1 and x != c2]
                new_selected.append(best_idx)
                new_selected = remove_redundants(new_selected)
                if len(new_selected) < len(selected):
                    return new_selected
        return None

    def local_search_3_2(selected):
        """
        Try to remove 3 codewords and add 2 to reduce size by 1.
        """
        if len(selected) < 5:
            return None
            
        for _ in range(10):
            i1, i2, i3 = random.sample(range(len(selected)), 3)
            c1, c2, c3 = selected[i1], selected[i2], selected[i3]
            
            rest_mask = 0
            for i in range(len(selected)):
                if i != i1 and i != i2 and i != i3:
                    rest_mask |= balls[selected[i]]
                    
            uncovered = all_ones_mask & ~rest_mask
            
            if uncovered == 0:
                new_selected = [x for x in selected if x not in (c1, c2, c3)]
                new_selected = remove_redundants(new_selected)
                if len(new_selected) < len(selected):
                    return new_selected
                continue

            # Find best 2 words to cover uncovered
            # This is harder. Greedy pick 1, then pick 1 more?
            # Or just check if any pair covers it.
            # Simplified: Pick best 1, see if remaining can be covered by 1 more.
            
            best_pair = None
            best_cover = 0
            
            # Heuristic: Pick the word that covers most of 'uncovered'
            candidates = []
            for k in range(total_words):
                if k in (c1, c2, c3): continue
                c = (balls[k] & uncovered).bit_count()
                if c > 0:
                    candidates.append((c, k))
            
            candidates.sort(reverse=True)
            
            # Try top 5 candidates for first word
            for _, k1 in candidates[:5]:
                rem_uncovered = uncovered & ~balls[k1]
                if rem_uncovered == 0:
                    # Just 1 word needed? Then 3-1 move? 
                    # We are in 3-2, but if 1 word covers it, it's even better.
                    # But we are removing 3. If 1 word covers it, we remove 3 add 1.
                    # This reduces size by 2.
                    new_selected = [x for x in selected if x not in (c1, c2, c3)]
                    new_selected.append(k1)
                    new_selected = remove_redundants(new_selected)
                    if len(new_selected) < len(selected):
                        return new_selected
                    continue
                
                # Find best 2nd word for rem_uncovered
                best_k2 = -1
                max_c2 = 0
                for k2 in range(total_words):
                    if k2 in (c1, c2, c3, k1): continue
                    c2 = (balls[k2] & rem_uncovered).bit_count()
                    if c2 > max_c2:
                        max_c2 = c2
                        best_k2 = k2
                
                if best_k2 != -1 and (balls[best_k2] & rem_uncovered) == rem_uncovered:
                    new_selected = [x for x in selected if x not in (c1, c2, c3)]
                    new_selected.extend([k1, best_k2])
                    new_selected = remove_redundants(new_selected)
                    if len(new_selected) < len(selected):
                        return new_selected
        return None

    best_code = None
    best_size = total_words + 1
    start_time = time.time()
    time_limit = 115.0
    
    # Multi-start
    while time.time() - start_time < time_limit:
        # 1. Initial Greedy
        current_selected = greedy_cover(all_ones_mask)
        current_selected = remove_redundants(current_selected)
        
        if len(current_selected) < best_size:
            best_size = len(current_selected)
            best_code = [all_words[i] for i in current_selected]
        
        # 2. Local Search Loop
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Try 2-1
            new_sel = local_search_2_1(current_selected)
            if new_sel:
                current_selected = new_sel
                improved = True
                if len(current_selected) < best_size:
                    best_size = len(current_selected)
                    best_code = [all_words[i] for i in current_selected]
                continue
            
            # Try 3-2
            new_sel = local_search_3_2(current_selected)
            if new_sel:
                current_selected = new_sel
                improved = True
                if len(current_selected) < best_size:
                    best_size = len(current_selected)
                    best_code = [all_words[i] for i in current_selected]
                continue
            
            # 3. Perturb and Repair
            perturb_amount = random.randint(2, 5)
            if len(current_selected) <= perturb_amount:
                # Restart
                break
            
            indices_to_remove = random.sample(range(len(current_selected)), perturb_amount)
            indices_to_remove_set = set(indices_to_remove)
            
            covered_mask = 0
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    covered_mask |= balls[current_selected[i]]
            
            uncovered_mask = all_ones_mask & ~covered_mask
            
            repair_selected = greedy_cover(uncovered_mask)
            
            new_selected = []
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    new_selected.append(current_selected[i])
            new_selected.extend(repair_selected)
            
            new_selected = remove_redundants(new_selected)
            
            if len(new_selected) < best_size:
                best_size = len(new_selected)
                best_code = [all_words[i] for i in new_selected]
            
            # Accept if better or with small probability if similar
            if len(new_selected) <= len(current_selected) or (random.random() < 0.1 and len(new_selected) < len(current_selected) + 2):
                current_selected = new_selected
            else:
                # Keep current, but maybe try another perturbation?
                # The while loop will try again.
                pass

    return best_code
