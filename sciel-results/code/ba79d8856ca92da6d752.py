import time
import random

def construct():
    n = 12
    m = 40
    # Use a safety margin well under the 90s wall-clock limit
    deadline = time.time() + 82.0
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    def calculate_score(masks):
        score = 0
        for i in range(m):
            for j in range(i+1, m):
                if is_comparable(masks[i], masks[j]):
                    score += 1
        return score

    def score_delta(swap_idx, new_mask, masks):
        old_mask = masks[swap_idx]
        delta = 0
        for j in range(m):
            if j == swap_idx: continue
            other = masks[j]
            if is_comparable(new_mask, other):
                delta += 1
            if is_comparable(old_mask, other):
                delta -= 1
        return delta

    def generate_structured_candidate(existing_masks, exclude_mask):
        """Generate a candidate mask that is likely comparable with existing masks.
        Strategy: pick a random existing mask, then either add or remove 1-2 bits."""
        # Pick a random existing mask as base
        base = random.choice(existing_masks)
        if base == exclude_mask:
            # Pick another one
            base = random.choice(existing_masks)
        
        # Decide: extend (add bits) or restrict (remove bits)
        if random.random() < 0.5:
            # Add 1-2 bits to base
            new_mask = base
            bits_to_add = [i for i in range(n) if not (new_mask >> i) & 1]
            random.shuffle(bits_to_add)
            num_add = min(random.randint(1, 2), len(bits_to_add))
            for i in range(num_add):
                new_mask |= (1 << bits_to_add[i])
        else:
            # Remove 1-2 bits from base
            new_mask = base
            bits_to_remove = [i for i in range(n) if (new_mask >> i) & 1]
            random.shuffle(bits_to_remove)
            num_remove = min(random.randint(1, 2), len(bits_to_remove))
            for i in range(num_remove):
                new_mask &= ~(1 << bits_to_remove[i])
        
        return new_mask

    def greedy_initialize():
        """Greedily build a set of m masks maximizing comparable pairs."""
        selected = []
        selected_set = set()
        
        # Start with a few random masks
        initial_count = random.randint(3, 6)
        for _ in range(initial_count):
            mask = random.randint(0, (1 << n) - 1)
            while mask in selected_set:
                mask = random.randint(0, (1 << n) - 1)
            selected.append(mask)
            selected_set.add(mask)
        
        # Greedily add masks that maximize comparable pairs with existing
        while len(selected) < m:
            best_mask = None
            best_score = -1
            
            # Try a pool of candidates
            candidates = set()
            # Structured candidates from existing masks
            for existing in selected:
                # Add 1-2 bits
                new_m = existing
                bits_to_add = [i for i in range(n) if not (new_m >> i) & 1]
                random.shuffle(bits_to_add)
                for k in range(min(2, len(bits_to_add))):
                    new_m |= (1 << bits_to_add[k])
                    candidates.add(new_m)
                # Remove 1-2 bits
                new_m = existing
                bits_to_remove = [i for i in range(n) if (new_m >> i) & 1]
                random.shuffle(bits_to_remove)
                for k in range(min(2, len(bits_to_remove))):
                    new_m &= ~(1 << bits_to_remove[k])
                    candidates.add(new_m)
            
            # Add some random candidates
            for _ in range(20):
                candidates.add(random.randint(0, (1 << n) - 1))
            
            # Score each candidate
            for cand in candidates:
                if cand in selected_set:
                    continue
                # Count comparable pairs with selected
                comp_count = 0
                for sel in selected:
                    if is_comparable(cand, sel):
                        comp_count += 1
                if comp_count > best_score:
                    best_score = comp_count
                    best_mask = cand
            
            if best_mask is None:
                # Fallback: pick a random unselected mask
                while True:
                    best_mask = random.randint(0, (1 << n) - 1)
                    if best_mask not in selected_set:
                        break
            
            selected.append(best_mask)
            selected_set.add(best_mask)
        
        return selected

    def exact_local_search(masks, time_limit):
        """Exact local search: try all single-mask replacements, pick best improvement."""
        masks = masks[:]
        current_score = calculate_score(masks)
        mask_set = set(masks)
        
        start_time = time.time()
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            best_delta = 0
            best_swap_idx = -1
            best_new_mask = -1
            
            # Try replacing each mask with a better alternative
            for i in range(m):
                old_mask = masks[i]
                
                # Generate candidate replacements
                candidates = set()
                
                # Structured: add/remove 1-2 bits from existing masks
                for j in range(m):
                    if j == i:
                        continue
                    base = masks[j]
                    # Add 1-2 bits
                    new_m = base
                    bits_to_add = [k for k in range(n) if not (new_m >> k) & 1]
                    random.shuffle(bits_to_add)
                    for k in range(min(2, len(bits_to_add))):
                        new_m |= (1 << bits_to_add[k])
                        candidates.add(new_m)
                    # Remove 1-2 bits
                    new_m = base
                    bits_to_remove = [k for k in range(n) if (new_m >> k) & 1]
                    random.shuffle(bits_to_remove)
                    for k in range(min(2, len(bits_to_remove))):
                        new_m &= ~(1 << bits_to_remove[k])
                        candidates.add(new_m)
                
                # Add some random candidates
                for _ in range(10):
                    candidates.add(random.randint(0, (1 << n) - 1))
                
                # Evaluate each candidate
                for cand in candidates:
                    if cand == old_mask or cand in mask_set:
                        continue
                    delta = score_delta(i, cand, masks)
                    if delta > best_delta:
                        best_delta = delta
                        best_swap_idx = i
                        best_new_mask = cand
            
            if best_delta > 0 and best_swap_idx >= 0:
                old = masks[best_swap_idx]
                mask_set.discard(old)
                mask_set.add(best_new_mask)
                masks[best_swap_idx] = best_new_mask
                current_score += best_delta
                improved = True
        
        return masks, current_score

    def final_polish(masks, time_limit):
        """A final local search pass with only improving moves and tighter step cap."""
        masks = masks[:]
        current_score = calculate_score(masks)
        mask_set = set(masks)
        
        start_time = time.time()
        max_steps = 50000
        steps = 0
        no_improve = 0
        max_no_improve = 2000
        
        while steps < max_steps and no_improve < max_no_improve and time.time() < start_time + time_limit:
            steps += 1
            no_improve += 1
            
            i = random.randint(0, m - 1)
            old_mask = masks[i]
            
            # Generate candidate: 70% structured, 20% random flip, 10% fully random
            r = random.random()
            if r < 0.7:
                new_mask = generate_structured_candidate(masks, old_mask)
            elif r < 0.9:
                new_mask = old_mask
                num_flips = random.randint(1, 2)
                for _ in range(num_flips):
                    bit = random.randint(0, n - 1)
                    new_mask ^= (1 << bit)
            else:
                new_mask = random.randint(0, (1 << n) - 1)
            
            if new_mask == old_mask:
                continue
            
            # Fast duplicate check using set
            if new_mask in mask_set and new_mask != old_mask:
                continue
            
            delta = score_delta(i, new_mask, masks)
            
            if delta > 0:
                # Update set
                mask_set.discard(old_mask)
                mask_set.add(new_mask)
                masks[i] = new_mask
                current_score += delta
                no_improve = 0
        
        return masks, current_score

    best_score = -1
    best_masks = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Initialization
        if restart % 4 == 0:
            # Greedy initialization
            masks = greedy_initialize()
        elif restart % 4 == 1:
            # Random
            masks = list(random.sample(range(1 << n), m))
        elif restart % 4 == 2:
            # Chain-based init: build a maximal chain, then fill rest
            chain = []
            current = 0
            order = list(range(n))
            random.shuffle(order)
            chain.append(current)
            for idx in order:
                current |= (1 << idx)
                chain.append(current)
            # Now we have 13 chain elements. Add random for the rest
            existing = set(chain)
            while len(existing) < m:
                nm = random.randint(0, (1 << n) - 1)
                existing.add(nm)
            masks = list(existing)
            random.shuffle(masks)
            masks = masks[:m]
        else:
            # Clustered: Pick a random base, generate neighbors
            base = random.randint(0, (1 << n) - 1)
            candidates = set()
            candidates.add(base)
            # Generate 200 random masks near base
            for _ in range(200):
                b = base
                for _ in range(random.randint(1, 3)):
                    bit = random.randint(0, n-1)
                    b ^= (1 << bit)
                candidates.add(b)
            # Add some fully random
            for _ in range(50):
                candidates.add(random.randint(0, (1 << n) - 1))
            
            # Select 40 distinct
            masks_list = list(candidates)
            random.shuffle(masks_list)
            masks = masks_list[:m]
            # Ensure distinct
            if len(masks) < m:
                existing = set(masks)
                while len(masks) < m:
                    nm = random.randint(0, (1 << n) - 1)
                    if nm not in existing:
                        masks.append(nm)
                        existing.add(nm)

        current_score = calculate_score(masks)
        
        # Simulated Annealing with structured mutations
        T = 10.0
        min_T = 0.05
        cooling = 0.999
        no_improve_count = 0
        max_no_improve = 5000
        
        local_best_score = current_score
        local_best_masks = masks[:]
        mask_set = set(masks)
        
        steps_in_run = 0
        max_steps_per_run = 20000
        
        while steps_in_run < max_steps_per_run and time.time() < deadline:
            steps_in_run += 1
            no_improve_count += 1
            
            # Generate Neighbor
            i = random.randint(0, m - 1)
            old_mask = masks[i]
            
            # 60% Structured, 25% Random flip, 15% Fully random
            r = random.random()
            if r < 0.6:
                new_mask = generate_structured_candidate(masks, old_mask)
            elif r < 0.85:
                new_mask = old_mask
                num_flips = random.randint(1, 3)
                for _ in range(num_flips):
                    bit = random.randint(0, n - 1)
                    new_mask ^= (1 << bit)
            else:
                new_mask = random.randint(0, (1 << n) - 1)
            
            if new_mask == old_mask:
                continue
                
            # Fast duplicate check
            if new_mask in mask_set and new_mask != old_mask:
                continue
            
            delta = score_delta(i, new_mask, masks)
            
            # Acceptance
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                mask_set.discard(old_mask)
                mask_set.add(new_mask)
                masks[i] = new_mask
                current_score += delta
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_masks = masks[:]
                    no_improve_count = 0
                    
            T *= cooling
            
            if no_improve_count > max_no_improve:
                break
        
        # Final polish pass on the best local solution
        if time.time() < deadline - 5.0:
            polished_masks, polished_score = final_polish(local_best_masks, 3.0)
            if polished_score > local_best_score:
                local_best_score = polished_score
                local_best_masks = polished_masks
        
        # Exact local search polish
        if time.time() < deadline - 3.0:
            exact_masks, exact_score = exact_local_search(local_best_masks, 2.0)
            if exact_score > local_best_score:
                local_best_score = exact_score
                local_best_masks = exact_masks
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_masks = local_best_masks[:]
            
    if best_masks is None:
        best_masks = list(range(m))
        
    return [mask_to_list(mask) for mask in best_masks]
