import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 88.0
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
    
    # Precompute popcount for faster counting
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def fast_popcount(x):
        count = 0
        while x:
            count += popcount[x & 0xFFFF]
            x >>= 16
        return count
    
    def mask_to_list(mask):
        """Convert bitmask to list of set bit positions."""
        result = []
        m = mask
        while m:
            lsb = m & (-m)
            result.append(lsb.bit_length() - 1)
            m ^= lsb
        return result
    
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = fast_popcount(current_mask)
    
    # Phase 1: Add evens with 0 conflicts
    changed = True
    while changed and time.time() < DEADLINE - 55:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 55:
                break
            if (current_mask >> e) & 1:
                continue
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
                best_mask = current_mask
                best_size = fast_popcount(current_mask)
                changed = True
    
    # Phase 1b: Try adding evens that cause minimal conflicts
    changed = True
    while changed and time.time() < DEADLINE - 50:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 50:
                break
            if (current_mask >> e) & 1:
                continue
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            if num_conflicts <= 1:
                new_mask = (current_mask & ~conflicts) | (1 << e)
                new_size = best_size - num_conflicts + 1
                if new_size > best_size:
                    current_mask = new_mask
                    best_mask = current_mask
                    best_size = new_size
                    changed = True
    
    # Phase 2: Simulated Annealing with optimized moves and restarts
    T = 2.0
    cooling = 0.9998
    max_iters = 300000
    
    current_size = best_size
    in_set = mask_to_list(current_mask)
    not_in_set = [i for i in range(N) if i not in set(in_set)]
    
    # Precompute for quick access
    evens_not_in = [i for i in evens_indices if i not in set(in_set)]
    odds_not_in = [i for i in odds_indices if i not in set(in_set)]
    
    for i_iter in range(max_iters):
        if time.time() > DEADLINE - 3:
            break
        
        r = random.random()
        if r < 0.5:
            # Single add move
            if random.random() < 0.7 and evens_not_in:
                e = random.choice(evens_not_in)
            elif odds_not_in:
                e = random.choice(odds_not_in)
            elif evens_not_in:
                e = random.choice(evens_not_in)
            else:
                continue
            
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            new_size = current_size - num_conflicts + 1
            
            if new_size > current_size:
                current_mask = (current_mask & ~conflicts) | (1 << e)
                current_size = new_size
                # Update lists
                in_set.append(e)
                if e in evens_not_in:
                    evens_not_in.remove(e)
                elif e in odds_not_in:
                    odds_not_in.remove(e)
                # Remove conflicts from in_set, add to not_in_set
                for c in mask_to_list(conflicts):
                    in_set.remove(c)
                    if c in evens_not_in:
                        pass  # already there
                    else:
                        evens_not_in.append(c) if c % 2 == 0 else odds_not_in.append(c)
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.4:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        in_set.append(e)
                        if e in evens_not_in:
                            evens_not_in.remove(e)
                        elif e in odds_not_in:
                            odds_not_in.remove(e)
                        for c in mask_to_list(conflicts):
                            in_set.remove(c)
                            if c % 2 == 0:
                                evens_not_in.append(c)
                            else:
                                odds_not_in.append(c)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        current_size = new_size
                        in_set.append(e)
                        if e in evens_not_in:
                            evens_not_in.remove(e)
                        elif e in odds_not_in:
                            odds_not_in.remove(e)
                        for c in mask_to_list(conflicts):
                            in_set.remove(c)
                            if c % 2 == 0:
                                evens_not_in.append(c)
                            else:
                                odds_not_in.append(c)
        elif r < 0.7:
            # Swap move: remove 1, add 1
            if not in_set:
                continue
            rm_idx = random.choice(in_set)
            temp_mask = current_mask & ~(1 << rm_idx)
            
            # Find candidate not in temp_mask
            candidates = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
            if not candidates:
                continue
            e = random.choice(candidates)
            
            conflicts_with_temp = conflict_mask[e] & temp_mask
            if conflicts_with_temp == 0:
                if random.random() < 0.3:
                    current_mask = temp_mask | (1 << e)
                    in_set.remove(rm_idx)
                    in_set.append(e)
                    if rm_idx % 2 == 0:
                        evens_not_in.append(rm_idx)
                    else:
                        odds_not_in.append(rm_idx)
                    if e in evens_not_in:
                        evens_not_in.remove(e)
                    elif e in odds_not_in:
                        odds_not_in.remove(e)
        elif r < 0.85:
            # Multi-element exchange: remove k, add 1
            k = random.choice([2, 3])
            if len(in_set) < k:
                continue
            removed = random.sample(in_set, k)
            
            temp_mask = current_mask
            for rm in removed:
                temp_mask &= ~(1 << rm)
            
            candidates = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
            if not candidates:
                continue
            
            zero_conflict = [idx for idx in candidates if (conflict_mask[idx] & temp_mask) == 0]
            if zero_conflict:
                e = random.choice(zero_conflict)
            else:
                e = random.choice(candidates)
            
            conflicts_with_temp = conflict_mask[e] & temp_mask
            if conflicts_with_temp != 0:
                num_conf = fast_popcount(conflicts_with_temp)
                new_size = current_size - k - num_conf + 1
                if new_size <= current_size:
                    continue
            
            new_size = current_size - k + 1
            
            if new_size > current_size:
                current_mask = temp_mask | (1 << e)
                current_size = new_size
                for rm in removed:
                    in_set.remove(rm)
                    if rm % 2 == 0:
                        evens_not_in.append(rm)
                    else:
                        odds_not_in.append(rm)
                in_set.append(e)
                if e in evens_not_in:
                    evens_not_in.remove(e)
                elif e in odds_not_in:
                    odds_not_in.remove(e)
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.3:
                        current_mask = temp_mask | (1 << e)
                        for rm in removed:
                            in_set.remove(rm)
                            if rm % 2 == 0:
                                evens_not_in.append(rm)
                            else:
                                odds_not_in.append(rm)
                        in_set.append(e)
                        if e in evens_not_in:
                            evens_not_in.remove(e)
                        elif e in odds_not_in:
                            odds_not_in.remove(e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = temp_mask | (1 << e)
                        current_size = new_size
                        for rm in removed:
                            in_set.remove(rm)
                            if rm % 2 == 0:
                                evens_not_in.append(rm)
                            else:
                                odds_not_in.append(rm)
                        in_set.append(e)
                        if e in evens_not_in:
                            evens_not_in.remove(e)
                        elif e in odds_not_in:
                            odds_not_in.remove(e)
        else:
            # Random removal
            if in_set:
                r_idx = random.choice(in_set)
                current_mask &= ~(1 << r_idx)
                current_size -= 1
                in_set.remove(r_idx)
                if r_idx % 2 == 0:
                    evens_not_in.append(r_idx)
                else:
                    odds_not_in.append(r_idx)
        
        T *= cooling
        if T < 0.01:
            T = 0.01
        
        # Restart mechanism: every 30000 iters or if stuck, restart from best
        if i_iter % 30000 == 0 and i_iter > 0:
            if current_size < best_size - 2:
                # Restart from best with perturbation
                current_mask = best_mask
                current_size = best_size
                in_set = mask_to_list(current_mask)
                evens_not_in = [i for i in evens_indices if i not in set(in_set)]
                odds_not_in = [i for i in odds_indices if i not in set(in_set)]
                T = 1.0
    
    # Final polish: greedy add any element with 0 conflicts
    changed = True
    while changed and time.time() < DEADLINE - 1:
        changed = False
        for idx in range(N):
            if time.time() > DEADLINE - 1:
                break
            if (best_mask >> idx) & 1:
                continue
            if (conflict_mask[idx] & best_mask) == 0:
                best_mask |= (1 << idx)
                best_size += 1
                changed = True
    
    # Second polish pass: try adding elements that cause at most 1 conflict if net gain
    changed = True
    while changed and time.time() < DEADLINE - 0.5:
        changed = False
        for idx in range(N):
            if time.time() > DEADLINE - 0.5:
                break
            if (best_mask >> idx) & 1:
                continue
            conflicts = conflict_mask[idx] & best_mask
            num_conf = fast_popcount(conflicts)
            if num_conf <= 1:
                new_mask = (best_mask & ~conflicts) | (1 << idx)
                new_size = best_size - num_conf + 1
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
                    changed = True

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    
    return result
