# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    N = 256
    nodes = list(range(2, 257))
    num_nodes = len(nodes)
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_val = nodes

    # Precompute conflict masks
    conflict_mask = [0] * num_nodes
    
    for i in range(num_nodes):
        vi = idx_to_val[i]
        mask = 0
        for j in range(num_nodes):
            if i == j:
                continue
            vj = idx_to_val[j]
            if math.lcm(vi, vj) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def get_score_frac(mask):
        score = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score += Fraction(1, idx_to_val[idx])
            m ^= lsb
        return score

    def local_search_simple(initial_mask, time_limit=3.0):
        """Local search with add, 1-for-1 swaps, and 2-for-1 swaps"""
        current_mask = initial_mask
        current_score = get_score_frac(current_mask)
        improved = True
        start_ls_time = time.time()
        
        while improved:
            improved = False
            
            # Check time
            if time.time() > deadline or time.time() - start_ls_time > time_limit:
                break
            
            # Add phase: try to add any v not in mask (small to large for higher weight)
            for v in range(2, 257):
                idx = val_to_idx[v]
                if not (current_mask & (1 << idx)):
                    if (conflict_mask[idx] & current_mask) == 0:
                        current_mask |= (1 << idx)
                        current_score += Fraction(1, v)
                        improved = True
            
            # 1-for-1 swap: remove u (larger), add v (smaller) where v < u
            in_mask = []
            not_in_mask = []
            for i in range(num_nodes):
                if current_mask & (1 << i):
                    in_mask.append(i)
                else:
                    not_in_mask.append(i)
            
            in_mask_desc = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask_asc = sorted(not_in_mask, key=lambda i: idx_to_val[i])
            
            found_swap = False
            for u_idx in in_mask_desc:
                if time.time() > deadline:
                    return current_mask, current_score
                u_val = idx_to_val[u_idx]
                mask_without_u = current_mask & ~(1 << u_idx)
                
                for v_idx in not_in_mask_asc:
                    v_val = idx_to_val[v_idx]
                    if v_val >= u_val:
                        break
                    if (conflict_mask[v_idx] & mask_without_u) == 0:
                        current_mask = (current_mask & ~(1 << u_idx)) | (1 << v_idx)
                        current_score += Fraction(1, v_val) - Fraction(1, u_val)
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            # 2-for-1 swap: remove two larger elements, add one smaller element
            # This can unlock configurations where 1-for-1 swaps are stuck
            if not found_swap and len(in_mask) >= 2:
                in_mask_desc2 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
                not_in_mask_asc2 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
                
                found_2swap = False
                for i in range(len(in_mask_desc2)):
                    if time.time() > deadline:
                        return current_mask, current_score
                    u1_idx = in_mask_desc2[i]
                    u1_val = idx_to_val[u1_idx]
                    for j in range(i + 1, len(in_mask_desc2)):
                        u2_idx = in_mask_desc2[j]
                        u2_val = idx_to_val[u2_idx]
                        # Only consider removing two elements if their combined reciprocal
                        # is less than some candidate we might add
                        if Fraction(1, u1_val) + Fraction(1, u2_val) < Fraction(1, 256):
                            continue  # Not worth trying to replace with anything >= 256
                        
                        mask_without_u1_u2 = current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)
                        
                        for v_idx in not_in_mask_asc2:
                            v_val = idx_to_val[v_idx]
                            if v_val >= min(u1_val, u2_val):
                                break
                            # Check v doesn't conflict with remaining mask
                            if (conflict_mask[v_idx] & mask_without_u1_u2) == 0:
                                gain = Fraction(1, v_val) - Fraction(1, u1_val) - Fraction(1, u2_val)
                                if gain > 0:
                                    current_mask = mask_without_u1_u2 | (1 << v_idx)
                                    current_score += gain
                                    improved = True
                                    found_2swap = True
                                    break
                        
                        if found_2swap:
                            break
                    if found_2swap:
                        break
            
        return current_mask, current_score

    import random
    rng = random.Random(42)

    best_mask = 0
    best_score = Fraction(0)

    # Generate diverse initial solutions
    # 1. Greedy by weight (small to large)
    mask1 = 0
    for v in range(2, 257):
        idx = val_to_idx[v]
        if (conflict_mask[idx] & mask1) == 0:
            mask1 |= (1 << idx)
    
    # 2. Large numbers set {129..256}
    mask2 = 0
    for v in range(129, 257):
        idx = val_to_idx[v]
        mask2 |= (1 << idx)
        
    # 3. Randomized Greedy
    def randomized_greedy():
        m = 0
        available = list(range(2, 257))
        while available:
            compatible = [v for v in available if (conflict_mask[val_to_idx[v]] & m) == 0]
            if not compatible:
                break
            compatible.sort()
            k = min(3, len(compatible))
            v = rng.choice(compatible[:k])
            idx = val_to_idx[v]
            m |= (1 << idx)
            available.remove(v)
        return m

    mask3 = randomized_greedy()
    
    # Run local search on initial masks
    for init_mask in [mask1, mask2, mask3]:
        if time.time() > deadline - 10.0:
            break
        m_ls, s_ls = local_search_simple(init_mask, time_limit=3.0)
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls
    
    # Multiple restarts with perturbation
    def perturb(mask, rng):
        if mask == 0:
            return mask
        bits = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bits.append(idx)
            m ^= lsb
        
        if not bits:
            return mask
            
        num_remove = rng.choice([1, 1, 2])
        bits_sorted = sorted(bits, key=lambda i: idx_to_val[i], reverse=True)
        half = max(1, len(bits_sorted) // 2)
        candidates = bits_sorted[half:]
        if len(candidates) < num_remove:
            candidates = bits_sorted
        indices_to_remove = rng.sample(candidates, min(num_remove, len(candidates)))
        
        new_mask = mask
        for idx in indices_to_remove:
            new_mask &= ~(1 << idx)
        return new_mask

    # Try perturbations from best solution
    for _ in range(15):
        if time.time() > deadline - 5.0:
            break
        
        m_pert = perturb(best_mask, rng)
        remaining = deadline - time.time() - 1.0
        if remaining < 1.0:
            break
        m_pol, s_pol = local_search_simple(m_pert, time_limit=min(remaining, 2.0))
        if s_pol > best_score:
            best_score = s_pol
            best_mask = m_pol

    # Final high-precision polish: exhaustive single-add and single-swap check
    if time.time() < deadline - 1.0:
        m_final = best_mask
        s_final = best_score
        
        # Try every possible single addition
        for v in range(2, 257):
            idx = val_to_idx[v]
            if not (m_final & (1 << idx)):
                if (conflict_mask[idx] & m_final) == 0:
                    new_mask = m_final | (1 << idx)
                    new_score = s_final + Fraction(1, v)
                    if new_score > s_final:
                        m_final = new_mask
                        s_final = new_score
        
        # Try every possible 1-for-1 swap (full pass, not greedy)
        in_mask_final = [i for i in range(num_nodes) if m_final & (1 << i)]
        not_in_mask_final = [i for i in range(num_nodes) if not (m_final & (1 << i))]
        
        best_swap_gain = Fraction(0)
        best_swap_u = None
        best_swap_v = None
        
        for u_idx in in_mask_final:
            u_val = idx_to_val[u_idx]
            mask_without_u = m_final & ~(1 << u_idx)
            for v_idx in not_in_mask_final:
                v_val = idx_to_val[v_idx]
                if (conflict_mask[v_idx] & mask_without_u) == 0:
                    gain = Fraction(1, v_val) - Fraction(1, u_val)
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_u = u_idx
                        best_swap_v = v_idx
        
        if best_swap_u is not None:
            m_final = (m_final & ~(1 << best_swap_u)) | (1 << best_swap_v)
            s_final += best_swap_gain
        
        # Repeat add+swap until no improvement
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            # Add phase
            any_add = False
            for v in range(2, 257):
                idx = val_to_idx[v]
                if not (m_final & (1 << idx)):
                    if (conflict_mask[idx] & m_final) == 0:
                        m_final |= (1 << idx)
                        s_final += Fraction(1, v)
                        any_add = True
            if not any_add:
                break
            
            # Swap phase
            in_mask_final = [i for i in range(num_nodes) if m_final & (1 << i)]
            not_in_mask_final = [i for i in range(num_nodes) if not (m_final & (1 << i))]
            
            best_swap_gain = Fraction(0)
            best_swap_u = None
            best_swap_v = None
            
            for u_idx in in_mask_final:
                u_val = idx_to_val[u_idx]
                mask_without_u = m_final & ~(1 << u_idx)
                for v_idx in not_in_mask_final:
                    v_val = idx_to_val[v_idx]
                    if (conflict_mask[v_idx] & mask_without_u) == 0:
                        gain = Fraction(1, v_val) - Fraction(1, u_val)
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_u = u_idx
                            best_swap_v = v_idx
            
            if best_swap_u is None:
                break
            m_final = (m_final & ~(1 << best_swap_u)) | (1 << best_swap_v)
            s_final += best_swap_gain
        
        if s_final > best_score:
            best_score = s_final
            best_mask = m_final

    # Convert mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx_to_val[idx])
        m ^= lsb
        
    return result
# EVOLVE-BLOCK-END
