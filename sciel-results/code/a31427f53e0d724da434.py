"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s kill
    
    N = 256
    # Precompute reciprocal weights
    recip_float = [0.0] * (N + 1)
    recip_frac = [Fraction(0, 1)] * (N + 1)
    for i in range(1, N + 1):
        recip_float[i] = 1.0 / i
        recip_frac[i] = Fraction(1, i)
        
    # Precompute for each x, the set of pairs (a, b) such that {a, x, b} forms a 3-AP
    # For validity checking, we need: if we add x, no pair (a, b) in S can form AP with x
    # AP conditions for adding x to S:
    # 1. x is middle: a, x, b => a + b = 2x, so b = 2x - a
    # 2. a is middle: x, a, b => x + b = 2a, so b = 2a - x  
    # 3. b is middle: a, b, x => a + x = 2b, so a = 2b - x
    # 
    # For fast checking, precompute for each x:
    # - conflicts_middle[x] = set of (a, b) where a + b = 2x (x is middle)
    # - For each a in S, we can check if 2x-a is also in S (covers case 1)
    # - For each a in S, we can check if 2a-x is also in S (covers cases 2&3)
    
    # Actually, the original is_valid_addition is O(|S|) which is fine for |S| ~ 50-100
    # But we can make it faster with bit operations
    
    def bit_set(mask, i):
        return (mask >> (i - 1)) & 1
    
    def set_bit(mask, i):
        return mask | (1 << (i - 1))
        
    def clear_bit(mask, i):
        return mask & ~(1 << (i - 1))

    def is_valid_addition_fast(mask, x):
        """
        Fast check if x can be added to set represented by mask.
        Uses the fact that for a 3-AP containing x and two elements from S,
        we can check all three cases efficiently.
        """
        if bit_set(mask, x):
            return False
            
        # Case 1: x is the middle. Check if there exist a, b in S with a + b = 2*x
        # For each a in S, check if b = 2*x - a is also in S
        m = mask
        while m:
            lsb = m & -m
            a = lsb.bit_length()
            b = 2 * x - a
            if 1 <= b <= N and b != x and b != a and bit_set(mask, b):
                return False
            m &= m - 1
            
        # Case 2: Some a in S is the middle. Check if there exist a, b in S with x + b = 2*a
        # For each a in S, check if b = 2*a - x is also in S
        m = mask
        while m:
            lsb = m & -m
            a = lsb.bit_length()
            b = 2 * a - x
            if 1 <= b <= N and b != x and b != a and bit_set(mask, b):
                return False
            m &= m - 1
            
        return True

    def get_weight(mask):
        total = 0.0
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length()
            total += recip_float[x]
            m &= m - 1
        return total

    # 1. Multiple greedy initializations with different strategies
    best_mask = 0
    best_weight = 0.0
    
    # Strategy 1: Pure greedy by weight (smallest first)
    order1 = list(range(1, N + 1))
    order1.sort(key=lambda x: -recip_float[x])
    mask1 = 0
    for x in order1:
        if is_valid_addition_fast(mask1, x):
            mask1 = set_bit(mask1, x)
    w1 = get_weight(mask1)
    if w1 > best_weight:
        best_weight = w1
        best_mask = mask1
    
    # Strategy 2: Greedy with random perturbation
    for _ in range(5):
        if time.time() > DEADLINE - 50:
            break
        perturbed = [recip_float[i] * (1 + random.uniform(-0.2, 0.2)) for i in range(N + 1)]
        order2 = list(range(1, N + 1))
        order2.sort(key=lambda x: -perturbed[x])
        mask2 = 0
        for x in order2:
            if is_valid_addition_fast(mask2, x):
                mask2 = set_bit(mask2, x)
        w2 = get_weight(mask2)
        if w2 > best_weight:
            best_weight = w2
            best_mask = mask2
    
    # Strategy 3: Greedy that prioritizes small numbers but allows some flexibility
    # Try different thresholds for when to skip a number
    for threshold in [0.0, 0.1, 0.2, 0.3]:
        if time.time() > DEADLINE - 45:
            break
        order3 = list(range(1, N + 1))
        # Custom sort: primarily by weight, but with some randomness for close values
        def custom_key(x):
            base = -recip_float[x]
            noise = random.uniform(0, threshold) if random.random() < 0.3 else 0
            return base + noise
        order3.sort(key=custom_key)
        mask3 = 0
        for x in order3:
            if is_valid_addition_fast(mask3, x):
                mask3 = set_bit(mask3, x)
        w3 = get_weight(mask3)
        if w3 > best_weight:
            best_weight = w3
            best_mask = mask3

    # 2. Enhanced local search
    current_mask = best_mask
    current_weight = best_weight
    
    # Precompute list of elements for faster iteration
    def get_elements(mask):
        elems = []
        m = mask
        while m:
            lsb = m & -m
            elems.append(lsb.bit_length())
            m &= m - 1
        return elems
    
    max_iterations = 50000
    no_improve_count = 0
    max_no_improve = 200
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        improved = False
        
        # A. Try Additions (most important, cheapest)
        for x in range(1, N + 1):
            if bit_set(current_mask, x):
                continue
            if is_valid_addition_fast(current_mask, x):
                new_mask = set_bit(current_mask, x)
                new_w = current_weight + recip_float[x]
                if new_w > current_weight + 1e-12:
                    current_mask = new_mask
                    current_weight = new_w
                    if current_weight > best_weight:
                        best_weight = current_weight
                        best_mask = current_mask
                    improved = True
                    break 
        
        if improved:
            no_improve_count = 0
            continue
            
        no_improve_count += 1
        
        # B. Try 1-for-1 Swaps
        elements_in_set = get_elements(current_mask)
        found_swap = False
        
        # Sort elements by weight descending (heaviest first) to try removing them
        elements_sorted = sorted(elements_in_set, key=lambda x: -recip_float[x])
        
        for u in elements_sorted:
            # Try replacing u with v where v < u (to gain weight)
            for v in range(1, u):
                if bit_set(current_mask, v):
                    continue
                
                temp_mask = clear_bit(current_mask, u)
                if is_valid_addition_fast(temp_mask, v):
                    new_mask = set_bit(temp_mask, v)
                    new_w = current_weight - recip_float[u] + recip_float[v]
                    if new_w > current_weight + 1e-12:
                        current_mask = new_mask
                        current_weight = new_w
                        if current_weight > best_weight:
                            best_weight = current_weight
                            best_mask = current_mask
                        found_swap = True
                        break
            if found_swap:
                break
        
        if found_swap:
            no_improve_count = 0
            continue
        
        # C. Try 2-for-1 Swaps (expensive, only if time permits)
        if time.time() < DEADLINE - 10:
            elements_in_set = get_elements(current_mask)
            found_2_1 = False
            
            # Only try a limited number of pairs to keep it fast
            max_pairs_to_try = 50
            pairs_tried = 0
            
            for i in range(len(elements_in_set)):
                if pairs_tried >= max_pairs_to_try or time.time() > DEADLINE - 5:
                    break
                u1 = elements_in_set[i]
                for j in range(i + 1, len(elements_in_set)):
                    if pairs_tried >= max_pairs_to_try or time.time() > DEADLINE - 5:
                        break
                    u2 = elements_in_set[j]
                    pairs_tried += 1
                    
                    temp_mask = clear_bit(clear_bit(current_mask, u1), u2)
                    weight_loss = recip_float[u1] + recip_float[u2]
                    
                    # Look for best v
                    best_v = -1
                    best_v_w = 0.0
                    for v in range(1, N + 1):
                        if bit_set(current_mask, v):
                            continue
                        if recip_float[v] <= weight_loss:
                            continue
                        if is_valid_addition_fast(temp_mask, v):
                            if recip_float[v] > best_v_w:
                                best_v_w = recip_float[v]
                                best_v = v
                                
                    if best_v != -1:
                        new_mask = set_bit(temp_mask, best_v)
                        new_w = current_weight - weight_loss + best_v_w
                        if new_w > current_weight + 1e-12:
                            current_mask = new_mask
                            current_weight = new_w
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_mask = current_mask
                            found_2_1 = True
                            break
                if found_2_1:
                    break
                    
            if found_2_1:
                no_improve_count = 0
                continue
        
        # D. If stuck, try a random restart with perturbation
        if no_improve_count > max_no_improve and time.time() < DEADLINE - 15:
            # Perturb weights more aggressively
            perturbation = random.uniform(0.1, 0.5)
            perturbed_weights = [recip_float[i] * (1 + random.uniform(-perturbation, perturbation)) for i in range(N + 1)]
            order_restart = list(range(1, N + 1))
            order_restart.sort(key=lambda x: -perturbed_weights[x])
            
            mask_restart = 0
            for x in order_restart:
                if is_valid_addition_fast(mask_restart, x):
                    mask_restart = set_bit(mask_restart, x)
            w_restart = get_weight(mask_restart)
            
            if w_restart > best_weight:
                best_weight = w_restart
                best_mask = mask_restart
                current_mask = mask_restart
                current_weight = w_restart
                no_improve_count = 0
            else:
                # If restart didn't help, break to avoid wasting time
                break
        elif no_improve_count > max_no_improve * 2:
            # Too many iterations without improvement, break
            break

    # Final polish: try one more round of additions and simple swaps
    if time.time() < DEADLINE - 2:
        for _ in range(100):
            if time.time() > DEADLINE - 1:
                break
            improved = False
            for x in range(1, N + 1):
                if bit_set(best_mask, x):
                    continue
                if is_valid_addition_fast(best_mask, x):
                    new_mask = set_bit(best_mask, x)
                    new_w = best_weight + recip_float[x]
                    if new_w > best_weight + 1e-12:
                        best_mask = new_mask
                        best_weight = new_w
                        improved = True
                        break
            if not improved:
                break

    # Final Result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        result.append(lsb.bit_length())
        m &= m - 1
        
    result.sort()
    
    return result

# EVOLVE-BLOCK-END
