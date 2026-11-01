import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    total = q ** n
    steps = [1] * n
    for i in range(1, n):
        steps[i] = steps[i - 1] * q
    
    def int_to_digits(x):
        return [x // steps[i] % q for i in range(n)]
        
    def digits_to_int(d):
        return sum(d[i] * steps[i] for i in range(n) if d[i] != 0)

    # Precompute balls as lists of integers
    balls = []
    for c in range(total):
        digits = int_to_digits(c)
        covered = [c]
        for i in range(n):
            orig = digits[i]
            for d in range(q):
                if d == orig:
                    continue
                digits[i] = d
                w = digits_to_int(digits)
                covered.append(w)
            digits[i] = orig
        balls.append(covered)
    
    # Precompute ball masks for fast union operations
    # Each ball is a bitmask of total bits (2187 bits)
    ball_masks = []
    for c in range(total):
        mask = 0
        for w in balls[c]:
            mask |= (1 << w)
        ball_masks.append(mask)
    
    full_mask = (1 << total) - 1
    
    # Greedy initialization
    def greedy_init():
        uncovered = full_mask
        code = []
        
        while uncovered:
            if time.time() > deadline - 15:
                break
            best_c = -1
            best_gain = 0
            
            # Find codeword that covers most uncovered positions
            for c in range(total):
                gain = bin(uncovered & ball_masks[c]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain == 0:
                break
            
            code.append(best_c)
            uncovered &= ~ball_masks[best_c]
        
        return code
    
    # Local search with removal-based optimization
    def local_search(initial_code, max_time):
        selected = list(initial_code)
        # Build coverage union
        union = 0
        for c in selected:
            union |= ball_masks[c]
        
        # Check if valid
        if union != full_mask:
            # Need to repair - add codewords greedily until valid
            while union != full_mask:
                if time.time() > max_time:
                    break
                # Find best codeword to add
                uncovered = ~union & full_mask
                best_c = -1
                best_gain = 0
                for c in range(total):
                    if c not in selected:
                        gain = bin(uncovered & ball_masks[c]).count('1')
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                if best_c == -1:
                    break
                selected.append(best_c)
                union |= ball_masks[best_c]
        
        best_size = len(selected)
        best_code = selected[:]
        t_start = time.time()
        
        # Iterative removal optimization
        while time.time() < max_time:
            changed = False
            # Try to remove each codeword
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > max_time:
                    break
                c_to_remove = selected[i]
                
                # Check if remaining codewords still cover everything
                union_others = 0
                for j in range(len(selected)):
                    if j != i:
                        union_others |= ball_masks[selected[j]]
                
                if union_others == full_mask:
                    # Safe to remove
                    selected.pop(i)
                    union = union_others
                    changed = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_code = selected[:]
                    break  # Restart from beginning since list changed
            
            if not changed:
                # No more removals possible - try random swaps
                if len(selected) > 2:
                    for _ in range(100):
                        if time.time() > max_time:
                            break
                        i = random.randint(0, len(selected) - 1)
                        j = random.randint(0, len(selected) - 1)
                        if i == j:
                            continue
                        
                        c_i = selected[i]
                        c_j = selected[j]
                        
                        # Try swapping: remove c_i, add c_j if not already present
                        # Actually, let's try: remove one, add a random new one
                        c_remove = selected[i]
                        c_add = random.randint(0, total - 1)
                        
                        if c_add in selected:
                            continue
                        
                        # Check if removing c_remove and adding c_add maintains coverage
                        union_without = 0
                        for c in selected:
                            if c != c_remove:
                                union_without |= ball_masks[c]
                        
                        union_new = union_without | ball_masks[c_add]
                        
                        if union_new == full_mask:
                            # Accept swap
                            selected[i] = c_add
                            union = union_new
                            if len(selected) < best_size:
                                best_size = len(selected)
                                best_code = selected[:]
            
            # Also try: remove one codeword, then add a better one
            if len(selected) > 1 and time.time() < max_time:
                for i in range(len(selected)):
                    if time.time() > max_time:
                        break
                    c_remove = selected[i]
                    
                    # Union without this codeword
                    union_without = 0
                    for j in range(len(selected)):
                        if j != i:
                            union_without |= ball_masks[selected[j]]
                    
                    uncovered = ~union_without & full_mask
                    if uncovered == 0:
                        # Can just remove
                        selected.pop(i)
                        union = union_without
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_code = selected[:]
                        break
                    else:
                        # Need to find best replacement
                        best_c = -1
                        best_gain = 0
                        for c in range(total):
                            if c in selected:
                                continue
                            gain = bin(uncovered & ball_masks[c]).count('1')
                            if gain > best_gain:
                                best_gain = gain
                                best_c = c
                        
                        if best_c != -1 and best_gain == bin(uncovered).count('1'):
                            # Perfect replacement
                            selected[i] = best_c
                            union = union_without | ball_masks[best_c]
                            if len(selected) < best_size:
                                best_size = len(selected)
                                best_code = selected[:]
                            break
        
        return best_code, best_size
    
    # Multi-start
    best_overall_code = []
    best_overall_size = float('inf')
    
    # Start 1: Greedy
    if time.time() < deadline - 20:
        init_code = greedy_init()
        if init_code:
            code1, size1 = local_search(init_code, deadline - 10)
            if size1 < best_overall_size:
                best_overall_size = size1
                best_overall_code = code1
    
    # Start 2: Perturbed greedy
    if time.time() < deadline - 15 and init_code:
        perturbed = list(init_code)
        random.shuffle(perturbed)
        # Remove 15% randomly
        remove_count = max(1, len(perturbed) // 7)
        for _ in range(remove_count):
            if perturbed:
                perturbed.pop()
        
        code2, size2 = local_search(perturbed, deadline - 10)
        if size2 < best_overall_size:
            best_overall_size = size2
            best_overall_code = code2
    
    # Start 3: Random restart
    if time.time() < deadline - 15:
        random_code = []
        # Add random codewords until roughly valid size
        for _ in range(150):
            c = random.randint(0, total - 1)
            if c not in random_code:
                random_code.append(c)
        
        code3, size3 = local_search(random_code, deadline - 10)
        if size3 < best_overall_size:
            best_overall_size = size3
            best_overall_code = code3
    
    # Fallback
    if not best_overall_code:
        best_overall_code = greedy_init()
    
    # Convert to list of lists of digits
    final_code = []
    seen = set()
    for c in best_overall_code:
        if c not in seen:
            seen.add(c)
            final_code.append(int_to_digits(c))
    
    return final_code
