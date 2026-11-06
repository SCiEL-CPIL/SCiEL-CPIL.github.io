import time
import math
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit to be safe

    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    divisors.sort()
    
    # 2. Precompute masks for each (m, r)
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask_val = 0
            for x in range(r, base, m):
                mask_val |= (1 << x)
            masks[m][r] = mask_val
            
    # Initial state
    uncovered_mask = (1 << base) - 1
    used_moduli = set()
    classes = []
    freq = [0] * base
    unique_mask = 0  # Bitset of residues with freq == 1

    def add_class(m, r):
        nonlocal uncovered_mask, unique_mask
        for x in range(r, base, m):
            if freq[x] == 0:
                uncovered_mask &= ~(1 << x)
                unique_mask |= (1 << x)
            elif freq[x] == 1:
                unique_mask &= ~(1 << x)
            freq[x] += 1
        used_moduli.add(m)
        classes.append([m, r])

    def remove_class(m, r):
        nonlocal uncovered_mask, unique_mask
        for x in range(r, base, m):
            freq[x] -= 1
            if freq[x] == 1:
                unique_mask |= (1 << x)
            elif freq[x] == 0:
                unique_mask &= ~(1 << x)
                uncovered_mask |= (1 << x)
        used_moduli.remove(m)
        for i, (cm, cr) in enumerate(classes):
            if cm == m and cr == r:
                classes.pop(i)
                break

    # 3. Greedy Set Cover
    remaining_moduli = set(divisors)
    
    while remaining_moduli:
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in remaining_moduli:
            for r in range(m):
                gain = (uncovered_mask & masks[m][r]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_gain == 0:
            break
            
        add_class(best_m, best_r)
        remaining_moduli.remove(best_m)
        
        if time.time() > deadline:
            break

    best_score = uncovered_mask.bit_count()
    
    # 4. Simulated Annealing
    # Temperature starts high, decays
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    current_score = best_score
    
    # Precompute list of unused moduli for random selection
    unused_list = list(remaining_moduli)
    
    while T > min_T and time.time() < deadline:
        # Random move: Remove one used, add one unused
        if not classes or not unused_list:
            break
            
        idx_out = random.randint(0, len(classes) - 1)
        m_out, r_out = classes[idx_out]
        
        idx_in = random.randint(0, len(unused_list) - 1)
        m_in = unused_list[idx_in]
        r_in = random.randint(0, m_in - 1)
        
        # Apply move
        remove_class(m_out, r_out)
        add_class(m_in, r_in)
        
        unused_list[idx_in] = m_out
        # Note: classes list is modified by remove/add, indices might shift.
        # But we just need a valid state.
        
        new_score = uncovered_mask.bit_count()
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                # Save best state? We need to copy classes.
                # Copying is expensive. Let's just keep the current state as best if it's better.
                # But we need to return the best.
                # We can store the best classes in a separate variable.
                best_classes = [c[:] for c in classes]
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() > prob:
                    # Reject: Undo move
                    remove_class(m_in, r_in)
                    add_class(m_out, r_out)
                    unused_list[idx_in] = m_in
                    # Restore state
                else:
                    # Accept worse move
                    current_score = new_score
            else:
                # Accept better move (delta <= 0)
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
                    
        T *= cooling
        
    # Return best found
    # If best_classes was never set (greedy was best), use classes from greedy?
    # Wait, if we never found a better score in SA, `best_classes` might be undefined or stale.
    # Let's initialize best_classes with the greedy result.
    
    return best_classes if 'best_classes' in locals() else classes
