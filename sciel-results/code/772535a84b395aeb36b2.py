import time
import random

def construct():
    N = 128
    # Precompute APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # Precompute which APs involve each index
    ap_indices_for_pos = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices_for_pos[pos].append(idx)
            
    # Helper to check if an AP is violated given a list of 4 colors
    def is_violated(c0, c1, c2, c3):
        # Check if distinct count < 3
        # Fast path for common cases
        if c0 == c1:
            if c2 == c3:
                # 2 pairs or 4 same
                if c0 == c2:
                    return True # 4 same
                else:
                    return True # 2 pairs (A A B B)
            else:
                if c2 == c1:
                    return True # 3 same (A A A B)
                if c3 == c1:
                    return True # 3 same (A A B A)
                # c0==c1, c2!=c1, c3!=c1, c2!=c3
                # Colors: A A C D. Distinct: A, C, D. Count 3.
                # Wait, if c2==c3, it's 2 pairs.
                # If c2!=c3, and c2!=c1, c3!=c1.
                # Set is {A, C, D}. Size 3. Not violated.
                return False
        else:
            # c0 != c1
            if c2 == c0:
                if c3 == c1:
                    return True # A B A B (2 pairs)
                if c3 == c0:
                    return True # A B A A (3 same)
                # A B A C. Distinct A, B, C. Size 3.
                return False
            if c2 == c1:
                if c3 == c0:
                    return True # A B B A (2 pairs)
                if c3 == c1:
                    return True # A B B B (3 same)
                # A B B C. Distinct A, B, C. Size 3.
                return False
            if c2 == c3:
                # c0 != c1, c2 == c3.
                # A B C C. Distinct A, B, C. Size 3.
                # Unless A==C or B==C?
                # We know c0 != c1.
                # If c0 == c2, handled above? No, c2==c3.
                # If c0 == c2, then c0==c3.
                # Case c0==c2: A B A A. Handled in c2==c0 branch?
                # In c2==c0 branch: if c3==c0, return True.
                # So if we are here, c0 != c2 and c1 != c2.
                # So A, B, C are distinct.
                return False
            # c0 != c1, c0 != c2, c1 != c2, c2 != c3, c0 != c3?
            # If c0 == c3: A B C A. Distinct A, B, C. Size 3.
            # If c1 == c3: A B C B. Distinct A, B, C. Size 3.
            # If all distinct: A B C D. Size 4.
            return False

    # Simpler and likely fast enough:
    def is_violated_simple(c0, c1, c2, c3):
        return len({c0, c1, c2, c3}) < 3

    # Use simple version first, optimize if needed. 
    # Given N=128, M~2700, degree~80, k~4.
    # 10000 steps * 80 APs * 4 colors = 3.2M calls.
    # Python can do ~10M simple ops/sec. So ~0.3s per 10k steps.
    # We have 60s. So we can do ~2000k steps.

    def count_violations(colors):
        count = 0
        for ap in aps:
            if is_violated_simple(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                count += 1
        return count

    def get_violation_count_for_pos(colors, pos, new_val):
        v_count = 0
        for ap_idx in ap_indices_for_pos[pos]:
            ap = aps[ap_idx]
            c0 = colors[ap[0]] if ap[0] != pos else new_val
            c1 = colors[ap[1]] if ap[1] != pos else new_val
            c2 = colors[ap[2]] if ap[2] != pos else new_val
            c3 = colors[ap[3]] if ap[3] != pos else new_val
            if is_violated_simple(c0, c1, c2, c3):
                v_count += 1
        return v_count

    global_deadline = time.time() + 50.0 # Safety margin

    best_solution = None
    best_k = 128

    # Try increasing k
    for k in range(3, 15):
        if time.time() > global_deadline:
            break
            
        found = False
        # Restart loop
        for restart in range(500):
            if time.time() > global_deadline:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(N)]
            
            # Check initial state
            if count_violations(colors) == 0:
                best_solution = colors
                best_k = k
                found = True
                break
                
            # MinConflicts
            # We don't maintain global violation count, just local.
            # This is simpler.
            for step in range(5000):
                if time.time() > global_deadline:
                    break
                    
                # Pick a random position. 
                # To be more efficient, we could pick one with high local violation count.
                # But calculating that for all is expensive.
                # Let's stick to random for now, or random from a small set.
                pos = random.randint(0, N - 1)
                
                current_val = colors[pos]
                
                # Find best new val
                best_new_val = current_val
                min_v = get_violation_count_for_pos(colors, pos, current_val)
                
                for new_val in range(k):
                    if new_val == current_val:
                        continue
                    v = get_violation_count_for_pos(colors, pos, new_val)
                    if v < min_v:
                        min_v = v
                        best_new_val = new_val
                        
                # If no improvement, pick random
                if best_new_val == current_val:
                    # Try a few random ones to escape local minima
                    for _ in range(3):
                        rand_val = random.randint(0, k - 1)
                        if rand_val != current_val:
                            v = get_violation_count_for_pos(colors, pos, rand_val)
                            if v < min_v:
                                min_v = v
                                best_new_val = rand_val
                                break
                    if best_new_val == current_val:
                        best_new_val = random.randint(0, k - 1)
                        
                colors[pos] = best_new_val
                
                # Check if solved every 50 steps
                if step % 50 == 0:
                    if count_violations(colors) == 0:
                        best_solution = colors
                        best_k = k
                        found = True
                        break
            
            if found:
                break
                
        if found:
            break

    if best_solution is None:
        # Fallback
        best_solution = list(range(N))

    return best_solution
