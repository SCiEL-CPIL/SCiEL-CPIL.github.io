import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 256
    max_size = 64
    
    # Bitmask for differences: diff ranges from -(n-1) to (n-1), so offset by n-1
    # We need to track nonzero differences
    # diff in range [-(n-1), n-1], exclude 0
    # Map diff d to index d + (n-1), range [1, 2*(n-1)]
    DIFF_OFFSET = n - 1
    DIFF_SIZE = 2 * DIFF_OFFSET  # indices 1 to 2*DIFF_OFFSET-1 (0 unused for diff=0)
    
    def diff_to_bit(d):
        """Convert difference d (nonzero) to bit position"""
        idx = d + DIFF_OFFSET
        return 1 << (idx - 1)  # 0-indexed bit
    
    def can_add_to_A(x, A, sums_A, diffs_B_mask):
        if x + x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            d1 = x - a
            d2 = a - x
            if d1 != 0 and (diffs_B_mask >> (d1 + DIFF_OFFSET - 1)) & 1:
                return False
            if d2 != 0 and (diffs_B_mask >> (d2 + DIFF_OFFSET - 1)) & 1:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A_mask):
        if x + x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            d1 = x - b
            d2 = b - x
            if d1 != 0 and (diffs_A_mask >> (d1 + DIFF_OFFSET - 1)) & 1:
                return False
            if d2 != 0 and (diffs_A_mask >> (d2 + DIFF_OFFSET - 1)) & 1:
                return False
        return True

    def greedy_construct(order, deadline):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A_mask = 0
        diffs_B_mask = 0
        
        for x in order:
            if time.time() > deadline:
                break
            if len(A) >= max_size and len(B) >= max_size:
                break
                
            # Try A
            if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B_mask):
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    d1 = x - a
                    d2 = a - x
                    if d1 != 0:
                        diffs_A_mask |= (1 << (d1 + DIFF_OFFSET - 1))
                    if d2 != 0:
                        diffs_A_mask |= (1 << (d2 + DIFF_OFFSET - 1))
                sums_A.add(x + x)
            # Try B
            elif len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A_mask):
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    d1 = x - b
                    d2 = b - x
                    if d1 != 0:
                        diffs_B_mask |= (1 << (d1 + DIFF_OFFSET - 1))
                    if d2 != 0:
                        diffs_B_mask |= (1 << (d2 + DIFF_OFFSET - 1))
                sums_B.add(x + x)
                
        return A, B, sums_A, sums_B, diffs_A_mask, diffs_B_mask

    def local_search(A, B, sums_A, sums_B, diffs_A_mask, diffs_B_mask, deadline):
        """Try to improve by adding elements or doing swaps"""
        in_A = set(A)
        in_B = set(B)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any unused element
            unused = [x for x in range(1, n + 1) if x not in in_A and x not in in_B]
            random.shuffle(unused)
            
            for x in unused:
                if time.time() > deadline:
                    break
                    
                # Try add to A
                if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B_mask):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        d1 = x - a
                        d2 = a - x
                        if d1 != 0:
                            diffs_A_mask |= (1 << (d1 + DIFF_OFFSET - 1))
                        if d2 != 0:
                            diffs_A_mask |= (1 << (d2 + DIFF_OFFSET - 1))
                    sums_A.add(x + x)
                    in_A.add(x)
                    improved = True
                    break
                    
                # Try add to B
                if len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A_mask):
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        d1 = x - b
                        d2 = b - x
                        if d1 != 0:
                            diffs_B_mask |= (1 << (d1 + DIFF_OFFSET - 1))
                        if d2 != 0:
                            diffs_B_mask |= (1 << (d2 + DIFF_OFFSET - 1))
                    sums_B.add(x + x)
                    in_B.add(x)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try swaps: remove one element, try to add two new ones
            # Try removing from A and adding to A or B
            if len(A) > 0:
                a_idx = random.randint(0, len(A) - 1)
                a_val = A[a_idx]
                
                # Remove a_val from A
                A_temp = A[:a_idx] + A[a_idx+1:]
                
                # Recompute sums and diffs for A_temp
                sums_A_temp = set()
                diffs_A_mask_temp = 0
                for i in range(len(A_temp)):
                    for j in range(i, len(A_temp)):
                        sums_A_temp.add(A_temp[i] + A_temp[j])
                for i in range(len(A_temp)):
                    for j in range(len(A_temp)):
                        if i != j:
                            d = A_temp[j] - A_temp[i]
                            if d != 0:
                                diffs_A_mask_temp |= (1 << (d + DIFF_OFFSET - 1))
                
                # Try to add a new element to A_temp
                for x in range(1, n + 1):
                    if x in in_A or x in in_B or x == a_val:
                        continue
                    if len(A_temp) < max_size and can_add_to_A(x, A_temp, sums_A_temp, diffs_B_mask):
                        # Success: replace a_val with x in A
                        A = A_temp + [x]
                        in_A.remove(a_val)
                        in_A.add(x)
                        
                        # Update sums and diffs
                        sums_A = sums_A_temp
                        for a in A[:-1]:
                            sums_A.add(x + a)
                            d1 = x - a
                            d2 = a - x
                            if d1 != 0:
                                diffs_A_mask_temp |= (1 << (d1 + DIFF_OFFSET - 1))
                            if d2 != 0:
                                diffs_A_mask_temp |= (1 << (d2 + DIFF_OFFSET - 1))
                        sums_A.add(x + x)
                        diffs_A_mask = diffs_A_mask_temp
                        improved = True
                        break
                    
                    # Try to add x to B instead
                    if len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A_mask_temp):
                        # Move a_val from A to B, add x to A
                        B.append(x)
                        for b in B[:-1]:
                            sums_B.add(x + b)
                            d1 = x - b
                            d2 = b - x
                            if d1 != 0:
                                diffs_B_mask |= (1 << (d1 + DIFF_OFFSET - 1))
                            if d2 != 0:
                                diffs_B_mask |= (1 << (d2 + DIFF_OFFSET - 1))
                        sums_B.add(x + x)
                        in_B.add(x)
                        
                        A = A_temp
                        in_A.remove(a_val)
                        in_A.add(x)
                        
                        sums_A = sums_A_temp
                        for a in A[:-1]:
                            sums_A.add(x + a)
                            d1 = x - a
                            d2 = a - x
                            if d1 != 0:
                                diffs_A_mask_temp |= (1 << (d1 + DIFF_OFFSET - 1))
                            if d2 != 0:
                                diffs_A_mask_temp |= (1 << (d2 + DIFF_OFFSET - 1))
                        sums_A.add(x + x)
                        diffs_A_mask = diffs_A_mask_temp
                        improved = True
                        break
            
            # Try removing from B and adding to B or A
            if not improved and len(B) > 0:
                b_idx = random.randint(0, len(B) - 1)
                b_val = B[b_idx]
                
                # Remove b_val from B
                B_temp = B[:b_idx] + B[b_idx+1:]
                
                # Recompute sums and diffs for B_temp
                sums_B_temp = set()
                diffs_B_mask_temp = 0
                for i in range(len(B_temp)):
                    for j in range(i, len(B_temp)):
                        sums_B_temp.add(B_temp[i] + B_temp[j])
                for i in range(len(B_temp)):
                    for j in range(len(B_temp)):
                        if i != j:
                            d = B_temp[j] - B_temp[i]
                            if d != 0:
                                diffs_B_mask_temp |= (1 << (d + DIFF_OFFSET - 1))
                
                # Try to add a new element to B_temp
                for x in range(1, n + 1):
                    if x in in_A or x in in_B or x == b_val:
                        continue
                    if len(B_temp) < max_size and can_add_to_B(x, B_temp, sums_B_temp, diffs_A_mask):
                        # Success: replace b_val with x in B
                        B = B_temp + [x]
                        in_B.remove(b_val)
                        in_B.add(x)
                        
                        # Update sums and diffs
                        sums_B = sums_B_temp
                        for b in B[:-1]:
                            sums_B.add(x + b)
                            d1 = x - b
                            d2 = b - x
                            if d1 != 0:
                                diffs_B_mask_temp |= (1 << (d1 + DIFF_OFFSET - 1))
                            if d2 != 0:
                                diffs_B_mask_temp |= (1 << (d2 + DIFF_OFFSET - 1))
                        sums_B.add(x + x)
                        diffs_B_mask = diffs_B_mask_temp
                        improved = True
                        break
                        
                    # Try to add x to A instead
                    if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B_mask_temp):
                        # Move b_val from B to A, add x to B
                        A.append(x)
                        for a in A[:-1]:
                            sums_A.add(x + a)
                            d1 = x - a
                            d2 = a - x
                            if d1 != 0:
                                diffs_A_mask |= (1 << (d1 + DIFF_OFFSET - 1))
                            if d2 != 0:
                                diffs_A_mask |= (1 << (d2 + DIFF_OFFSET - 1))
                        sums_A.add(x + x)
                        in_A.add(x)
                        
                        B = B_temp
                        in_B.remove(b_val)
                        in_B.add(x)
                        
                        sums_B = sums_B_temp
                        for b in B[:-1]:
                            sums_B.add(x + b)
                            d1 = x - b
                            d2 = b - x
                            if d1 != 0:
                                diffs_B_mask_temp |= (1 << (d1 + DIFF_OFFSET - 1))
                            if d2 != 0:
                                diffs_B_mask_temp |= (1 << (d2 + DIFF_OFFSET - 1))
                        sums_B.add(x + x)
                        diffs_B_mask = diffs_B_mask_temp
                        improved = True
                        break
        
        return A, B, sums_A, sums_B, diffs_A_mask, diffs_B_mask

    best_A, best_B = [], []
    best_score = 0
    
    num_trials = 500  # More trials with faster checks
    
    for trial in range(num_trials):
        if time.time() > deadline - 2:
            break
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A, B, sums_A, sums_B, diffs_A_mask, diffs_B_mask = greedy_construct(order, deadline)
        
        # Polish with local search
        A, B, sums_A, sums_B, diffs_A_mask, diffs_B_mask = local_search(
            A, B, sums_A, sums_B, diffs_A_mask, diffs_B_mask, deadline
        )
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]
            
        # Early exit if we hit a good score
        if best_score >= 32:
            break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
