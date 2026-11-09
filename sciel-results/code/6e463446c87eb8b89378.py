import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Initial set: Upper half [65..128]
    current_mask = 0
    for i in range(65, 129):
        current_mask |= (1 << (i - 1))
        
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    def get_size(mask):
        return bin(mask).count('1')
    
    def mask_to_set(mask):
        """Convert mask to a Python set of 1-indexed elements."""
        s = set()
        m = mask
        while m:
            lb = m & (-m)
            s.add(lb.bit_length())
            m ^= lb
        return s
    
    def is_valid_addition(mask, x, elem_set):
        """
        Check if adding element x (1-indexed) to the set is valid.
        elem_set is the set of 1-indexed elements in mask.
        Returns True if valid.
        """
        if x in elem_set:
            return True
        
        # Check Case 1: x is a vertex (c=x)
        # Need a, b in S distinct from x, a != b, such that:
        # a+x in S, b+x in S, and (a+b in S or a+b == x)
        
        # Find all a in S such that a+x in S
        candidates_a = []
        max_a = n - x
        for a in elem_set:
            if a == x or a > max_a:
                continue
            if (a + x) in elem_set:
                candidates_a.append(a)
        
        k = len(candidates_a)
        if k > 1:
            for i in range(k):
                a = candidates_a[i]
                for j in range(i + 1, k):
                    b = candidates_a[j]
                    s = a + b
                    if s == x:
                        return False
                    if s <= n and s in elem_set:
                        return False
        
        # Check Case 2: x is a sum. x = a + b
        # Need c in S \ {a,b} such that a+c in S and b+c in S
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a:
                continue
            if a in elem_set and b in elem_set:
                # Check if there exists c in S \ {a,b} with a+c in S and b+c in S
                # c must be in 1..n-a and 1..n-b
                for c in elem_set:
                    if c == a or c == b:
                        continue
                    if c > n - a or c > n - b:
                        continue
                    if (a + c) in elem_set and (b + c) in elem_set:
                        return False
        
        return True

    def try_greedy_add(mask):
        """Try to add missing elements in ascending order. Returns new_mask."""
        elem_set = mask_to_set(mask)
        for x in range(1, n + 1):
            if x not in elem_set:
                if is_valid_addition(mask, x, elem_set):
                    mask |= (1 << (x - 1))
                    elem_set.add(x)
        return mask

    def try_greedy_add_desc(mask):
        """Try to add missing elements in descending order. Returns new_mask."""
        elem_set = mask_to_set(mask)
        for x in range(n, 0, -1):
            if x not in elem_set:
                if is_valid_addition(mask, x, elem_set):
                    mask |= (1 << (x - 1))
                    elem_set.add(x)
        return mask

    # Phase 1: Greedy addition from initial set (ascending then descending)
    current_mask = try_greedy_add(current_mask)
    current_mask = try_greedy_add_desc(current_mask)
    best_size = get_size(current_mask)
    best_mask = current_mask
    
    # Phase 2: Random local search with swap moves
    remaining_time = deadline - time.time()
    num_rounds = max(10, int(remaining_time / 2))
    
    for round_idx in range(num_rounds):
        if time.time() >= deadline - 5:
            break
        
        current = best_mask
        # Randomly remove 0-3 elements
        num_removals = random.randint(0, 3)
        for _ in range(num_removals):
            bits = [i for i in range(n) if (current >> i) & 1]
            if bits:
                idx = random.choice(bits)
                current &= ~(1 << idx)
        
        # Local search with add, remove, and swap moves
        round_deadline = min(time.time() + 1.5, deadline - 5)
        attempts = 0
        max_attempts = 80000
        while time.time() < round_deadline and attempts < max_attempts:
            attempts += 1
            move_type = random.random()
            
            if move_type < 0.5:
                # Add move
                x = random.randint(1, n)
                if not (current >> (x - 1)) & 1:
                    elem_set = mask_to_set(current)
                    if is_valid_addition(current, x, elem_set):
                        current |= (1 << (x - 1))
                        
            elif move_type < 0.7:
                # Remove move
                idx = random.randint(0, n - 1)
                if (current >> idx) & 1:
                    current &= ~(1 << idx)
                    
            else:
                # Swap move: remove one, try to add another
                bits = [i for i in range(n) if (current >> i) & 1]
                if bits:
                    remove_idx = random.choice(bits)
                    trial = current & ~(1 << remove_idx)
                    x = random.randint(1, n)
                    if not (trial >> (x - 1)) & 1:
                        elem_set = mask_to_set(trial)
                        if is_valid_addition(trial, x, elem_set):
                            trial |= (1 << (x - 1))
                            if get_size(trial) >= get_size(current):
                                current = trial
        
        if get_size(current) > best_size:
            best_size = get_size(current)
            best_mask = current
            best_mask = try_greedy_add(best_mask)
            best_mask = try_greedy_add_desc(best_mask)
            best_size = get_size(best_mask)
    
    # Final greedy passes
    best_mask = try_greedy_add(best_mask)
    best_mask = try_greedy_add_desc(best_mask)
    best_size = get_size(best_mask)
    
    # Final Verification and Cleanup
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    # Check for violations
    result_set = set(result)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(j + 1, len(result)):
                a, b, c = result[i], result[j], result[k]
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 in result_set and s2 in result_set and s3 in result_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
        
    if not valid:
        return list(range(65, 129))
        
    return result
