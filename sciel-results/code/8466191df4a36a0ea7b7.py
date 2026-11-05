import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Build conflict structure
    conflicts = []
    conflict_mask = [0] * n
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                conflicts.append((i, j))
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                adj[i].add(j)
                adj[j].add(i)
    
    weights = [1.0 / nums[i] for i in range(n)]
    
    def is_valid_mask(mask):
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            if (conflict_mask[i] & (mask ^ lsb)) != 0:
                return False
            m ^= lsb
        return True
    
    def mask_weight(mask):
        total = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            total += weights[i]
            m ^= lsb
        return total
    
    # Try MIP
    best_mask = 0
    best_val = 0.0
    
    c = np.array([-weights[i] for i in range(n)])
    m_conf = len(conflicts)
    if m_conf > 0:
        A = np.zeros((m_conf, n))
        for k, (i, j) in enumerate(conflicts):
            A[k, i] = 1.0
            A[k, j] = 1.0
        constraints = LinearConstraint(A, lb=np.zeros(m_conf), ub=np.ones(m_conf))
    else:
        constraints = LinearConstraint(np.zeros((1, n)), lb=0, ub=1)
    
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n)
    
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if result is not None and result.success and result.x is not None:
            mask = 0
            for i in range(n):
                if result.x[i] > 0.5:
                    mask |= (1 << i)
            if is_valid_mask(mask):
                best_mask = mask
                best_val = mask_weight(mask)
    except Exception:
        pass
    
    # Greedy fallback / initialization
    if best_val < 0.1:
        order = sorted(range(n), key=lambda i: -weights[i])
        mask = 0
        for i in order:
            if (conflict_mask[i] & mask) == 0:
                mask |= (1 << i)
        if is_valid_mask(mask):
            val = mask_weight(mask)
            if val > best_val:
                best_mask = mask
                best_val = val
    
    # Local search with bitmask
    def local_search(mask, max_time):
        nonlocal best_mask, best_val
        current_mask = mask
        current_val = mask_weight(current_mask)
        improved = True
        iters = 0
        max_iters = 5000
        
        while improved and iters < max_iters and time.time() < max_time:
            improved = False
            iters += 1
            
            # Phase 1: Try adding each non-selected element
            for i in range(n):
                if (current_mask >> i) & 1:
                    continue
                if (conflict_mask[i] & current_mask) == 0:
                    new_mask = current_mask | (1 << i)
                    new_val = current_val + weights[i]
                    if new_val > current_val + 1e-12:
                        current_mask = new_mask
                        current_val = new_val
                        improved = True
                        if current_val > best_val:
                            best_val = current_val
                            best_mask = current_mask
                        break
            
            # Phase 2: Try single swaps (remove one, add one)
            if not improved:
                for i in range(n):
                    if not ((current_mask >> i) & 1):
                        continue
                    for j in range(n):
                        if (current_mask >> j) & 1:
                            continue
                        new_mask = (current_mask & ~(1 << i)) | (1 << j)
                        if is_valid_mask(new_mask):
                            new_val = current_val - weights[i] + weights[j]
                            if new_val > current_val + 1e-12:
                                current_mask = new_mask
                                current_val = new_val
                                improved = True
                                if current_val > best_val:
                                    best_val = current_val
                                    best_mask = current_mask
                                break
                    if improved:
                        break
            
            # Phase 3: Try double swaps (remove two, add two) for escaping local optima
            if not improved and time.time() < max_time:
                selected = [i for i in range(n) if (current_mask >> i) & 1]
                unselected = [i for i in range(n) if not ((current_mask >> i) & 1)]
                found = False
                for idx_a in range(len(selected)):
                    for idx_b in range(idx_a + 1, len(selected)):
                        i, j = selected[idx_a], selected[idx_b]
                        for x in unselected:
                            if (conflict_mask[x] & current_mask) & ~( (1<<i)|(1<<j) ):
                                continue
                            for y in unselected:
                                if x == y:
                                    continue
                                if (conflict_mask[x] & (1<<y)) or (conflict_mask[y] & (1<<x)):
                                    continue
                                new_mask = (current_mask & ~((1<<i)|(1<<j))) | ((1<<x)|(1<<y))
                                if is_valid_mask(new_mask):
                                    new_val = current_val - weights[i] - weights[j] + weights[x] + weights[y]
                                    if new_val > current_val + 1e-12:
                                        current_mask = new_mask
                                        current_val = new_val
                                        improved = True
                                        found = True
                                        if current_val > best_val:
                                            best_val = current_val
                                            best_mask = current_mask
                                        break
                            if found:
                                break
                    if found:
                        break
                if found:
                    continue
            
            if not improved:
                break
        
        return current_mask, current_val
    
    # Run local search from MIP/greedy solution
    if time.time() < DEADLINE:
        best_mask, best_val = local_search(best_mask, DEADLINE)
    
    # Additional restarts with different orderings if time remains
    if time.time() < DEADLINE - 5:
        # Try reverse-greedy (smallest first)
        order = sorted(range(n), key=lambda i: weights[i])
        mask = 0
        for i in order:
            if (conflict_mask[i] & mask) == 0:
                mask |= (1 << i)
        if is_valid_mask(mask):
            val = mask_weight(mask)
            if val > best_val:
                best_mask, best_val = local_search(mask, DEADLINE)
    
    if time.time() < DEADLINE - 5:
        # Try random restarts
        import random
        random.seed(42)
        for _ in range(10):
            if time.time() >= DEADLINE - 2:
                break
            order = list(range(n))
            random.shuffle(order)
            mask = 0
            for i in order:
                if (conflict_mask[i] & mask) == 0:
                    mask |= (1 << i)
            if is_valid_mask(mask):
                val = mask_weight(mask)
                if val > best_val:
                    best_mask, best_val = local_search(mask, DEADLINE)
    
    # Final polish: exhaustive single-add and single-swap pass
    if time.time() < DEADLINE - 1:
        current_mask = best_mask
        current_val = best_val
        # Add all possible
        for i in range(n):
            if not ((current_mask >> i) & 1):
                if (conflict_mask[i] & current_mask) == 0:
                    new_mask = current_mask | (1 << i)
                    new_val = current_val + weights[i]
                    if new_val > current_val + 1e-12:
                        current_mask = new_mask
                        current_val = new_val
        # Swap pass
        for i in range(n):
            if not ((current_mask >> i) & 1):
                continue
            for j in range(n):
                if (current_mask >> j) & 1:
                    continue
                new_mask = (current_mask & ~(1 << i)) | (1 << j)
                if is_valid_mask(new_mask):
                    new_val = current_val - weights[i] + weights[j]
                    if new_val > current_val + 1e-12:
                        current_mask = new_mask
                        current_val = new_val
        if current_val > best_val:
            best_mask = current_mask
            best_val = current_val
    
    # Convert mask to result list
    result_list = []
    m = best_mask
    while m:
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        result_list.append(nums[i])
        m ^= lsb
    
    if not result_list:
        result_list = [2]
    
    return result_list
