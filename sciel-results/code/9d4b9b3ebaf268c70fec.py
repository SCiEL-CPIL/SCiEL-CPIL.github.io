import time
import random

N = 512
DEADLINE_SECONDS = 55

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # 1. Precomputation
    # bit_mask[i] represents the bit for element i+1
    bit_mask = [1 << i for i in range(N)]
    
    # conflicts[x] is a list of bitmasks. 
    # Each mask represents the other 3 elements in a 4-AP containing x.
    # If (S & mask) == mask, then adding x creates a 4-AP.
    conflicts = [[] for _ in range(N)] # Index 0 for element 1, ... Index N-1 for element N
    
    # Generate all 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            
            # Elements are a, b, c, e (1-indexed)
            # Indices are a-1, b-1, c-1, e-1
            idx_a = a - 1
            idx_b = b - 1
            idx_c = c - 1
            idx_e = e - 1
            
            mask_all = bit_mask[idx_a] | bit_mask[idx_b] | bit_mask[idx_c] | bit_mask[idx_e]
            
            # For each element in the AP, the conflict mask is the others
            # For a: mask is b|c|e
            # For b: mask is a|c|e
            # etc.
            
            mask_bce = mask_all ^ bit_mask[idx_a]
            mask_ace = mask_all ^ bit_mask[idx_b]
            mask_abe = mask_all ^ bit_mask[idx_c]
            mask_abc = mask_all ^ bit_mask[idx_e]
            
            conflicts[idx_a].append(mask_bce)
            conflicts[idx_b].append(mask_ace)
            conflicts[idx_c].append(mask_abe)
            conflicts[idx_e].append(mask_abc)
    
    # Precompute conflict count for each element (how many 4-APs it's part of)
    conflict_count = [len(conflicts[i]) for i in range(N)]
    
    def is_valid_add(x_idx, S):
        """Check if element x_idx (0-indexed) can be added to S."""
        for m in conflicts[x_idx]:
            if (S & m) == m:
                return False
        return True
    
    def greedy_add_smart(S, order):
        """Add elements from order to S if valid, using smart ordering."""
        for x_idx in order:
            if not (S & bit_mask[x_idx]):
                if is_valid_add(x_idx, S):
                    S |= bit_mask[x_idx]
        return S
    
    def greedy_add_by_conflict(S):
        """Greedy add in order of increasing conflict count (fewer conflicts first)."""
        # Sort indices by conflict count (ascending)
        order = sorted(range(N), key=lambda x: conflict_count[x])
        return greedy_add_smart(S, order)
    
    # 2. Initial Greedy Construction - try multiple strategies
    best_S = 0
    best_size = 0
    
    # Strategy 1: Standard greedy (1 to N)
    S1 = greedy_add_smart(0, list(range(N)))
    if S1.bit_count() > best_size:
        best_S = S1
        best_size = S1.bit_count()
    
    # Strategy 2: Greedy by conflict count (fewer conflicts first)
    S2 = greedy_add_by_conflict(0)
    if S2.bit_count() > best_size:
        best_S = S2
        best_size = S2.bit_count()
    
    # Strategy 3: Randomized greedy multiple times
    for _ in range(20):
        if time.time() >= deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        S3 = greedy_add_smart(0, order)
        if S3.bit_count() > best_size:
            best_S = S3
            best_size = S3.bit_count()
    
    S = best_S
    
    # 3. Local Search with Simulated Annealing
    current_temp = 1.0
    cooling_rate = 0.995
    min_temp = 0.001
    
    while time.time() < deadline:
        # Check time more frequently
        if time.time() >= deadline - 2:
            break
        
        # Perturbation: remove k random elements
        k = random.randint(1, 8)
        set_indices = [i for i in range(N) if (S & bit_mask[i])]
        
        if len(set_indices) < k:
            k = max(1, len(set_indices) // 2)
            if k == 0:
                continue
        
        to_remove = random.sample(set_indices, k)
        
        S_perturbed = S
        for idx in to_remove:
            S_perturbed &= ~bit_mask[idx]
        
        # Re-greedy with different strategies
        # Strategy A: Random order
        order_rand = list(range(N))
        random.shuffle(order_rand)
        S_new_a = greedy_add_smart(S_perturbed, order_rand)
        
        # Strategy B: Conflict-ordered
        S_new_b = greedy_add_by_conflict(S_perturbed)
        
        # Pick the better one
        if S_new_a.bit_count() >= S_new_b.bit_count():
            S_new = S_new_a
        else:
            S_new = S_new_b
        
        size_new = S_new.bit_count()
        size_current = S.bit_count()
        
        # Simulated annealing acceptance
        delta = size_new - size_current
        if delta >= 0:
            S = S_new
            if size_new > best_size:
                best_S = S_new
                best_size = size_new
        else:
            # Accept worse solution with probability exp(delta/temp)
            if random.random() < pow(2.718281828, delta / max(current_temp, 0.001)):
                S = S_new
        
        # Cool down
        current_temp *= cooling_rate
        if current_temp < min_temp:
            # Restart with higher temperature occasionally
            current_temp = 0.5
    
    # Final pass: try to add any remaining elements
    for i in range(N):
        if time.time() >= deadline:
            break
        if not (best_S & bit_mask[i]):
            if is_valid_add(i, best_S):
                best_S |= bit_mask[i]
    
    # Convert best_S to sorted list
    result = [i + 1 for i in range(N) if (best_S & bit_mask[i])]
    return result
