import time
import random

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin before 90s kill

    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 512, d >= 1
    # For each element x, precompute which other elements it conflicts with
    # Actually, for the greedy approach, we need to check if adding x creates a 4-AP
    # with existing elements. Let's precompute for each x, the list of (y1, y2, y3)
    # such that {x, y1, y2, y3} forms a 4-AP. But that's expensive to store.
    
    # Better approach: For each x, precompute all d values and the positions
    # that would form a 4-AP with x.
    
    # Let's use a bitmask approach for speed
    # Represent the set S as a bitmask (Python int)
    
    best_size = 0
    best_mask = 0
    best_set = set()
    
    def can_add(mask, x):
        """Check if adding x to the set represented by mask creates a 4-AP."""
        # x can be any of the 4 terms in a 4-AP
        # For each d, check if the other 3 terms are in mask
        max_d = N // 3  # 170
        for d in range(1, max_d + 1):
            # Case 1: x is the smallest (a)
            # Terms: x, x+d, x+2d, x+3d
            if x + 3 * d <= N:
                if ((mask >> (x + d - 1)) & 1) and ((mask >> (x + 2 * d - 1)) & 1) and ((mask >> (x + 3 * d - 1)) & 1):
                    return False
            # Case 2: x is the second term (a+d)
            # Terms: x-d, x, x+d, x+2d
            if x - d >= 1 and x + 2 * d <= N:
                if ((mask >> (x - d - 1)) & 1) and ((mask >> (x + d - 1)) & 1) and ((mask >> (x + 2 * d - 1)) & 1):
                    return False
            # Case 3: x is the third term (a+2d)
            # Terms: x-2d, x-d, x, x+d
            if x - 2 * d >= 1 and x + d <= N:
                if ((mask >> (x - 2 * d - 1)) & 1) and ((mask >> (x - d - 1)) & 1) and ((mask >> (x + d - 1)) & 1):
                    return False
            # Case 4: x is the largest (a+3d)
            # Terms: x-3d, x-2d, x-d, x
            if x - 3 * d >= 1:
                if ((mask >> (x - 3 * d - 1)) & 1) and ((mask >> (x - 2 * d - 1)) & 1) and ((mask >> (x - d - 1)) & 1):
                    return False
        return True

    def greedy_random(seed):
        """Greedy algorithm with random ordering."""
        rng = random.Random(seed)
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        
        mask = 0
        for x in perm:
            if can_add(mask, x):
                mask |= (1 << (x - 1))
        return mask

    def greedy_natural():
        """Greedy algorithm with natural ordering."""
        mask = 0
        for x in range(1, N + 1):
            if can_add(mask, x):
                mask |= (1 << (x - 1))
        return mask

    # Try natural order first
    mask = greedy_natural()
    size = bin(mask).count('1')
    best_size = size
    best_mask = mask
    best_set = set(i + 1 for i in range(N) if (mask >> i) & 1)

    # Try many random permutations
    num_tries = 0
    for seed in range(10000):
        if time.time() > deadline:
            break
        
        mask = greedy_random(seed)
        size = bin(mask).count('1')
        
        if size > best_size:
            best_size = size
            best_mask = mask
            best_set = set(i + 1 for i in range(N) if (mask >> i) & 1)
        
        num_tries += 1

    # Local search: try to improve by swapping elements
    if time.time() < deadline - 5:
        current_mask = best_mask
        current_size = best_size
        
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding random elements not in the set
            candidates = [x for x in range(1, N + 1) if not ((current_mask >> (x - 1)) & 1)]
            random.shuffle(candidates)
            
            for x in candidates[:50]:  # Try up to 50 candidates
                if can_add(current_mask, x):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    improved = True
                    break
            
            if not improved:
                # Try removing one element and adding two others
                elements = [x for x in range(1, N + 1) if (current_mask >> (x - 1)) & 1]
                random.shuffle(elements)
                
                for rem in elements[:20]:  # Try removing up to 20 elements
                    new_mask = current_mask & ~(1 << (rem - 1))
                    new_size = current_size - 1
                    
                    # Try to add elements
                    candidates = [x for x in range(1, N + 1) if not ((new_mask >> (x - 1)) & 1)]
                    random.shuffle(candidates)
                    
                    added = 0
                    for x in candidates:
                        if added >= 2:
                            break
                        if can_add(new_mask, x):
                            new_mask |= (1 << (x - 1))
                            new_size += 1
                            added += 1
                    
                    if new_size > current_size:
                        current_mask = new_mask
                        current_size = new_size
                        improved = True
                        break
            
            if not improved:
                break
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            best_set = set(i + 1 for i in range(N) if (best_mask >> i) & 1)

    return sorted(best_set)
