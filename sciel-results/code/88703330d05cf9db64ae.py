# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses randomized local search (Max Independent Set heuristic) with bitsets for speed.
    """
    m = 205
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Precompute conflict masks
    # masks[i] is a bitset where bit j is 1 if i and j conflict
    masks = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << j)
        masks[i] = mask
        
    # 3. Helper to get list of set bits
    def get_bits(mask):
        return [i for i in range(m) if (mask >> i) & 1]

    # 4. Greedy Initialization
    # Pick vertices that have fewest conflicts with other available vertices
    def greedy_init():
        cur_mask = 0
        cur_size = 0
        while True:
            # Find all addable
            addable = []
            for v in range(m):
                if not (cur_mask >> v) & 1:
                    if (masks[v] & cur_mask) == 0:
                        addable.append(v)
            if not addable:
                break
            
            # Heuristic: Pick v with minimum conflicts within 'addable'
            best_v = addable[0]
            min_conf = m
            for v in addable:
                conf = 0
                for u in addable:
                    if u != v and (masks[v] >> u) & 1:
                        conf += 1
                if conf < min_conf:
                    min_conf = conf
                    best_v = v
            
            cur_mask |= (1 << best_v)
            cur_size += 1
        return cur_mask, cur_size

    best_mask, best_size = greedy_init()
    current_mask, current_size = best_mask, best_size
    
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    random.seed(42)
    
    # 5. Randomized Local Search
    while time.time() < deadline:
        # Find addable
        addable = []
        for v in range(m):
            if not (current_mask >> v) & 1:
                if (masks[v] & current_mask) == 0:
                    addable.append(v)
        
        if addable:
            # Pick a random addable vertex
            v = random.choice(addable)
            current_mask |= (1 << v)
            current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        else:
            # Stuck. Kick: remove some vertices
            bits = get_bits(current_mask)
            if not bits:
                # Empty set, just add a random one
                v = random.randint(0, m - 1)
                current_mask |= (1 << v)
                current_size = 1
            else:
                # Remove 1 to 4 random vertices
                k = random.choice([1, 2, 3, 4])
                k = min(k, len(bits))
                to_remove = random.sample(bits, k)
                for r in to_remove:
                    current_mask &= ~(1 << r)
                current_size -= k
                
    # 6. Convert best_mask to array
    result = []
    for i in range(m):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
