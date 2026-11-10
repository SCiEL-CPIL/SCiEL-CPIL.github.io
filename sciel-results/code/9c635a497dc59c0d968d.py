# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses Bitmask representation and Randomized Local Search with swap moves.
    """
    m = 205
    k = 2
    
    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Precompute Conflict Masks
    conflict_masks = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            diff = (i - j) % m
            if diff in forbidden:
                mask |= (1 << j)
        conflict_masks[i] = mask
    
    # Helper: Greedy Fill - add as many compatible vertices as possible
    def greedy_fill(S, mask):
        improved = False
        while True:
            compatible = []
            for v in range(m):
                if v not in S and (conflict_masks[v] & mask) == 0:
                    compatible.append(v)
            if not compatible:
                break
            v_to_add = random.choice(compatible)
            S.add(v_to_add)
            mask |= (1 << v_to_add)
            improved = True
        return improved
    
    # Helper: Try 1-for-2 swap
    # Remove one vertex from S, try to add two compatible vertices
    def try_swap_1_for_2(S, mask):
        if len(S) < 1:
            return False
        # Try removing each vertex in S
        for v_remove in list(S):
            # Temporarily remove v_remove
            new_mask = mask & ~(1 << v_remove)
            # Find compatible vertices with new_mask
            compatible = []
            for v in range(m):
                if v == v_remove:
                    continue
                if v not in S and (conflict_masks[v] & new_mask) == 0:
                    compatible.append(v)
            if len(compatible) >= 2:
                # Try to find two compatible vertices that don't conflict with each other
                random.shuffle(compatible)
                for i in range(len(compatible)):
                    for j in range(i + 1, len(compatible)):
                        v1, v2 = compatible[i], compatible[j]
                        # Check if v1 and v2 conflict
                        if (conflict_masks[v1] & (1 << v2)) == 0:
                            # Accept the swap
                            S.remove(v_remove)
                            S.add(v1)
                            S.add(v2)
                            return True
        return False
    
    # Helper: Try 1-for-1 swap
    # Remove one vertex from S, add a different compatible vertex
    def try_swap_1_for_1(S, mask):
        if len(S) < 1:
            return False
        for v_remove in list(S):
            new_mask = mask & ~(1 << v_remove)
            compatible = []
            for v in range(m):
                if v == v_remove and v not in S:
                    continue
                if v not in S and (conflict_masks[v] & new_mask) == 0:
                    compatible.append(v)
            if compatible:
                v_add = random.choice(compatible)
                S.remove(v_remove)
                S.add(v_add)
                return True
        return False
    
    # Local Search
    start_time = time.time()
    deadline = 600  # seconds
    random.seed(42)
    
    best_S = set()
    best_mask = 0
    best_size = 0
    
    # Multiple randomized greedy initializations
    for init_iter in range(20):
        if time.time() - start_time > deadline - 30:
            break
        
        init_S = set()
        init_mask = 0
        order = list(range(m))
        random.shuffle(order)
        
        for v in order:
            if (conflict_masks[v] & init_mask) == 0:
                init_S.add(v)
                init_mask |= (1 << v)
        
        if len(init_S) > best_size:
            best_size = len(init_S)
            best_S = set(init_S)
            best_mask = init_mask
    
    # Main local search loop
    current_S = set(best_S)
    current_mask = best_mask
    current_size = len(current_S)
    
    no_improve_count = 0
    max_no_improve = 100
    iteration = 0
    
    while time.time() - start_time < deadline - 10:
        iteration += 1
        improved = False
        
        # Phase 1: Greedy fill
        if greedy_fill(current_S, current_mask):
            improved = True
        
        # Update current_mask after fill
        current_mask = 0
        for v in current_S:
            current_mask |= (1 << v)
        current_size = len(current_S)
        
        if current_size > best_size:
            best_size = current_size
            best_S = set(current_S)
            best_mask = current_mask
            no_improve_count = 0
        else:
            # Try swap moves
            swap_improved = False
            
            # Try 1-for-2 swaps (increases size by 1)
            if current_size < 205:
                for _ in range(5):  # Try a few swap attempts
                    if time.time() - start_time > deadline - 10:
                        break
                    if try_swap_1_for_2(current_S, current_mask):
                        swap_improved = True
                        break
            
            if swap_improved:
                current_mask = 0
                for v in current_S:
                    current_mask |= (1 << v)
                current_size = len(current_S)
                improved = True
                
                if current_size > best_size:
                    best_size = current_size
                    best_S = set(current_S)
                    best_mask = current_mask
                    no_improve_count = 0
            else:
                # Try 1-for-1 swaps (same size, different configuration)
                if random.random() < 0.3:
                    if try_swap_1_for_1(current_S, current_mask):
                        current_mask = 0
                        for v in current_S:
                            current_mask |= (1 << v)
                        # Don't count as improvement for size, but helps escape local optima
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            
            if not improved:
                # Perturbation: remove random vertices
                if no_improve_count > max_no_improve:
                    kick_size = random.randint(2, 5)
                    if len(current_S) > kick_size:
                        to_remove = random.sample(list(current_S), kick_size)
                        for v in to_remove:
                            current_S.remove(v)
                        current_mask = 0
                        for v in current_S:
                            current_mask |= (1 << v)
                        current_size = len(current_S)
                    no_improve_count = 0
                elif len(current_S) > 0:
                    kick_size = random.randint(1, 2)
                    if len(current_S) >= kick_size:
                        to_remove = random.sample(list(current_S), kick_size)
                        for v in to_remove:
                            current_S.remove(v)
                        current_mask = 0
                        for v in current_S:
                            current_mask |= (1 << v)
                        current_size = len(current_S)

    return np.array(list(best_S), dtype=np.int64)

# EVOLVE-BLOCK-END
