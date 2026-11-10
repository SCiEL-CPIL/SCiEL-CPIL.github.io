# EVOLVE-BLOCK-START
import numpy as np
import time
import random

M = 205

def _compute_forbidden():
    forbidden = set()
    for x in range(1, M):
        val = pow(x, 2, M)
        if val != 0:
            forbidden.add(val)
    return forbidden

FORBIDDEN = _compute_forbidden()

# Precompute conflict masks
# conflict_mask[i] is a bitset of all j such that i and j cannot coexist
CONFLICT_MASKS = [0] * M
for i in range(M):
    mask = 0
    for j in range(M):
        if i != j and (i - j) % M in FORBIDDEN:
            mask |= (1 << j)
    CONFLICT_MASKS[i] = mask

def _popcount(x):
    return bin(x).count('1')

def _can_add(cand, sel_bits):
    return (sel_bits & CONFLICT_MASKS[cand]) == 0

def _greedy_init():
    sel = 0
    remaining = list(range(M))
    random.shuffle(remaining)
    for v in remaining:
        if _can_add(v, sel):
            sel |= (1 << v)
    return sel

def _local_search(bits, deadline):
    """
    Perform local search to maximize the size of the independent set.
    Uses 1-swaps and 2-swaps.
    """
    current = bits
    current_size = _popcount(current)
    best = current
    best_size = current_size
    
    # List of vertices for random selection
    vertices = list(range(M))
    
    # We will try to improve by:
    # 1. Adding a vertex if possible (Size +1)
    # 2. Swapping 1 vertex (Size 0) - helps rearrange for future additions
    # 3. Swapping 2 vertices (Size 0) - helps escape local minima
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add a vertex
        # We can iterate through all vertices not in current
        # To make this faster, we can just pick random ones, but a full scan is cheap (205 items)
        for v in vertices:
            if not (current >> v) & 1:
                if _can_add(v, current):
                    current |= (1 << v)
                    current_size += 1
                    best = current
                    best_size = current_size
                    improved = True
                    break # Restart scan after addition
        
        if improved:
            continue
            
        # Phase 2: Try 1-swaps
        # Pick a random u in S and v not in S
        # If we find a valid swap, we accept it (neutral move)
        # To avoid cycling, we can use a Tabu list or just random restarts
        # For now, we accept any valid swap.
        
        # Efficient swap finding:
        # We can iterate through u in S and check all v not in S.
        # But that's O(N^2). 205^2 = 42k. This is very fast.
        # Let's do a full scan for 1-swaps to find ANY valid swap.
        found_swap = False
        # Get list of vertices in S
        s_vertices = [i for i in range(M) if (current >> i) & 1]
        # Get list of vertices not in S
        not_s_vertices = [i for i in range(M) if not (current >> i) & 1]
        
        # Randomize order to avoid bias
        random.shuffle(s_vertices)
        random.shuffle(not_s_vertices)
        
        for u in s_vertices:
            temp_state = current & ~(1 << u)
            # Check if any v in not_s_vertices can be added to temp_state
            # We can check all v
            for v in not_s_vertices:
                if (temp_state & CONFLICT_MASKS[v]) == 0:
                    # Valid swap
                    current = temp_state | (1 << v)
                    # Size remains same
                    found_swap = True
                    break
            if found_swap:
                break
        
        if found_swap:
            continue
            
        # Phase 3: Try 2-swaps
        # This is more expensive. We only do it if stuck.
        # Pick two random u1, u2 in S and two random v1, v2 not in S
        # Check if {v1, v2} is independent and compatible with S \ {u1, u2}
        
        # To avoid O(N^4), we sample.
        # Sample 50 random 2-swap attempts
        for _ in range(50):
            if len(s_vertices) < 2 or len(not_s_vertices) < 2:
                break
            u1, u2 = random.sample(s_vertices, 2)
            v1, v2 = random.sample(not_s_vertices, 2)
            
            # Check if v1 and v2 are compatible with each other
            if (1 << v1) & CONFLICT_MASKS[v2]:
                continue
                
            temp_state = current & ~(1 << u1) & ~(1 << u2)
            
            # Check if v1 and v2 are compatible with temp_state
            if (temp_state & CONFLICT_MASKS[v1]) == 0 and (temp_state & CONFLICT_MASKS[v2]) == 0:
                current = temp_state | (1 << v1) | (1 << v2)
                # Size remains same
                found_swap = True
                break
        
        if found_swap:
            continue
            
        # If no improvement found (no add, no 1-swap, no 2-swap)
        # We are in a local optimum.
        # Perturbation: Remove 3 random elements and add 3 random compatible ones?
        # Or just break and let ILS handle restart.
        # For this function, we just break to signal ILS to perturb.
        break
        
    return current, best, best_size

def furstenberg_sarkozy_205() -> np.ndarray:
    start = time.time()
    deadline = start + 580.0  # Safety margin
    
    random.seed(42)
    
    best_global = 0
    best_global_size = 0
    
    # Phase 1: Greedy initialization
    # Run multiple greedy starts to find a good initial solution
    for i in range(100):
        if time.time() > deadline - 100:
            break
        sel = _greedy_init()
        cnt = _popcount(sel)
        if cnt > best_global_size:
            best_global = sel
            best_global_size = cnt
            
    # Phase 2: Iterated Local Search
    # We will run local search, then perturb, then local search again.
    current_best = best_global
    current_size = best_global_size
    
    while time.time() < deadline - 10:
        # Local search on current_best
        improved_bits, best_bits, best_size = _local_search(current_best, deadline - 10)
        
        if best_size > best_global_size:
            best_global = best_bits
            best_global_size = best_size
            current_best = best_bits
            current_size = best_size
            
        # If we improved, we might stay here a bit more, but ILS usually perturbs immediately
        # Perturbation
        if time.time() > deadline - 10:
            break
            
        # Perturb current_best
        # Remove 5 random elements
        vertices_in = [i for i in range(M) if (current_best >> i) & 1]
        if len(vertices_in) < 5:
            # If set is small, just random restart
            current_best = _greedy_init()
        else:
            to_remove = random.sample(vertices_in, 5)
            for v in to_remove:
                current_best &= ~(1 << v)
            current_size -= 5
            
            # Add 5 random compatible elements
            # We try to add them one by one
            vertices_out = [i for i in range(M) if not (current_best >> i) & 1]
            random.shuffle(vertices_out)
            added = 0
            for v in vertices_out:
                if added >= 5:
                    break
                if _can_add(v, current_best):
                    current_best |= (1 << v)
                    current_size += 1
                    added += 1
                    
    result = [i for i in range(M) if (best_global >> i) & 1]
    return np.array(sorted(result), dtype=np.int64)

# EVOLVE-BLOCK-END
