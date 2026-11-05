"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build conflict sets: conflict[i] is set of j such that vertices[i] + vertices[j] is a square
    # Using 0-indexed internally
    conflict = [set() for _ in range(N)]
    valid_vertices = []
    
    for i in range(N):
        a = i + 1
        # Check self conflict: 2*a should not be a perfect square
        if 2 * a in squares:
            continue  # This vertex cannot be in the set
        valid_vertices.append(i)
    
    # Build conflicts for valid vertices
    for i in valid_vertices:
        a = i + 1
        for j in valid_vertices:
            if j == i:
                continue
            b = j + 1
            if a + b in squares:
                conflict[i].add(j)
    
    # Precompute a compatibility check table for speed
    # compat[i][j] = True if i and j can coexist (or i==j)
    # We'll use a different approach: for each vertex, store its conflict set as a frozenset or use bitsets
    
    # Use bitsets for faster compatibility checks
    # valid_vertices is a list of 0-indexed vertex indices
    n_valid = len(valid_vertices)
    valid_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # Build conflict bitsets: conflict_bits[i] is a bitset where bit j is set if valid_vertices[j] conflicts with valid_vertices[i]
    conflict_bits = [0] * n_valid
    for i, v in enumerate(valid_vertices):
        for j, w in enumerate(valid_vertices):
            if j != i and w in conflict[v]:
                conflict_bits[i] |= (1 << j)
    
    def can_add(current_mask, vertex_idx):
        """Check if vertex_idx can be added given current_mask (bitmask of selected vertices)"""
        return (current_mask & conflict_bits[vertex_idx]) == 0
    
    def greedy_construct(order_indices):
        """Greedy construction given an ordering of valid vertex indices"""
        mask = 0
        count = 0
        for idx in order_indices:
            if can_add(mask, idx):
                mask |= (1 << idx)
                count += 1
        return mask, count
    
    def local_improve(mask, count):
        """Try to add any non-conflicting vertices not in the set"""
        changed = True
        while changed:
            changed = False
            for idx in range(n_valid):
                if not (mask & (1 << idx)) and can_add(mask, idx):
                    mask |= (1 << idx)
                    count += 1
                    changed = True
        return mask, count
    
    def local_search(mask, count, max_iterations=5000):
        """Swap-based local search to improve the set"""
        for _ in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            # Try removing one element and adding another
            # Get list of selected vertices
            selected = [i for i in range(n_valid) if mask & (1 << i)]
            for remove_idx in selected:
                temp_mask = mask & ~(1 << remove_idx)
                # Try to add a vertex not in temp_mask
                for add_idx in range(n_valid):
                    if temp_mask & (1 << add_idx):
                        continue
                    if can_add(temp_mask, add_idx):
                        new_mask = temp_mask | (1 << add_idx)
                        new_count = count  # same size, but might enable further improvements
                        # Only accept if it allows adding more later? For now, just swap if it doesn't reduce size
                        # Actually we want to increase size, so a swap alone doesn't help unless it enables additions
                        # Let's just do the swap and then try local_improve
                        new_mask2, new_count2 = local_improve(new_mask, count)
                        if new_count2 > count:
                            mask, count = new_mask2, new_count2
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return mask, count
    
    def two_swap_search(mask, count, max_iterations=2000):
        """2-swap local search: remove two, add two to potentially increase size"""
        for _ in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            selected = [i for i in range(n_valid) if mask & (1 << i)]
            n_sel = len(selected)
            
            for i in range(n_sel):
                for j in range(i + 1, n_sel):
                    remove1 = selected[i]
                    remove2 = selected[j]
                    temp_mask = mask & ~(1 << remove1) & ~(1 << remove2)
                    temp_count = count - 2
                    
                    # Try to add two vertices not in temp_mask
                    for add1 in range(n_valid):
                        if temp_mask & (1 << add1):
                            continue
                        if not can_add(temp_mask, add1):
                            continue
                        temp_mask1 = temp_mask | (1 << add1)
                        temp_count1 = temp_count + 1
                        
                        for add2 in range(n_valid):
                            if add2 == add1 or temp_mask1 & (1 << add2):
                                continue
                            if not can_add(temp_mask1, add2):
                                continue
                            new_mask = temp_mask1 | (1 << add2)
                            new_count = temp_count1 + 1
                            
                            # Try to add more
                            new_mask2, new_count2 = local_improve(new_mask, new_count)
                            if new_count2 > count:
                                mask, count = new_mask2, new_count2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        return mask, count
    
    # Run multiple random restarts
    best_mask = 0
    best_count = 0
    
    # First, try a deterministic greedy approach
    # Order by degree (fewest conflicts first)
    degree_order = sorted(range(n_valid), key=lambda i: bin(conflict_bits[i]).count('1'))
    mask, count = greedy_construct(degree_order)
    mask, count = local_improve(mask, count)
    mask, count = local_search(mask, count)
    mask, count = two_swap_search(mask, count)
    best_mask, best_count = mask, count
    
    # Try reverse degree order
    reverse_degree_order = sorted(range(n_valid), key=lambda i: -bin(conflict_bits[i]).count('1'))
    mask, count = greedy_construct(reverse_degree_order)
    mask, count = local_improve(mask, count)
    mask, count = local_search(mask, count)
    mask, count = two_swap_search(mask, count)
    if count > best_count:
        best_mask, best_count = mask, count
    
    # Random restarts
    all_indices = list(range(n_valid))
    while time.time() - start_time < DEADLINE:
        random.shuffle(all_indices)
        mask, count = greedy_construct(all_indices)
        mask, count = local_improve(mask, count)
        mask, count = local_search(mask, count)
        mask, count = two_swap_search(mask, count)
        
        if count > best_count:
            best_mask, best_count = mask, count
    
    # Final polish on the best result
    best_mask, best_count = local_improve(best_mask, best_count)
    best_mask, best_count = local_search(best_mask, best_count)
    best_mask, best_count = two_swap_search(best_mask, best_count)
    
    # Convert back to 1-indexed
    result = []
    for idx in range(n_valid):
        if best_mask & (1 << idx):
            result.append(valid_vertices[idx] + 1)
    
    return result
# EVOLVE-BLOCK-END
