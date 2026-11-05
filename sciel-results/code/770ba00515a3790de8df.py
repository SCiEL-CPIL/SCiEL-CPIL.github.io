import time
import random
from itertools import combinations

def _build_conflict_bitsets(n=256):
    """
    Build conflict bitsets for the graph.
    Vertex v (1-indexed) corresponds to bit (v-1).
    conflicts[v] is an integer where bit j is set if v and j+1 conflict.
    """
    conflicts = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i] |= (1 << (j - 1))
                conflicts[j] |= (1 << (i - 1))
    return conflicts

def _greedy_init(conflicts, n=256, seed=42):
    """
    Find a good initial independent set using a greedy heuristic with random restarts.
    """
    best_mask = 0
    best_size = 0
    rng = random.Random(seed)
    
    def greedy(order):
        selected_mask = 0
        count = 0
        for v in order:
            v_bit = 1 << (v - 1)
            if (conflicts[v] & selected_mask) == 0:
                selected_mask |= v_bit
                count += 1
        return count, selected_mask

    # Try degree-sorted order (ascending degree)
    order = sorted(range(1, n + 1), key=lambda x: (conflicts[x].bit_count(), x))
    c, m = greedy(order)
    if c > best_size:
        best_size = c
        best_mask = m

    # Try random restarts
    for _ in range(200):
        rand_order = list(range(1, n + 1))
        rng.shuffle(rand_order)
        c, m = greedy(rand_order)
        if c > best_size:
            best_size = c
            best_mask = m

    return best_size, best_mask

def _local_search(conflicts, n=256, time_limit=50.0):
    """
    Local search to improve the independent set.
    Uses bitset operations for fast conflict checking.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(123)
    
    # Start with a good greedy solution
    best_size, best_mask = _greedy_init(conflicts, n, seed=999)
    
    # Get list of vertices from mask
    def mask_to_list(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx + 1)
            m ^= lsb
        return verts
    
    def list_to_mask(verts):
        m = 0
        for v in verts:
            m |= (1 << (v - 1))
        return m
    
    # Local search: try adding vertices, removing vertices, swapping
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_verts = mask_to_list(best_mask)
        current_set = set(current_verts)
        
        # Try adding a vertex
        for v in range(1, n + 1):
            if v in current_set:
                continue
            # Check if v conflicts with any vertex in current set
            if (conflicts[v] & best_mask) == 0:
                new_mask = best_mask | (1 << (v - 1))
                new_size = best_size + 1
                best_mask = new_mask
                best_size = new_size
                current_verts.append(v)
                current_set.add(v)
                improved = True
                break
        
        if time.time() > deadline:
            break
        
        # Try swapping: remove one vertex, add another
        if not improved:
            for i, v_remove in enumerate(current_verts):
                if time.time() > deadline:
                    break
                mask_without = best_mask & ~(1 << (v_remove - 1))
                # Try adding any vertex not in current set
                for v_add in range(1, n + 1):
                    if v_add in current_set or v_add == v_remove:
                        continue
                    # Check if v_add conflicts with (current set - v_remove)
                    if (conflicts[v_add] & mask_without) == 0:
                        # Valid swap
                        new_mask = mask_without | (1 << (v_add - 1))
                        # Size stays the same, but might enable further additions
                        # For now, just accept swaps that allow growth
                        # Actually, for local search on independent set, 
                        # we want to try to increase size.
                        # Let's try: remove v_remove, add v_add, then try to add more
                        # This is getting complex. Let's use a simpler approach.
                        pass
                # Skip complex swaps for now, focus on additions and removals
            # Try removing a vertex to allow adding two others?
            # This is a 2-exchange neighborhood.
            # For simplicity, let's try: for each vertex v in set, remove it,
            # then greedily add as many as possible from remaining.
            for v_remove in current_verts:
                if time.time() > deadline:
                    break
                mask_without = best_mask & ~(1 << (v_remove - 1))
                # Greedily add vertices
                add_order = sorted(range(1, n + 1), key=lambda x: (conflicts[x].bit_count(), x))
                trial_mask = mask_without
                trial_size = best_size - 1
                for u in add_order:
                    if (conflicts[u] & trial_mask) == 0 and (trial_mask >> (u-1)) & 1 == 0:
                        trial_mask |= (1 << (u - 1))
                        trial_size += 1
                if trial_size > best_size:
                    best_mask = trial_mask
                    best_size = trial_size
                    current_verts = mask_to_list(best_mask)
                    current_set = set(current_verts)
                    improved = True
                    break
    
    return best_size, best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    conflicts = _build_conflict_bitsets(n)
    
    # Step 1: Greedy initialization
    init_size, init_mask = _greedy_init(conflicts, n, seed=42)
    best_size = init_size
    best_mask = init_mask
    
    # Step 2: Local search with time limit
    remaining = deadline - 5.0 - time.time()
    if remaining > 2.0:
        ls_size, ls_mask = _local_search(conflicts, n, time_limit=remaining)
        if ls_size > best_size:
            best_size = ls_size
            best_mask = ls_mask
    
    # Reconstruct set from best_mask
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    # Verify
    result_set = set(result)
    for a in result:
        for b in result:
            if a < b and (a * b) % (a + b) == 0:
                # Invalid, fallback to greedy
                _, init_mask = _greedy_init(conflicts, n, seed=42)
                result = [i+1 for i in range(n) if (init_mask >> i) & 1]
                break
    
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(len(res))
