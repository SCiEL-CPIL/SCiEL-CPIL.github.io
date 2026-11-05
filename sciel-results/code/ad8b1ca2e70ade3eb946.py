import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin

    N = 256

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency and filter forbidden
    adj_mask = [0] * N
    forbidden = set()
    
    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden.add(i)
        
    # Only consider non-forbidden vertices
    # Map original index to new index? No, keep original indices for clarity, just skip forbidden in logic.
    # But bitmasks are faster if we compact them.
    # Let's compact.
    
    valid_vertices = [i for i in range(N) if i not in forbidden]
    M = len(valid_vertices)
    # Map old index to new index
    old_to_new = {old: new for new, old in enumerate(valid_vertices)}
    
    # Build adjacency in compact space
    compact_adj = [0] * M
    for i, old_i in enumerate(valid_vertices):
        v = old_i + 1
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                old_j = u - 1
                if old_j in old_to_new:
                    j = old_to_new[old_j]
                    if i != j:
                        compact_adj[i] |= (1 << j)

    # Sort vertices by degree for better B&B performance
    # We will permute the graph so that vertices are ordered by degree.
    # This helps if we use a simple "pick first available" or similar heuristics, 
    # but for dynamic degree selection, it's less critical. 
    # However, Tomita's original paper relies on a specific ordering.
    # Let's just use the compact graph and dynamic selection.

    def popcount(x):
        return bin(x).count('1')

    best_size = 0
    best_mask = 0

    def branch_and_bound(P, current_size):
        nonlocal best_size, best_mask
        
        if time.time() > DEADLINE:
            return

        if P == 0:
            if current_size > best_size:
                best_size = current_size
                # We need to reconstruct the set. 
                # Storing the mask is tricky in recursion without backtracking storage.
                # For now, we just track size. We can reconstruct if needed or store path.
                # Since we only return the size as the score, and the task asks for the set,
                # we should store the best set.
                # Let's pass a list or use a global stack.
                pass
            return

        # Upper bound
        if current_size + popcount(P) <= best_size:
            return

        # Select vertex v from P
        # Heuristic: Pick vertex with max degree in P
        # To find max degree in P efficiently:
        # We can iterate through bits in P.
        # M is at most 256.
        
        best_v = -1
        max_deg = -1
        temp_P = P
        while temp_P:
            # Get lowest set bit
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            # Degree of v in P: count bits in (compact_adj[v] & P)
            deg = popcount(compact_adj[v] & P)
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_P ^= lsb
        
        v = best_v
        bit_v = 1 << v
        
        # Branch 1: Include v
        # New P = P \ (N(v) U {v})
        new_P_include = P & ~(compact_adj[v] | bit_v)
        
        # To track the solution, we can store the best set of vertices.
        # Since we are in a recursive function, we can maintain a global `current_set` list.
        
        global current_set
        current_set.append(v)
        branch_and_bound(new_P_include, current_size + 1)
        current_set.pop()
        
        if time.time() > DEADLINE:
            return

        # Branch 2: Exclude v
        new_P_exclude = P & ~bit_v
        branch_and_bound(new_P_exclude, current_size)

    # We need to track the actual set to return it.
    # Let's modify the B&B to store the best set.
    
    best_set_mask = 0
    
    def bnb(P, current_size, current_mask):
        nonlocal best_size, best_set_mask
        
        if time.time() > DEADLINE:
            return

        if P == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return

        if current_size + popcount(P) <= best_size:
            return

        # Select vertex v from P with max degree in P
        best_v = -1
        max_deg = -1
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            deg = popcount(compact_adj[v] & P)
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_P ^= lsb
        
        v = best_v
        bit_v = 1 << v
        
        # Include v
        new_P_include = P & ~(compact_adj[v] | bit_v)
        bnb(new_P_include, current_size + 1, current_mask | bit_v)
        
        if time.time() > DEADLINE:
            return

        # Exclude v
        new_P_exclude = P & ~bit_v
        bnb(new_P_exclude, current_size, current_mask)

    # Initial P is all vertices
    initial_P = (1 << M) - 1
    bnb(initial_P, 0, 0)

    # Convert best_set_mask back to original values
    result = []
    for i in range(M):
        if (best_set_mask >> i) & 1:
            result.append(valid_vertices[i] + 1)
            
    if not result:
        # Fallback if B&B failed or timed out before finding anything (unlikely)
        # Use a simple greedy
        result = []
        mask = 0
        for i in range(M):
            if not (compact_adj[i] & mask):
                mask |= (1 << i)
                result.append(valid_vertices[i] + 1)

    return sorted(result)
