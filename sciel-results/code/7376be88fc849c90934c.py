import math
import time

def construct():
    n_max = 256
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # 1. Precompute conflicts
    # self_conflict[i] is True if 2*i is a square
    self_conflict = [False] * (n_max + 1)
    for i in range(1, n_max + 1):
        if is_perfect_square(2 * i):
            self_conflict[i] = True

    # conflict_mask[i] has bits set for j such that i+j is a square
    # We exclude self-conflicts from the valid node list entirely
    conflict_mask = [0] * (n_max + 1)
    for i in range(1, n_max + 1):
        if self_conflict[i]:
            continue
        mask = 0
        for j in range(1, n_max + 1):
            if i != j and not self_conflict[j] and is_perfect_square(i + j):
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Valid nodes: 1..256 excluding self-conflicts
    valid_nodes = [i for i in range(1, n_max + 1) if not self_conflict[i]]
    
    # Order nodes by degree (number of conflicts) descending for better pruning
    # Higher degree nodes restrict more, so deciding them early prunes more
    degrees = {i: bin(conflict_mask[i]).count('1') for i in valid_nodes}
    ordered_nodes = sorted(valid_nodes, key=lambda x: degrees[x], reverse=True)
    
    # Map original node index to position in ordered list? 
    # Actually, it's easier to just work with the bitsets directly.
    # Let's re-map indices to 0..k-1 for cleaner bit operations if needed, 
    # but Python big ints handle 1<<256 fine.
    
    # We will use a recursive branch and bound function.
    # State: 
    #   candidates: bitmask of nodes that can still be added
    #   current_set: bitmask of nodes already in the solution
    #   current_size: size of current_set
    #   best_size: global best size found so far
    #   best_set: global best set found so far
    
    # To speed up, we can also precompute a "dominance" or use a simpler heuristic 
    # inside the branch and bound to order candidates.
    
    # Since the graph is static, we can use a standard B&B.
    # To avoid recursion depth issues and overhead, we can implement it iteratively or 
    # use a stack. However, for 256 nodes, recursion depth is bounded by the size of 
    # the independent set (max ~100?), which is safe for Python if we increase limit?
    # Actually, max independent set size is likely < 100. Recursion depth < 100 is fine.
    
    # However, pure Python recursion with bit operations might be slow.
    # We will use a time-limited search.
    
    start_time = time.time()
    time_limit = 50.0 # 60s limit, keep 10s buffer
    
    best_size = 0
    best_mask = 0
    
    # Initial lower bound: Greedy
    def greedy_initial():
        rem = set(valid_nodes)
        sol = 0
        while rem:
            # Pick node with min degree in remaining
            best_v = None
            min_deg = float('inf')
            rem_mask = 0
            for v in rem:
                rem_mask |= (1 << v)
            for v in rem:
                deg = bin(conflict_mask[v] & rem_mask).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            sol |= (1 << best_v)
            # Remove best_v and its neighbors
            neigh = conflict_mask[best_v] | (1 << best_v)
            rem = {v for v in rem if not (neigh & (1 << v))}
        return sol

    initial_sol = greedy_initial()
    best_size = bin(initial_sol).count('1')
    best_mask = initial_sol
    
    # Branch and Bound
    # We explore the search tree.
    # At each step, we pick a candidate node v.
    # Branch 1: Include v. New candidates = candidates & ~conflict_mask[v] & ~(1<<v).
    # Branch 2: Exclude v. New candidates = candidates & ~(1<<v).
    
    # To make it efficient, we should pick the node with highest degree in `candidates` 
    # to prune the "Include" branch more effectively? Or lowest?
    # Standard MIS B&B:
    # 1. If candidates is empty, update best.
    # 2. Upper bound: current_size + popcount(candidates). If <= best_size, prune.
    # 3. Select a vertex v from candidates.
    # 4. Recurse with v included.
    # 5. Recurse with v excluded.
    
    # We will use a stack for DFS to avoid recursion limits and allow easy time checking.
    # Stack items: (candidates_mask, current_size, current_mask)
    
    # Initial candidates: all valid nodes
    all_valid_mask = 0
    for v in valid_nodes:
        all_valid_mask |= (1 << v)
        
    # We start with empty set, all candidates available.
    # But to get a good initial solution, we already did greedy.
    # We can start B&B from the greedy solution? No, B&B explores from scratch 
    # but prunes based on best_size.
    
    # To improve performance, we can order the candidates dynamically.
    # A simple heuristic: always pick the node in `candidates` with the highest degree 
    # within `candidates`. This tends to reduce the branching factor.
    
    stack = [(all_valid_mask, 0, 0)]
    
    while stack:
        if time.time() - start_time > time_limit:
            break
            
        candidates, cur_size, cur_mask = stack.pop()
        
        if bin(candidates).count('1') == 0:
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
            continue
            
        # Pruning: upper bound
        # Max possible size = cur_size + number of candidates
        if cur_size + bin(candidates).count('1') <= best_size:
            continue
            
        # Select a vertex v from candidates
        # Heuristic: Pick v with highest degree in candidates
        # To do this efficiently, we iterate through bits in candidates.
        # Since n=256, iterating through all 256 bits is fast enough.
        
        best_v = -1
        max_deg = -1
        temp_cand = candidates
        while temp_cand:
            lsb = temp_cand & -temp_cand
            v = lsb.bit_length() - 1
            # Degree of v in candidates
            deg = bin(conflict_mask[v] & candidates).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_cand ^= lsb
            
        if best_v == -1:
            continue
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New candidates: remove best_v and its neighbors from candidates
        new_cand_inc = candidates & ~(conflict_mask[best_v] | v_bit)
        # We push this to stack. 
        # Note: We don't know the exact size increase is 1, so cur_size + 1
        stack.append((new_cand_inc, cur_size + 1, cur_mask | v_bit))
        
        # Branch 2: Exclude best_v
        # New candidates: remove best_v from candidates
        new_cand_exc = candidates & ~v_bit
        stack.append((new_cand_exc, cur_size, cur_mask))
        
    # Convert best_mask to list
    result = []
    for i in range(1, n_max + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    # Final verification
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i, len(s)):
                if is_perfect_square(s[i] + s[j]):
                    return False
        return True

    if not is_valid(result) or len(result) == 0:
        # Fallback to greedy if something went wrong
        result = []
        for i in range(1, n_max + 1):
            if self_conflict[i]: continue
            ok = True
            for j in result:
                if is_perfect_square(i + j):
                    ok = False
                    break
            if ok:
                result.append(i)
                
    return sorted(result)
