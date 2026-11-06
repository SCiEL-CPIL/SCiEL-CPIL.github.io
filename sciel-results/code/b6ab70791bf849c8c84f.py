import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe within 60s
    
    # 1. Determine valid candidates (diagonal check)
    # 2a must not be a perfect square
    squares = set(k*k for k in range(1, 23)) # max sum 512, sqrt(512) ~ 22.6
    
    candidates = []
    for i in range(1, 257):
        if 2 * i not in squares:
            candidates.append(i)
            
    N = len(candidates)
    
    # 2. Build conflict masks
    # mask[i] is a bitset where bit j is 1 if candidates[i] + candidates[j] is a square
    masks = [0] * N
    for i in range(N):
        ci = candidates[i]
        m = 0
        for j in range(N):
            if i == j:
                continue
            cj = candidates[j]
            if ci + cj in squares:
                m |= (1 << j)
        masks[i] = m
    
    # Precompute degree for each node
    degrees = [masks[i].bit_count() for i in range(N)]
    
    # 3. Greedy initialization with min-degree heuristic
    def greedy_solve():
        remaining = (1 << N) - 1
        selected = 0
        while remaining:
            # Pick node with min degree in remaining
            best_node = -1
            min_deg = float('inf')
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (masks[idx] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                r ^= lsb
                
            if best_node == -1:
                break
                
            selected |= (1 << best_node)
            # Remove best_node and its neighbors
            remaining &= ~(masks[best_node] | (1 << best_node))
            
        return selected, selected.bit_count()

    best_set, best_size = greedy_solve()
    
    # 4. Local Search with better move selection
    def local_search(current_set, current_size):
        nonlocal best_set, best_size
        
        current = current_set
        current_sz = current_size
        best_local = current
        best_local_sz = current_size
        
        # Phase 1: Try to add any valid node (greedy addition)
        while True:
            if time.time() > deadline:
                break
            
            added = False
            # Find all nodes that can be added
            addable = []
            for i in range(N):
                bit = 1 << i
                if not (current & bit) and (masks[i] & current) == 0:
                    addable.append(i)
            
            if not addable:
                break
            
            # Add the node with minimum degree (to preserve future options)
            best_add = min(addable, key=lambda i: (masks[i] & ~current).bit_count())
            current |= (1 << best_add)
            current_sz += 1
            added = True
            
            if current_sz > best_local_sz:
                best_local = current
                best_local_sz = current_sz
            if current_sz > best_size:
                best_size = current_sz
                best_set = current
        
        # Phase 2: Try swaps to improve
        max_iterations = 500
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to find a beneficial swap
            found_swap = False
            for _ in range(20):  # Try a few random swaps
                # Pick a node not in current
                u_idx = random.randint(0, N - 1)
                u_bit = 1 << u_idx
                if current & u_bit:
                    continue
                
                # Check if u can be added after removing some v
                # Find all v in current such that (masks[u] & (current & ~v_bit)) == 0
                r = current
                while r:
                    lsb = r & (-r)
                    v_idx = lsb.bit_length() - 1
                    v_bit = 1 << v_idx
                    
                    if (masks[u_idx] & (current & ~v_bit)) == 0:
                        # Valid swap
                        new_set = (current & ~v_bit) | u_bit
                        # Accept swap to explore
                        current = new_set
                        found_swap = True
                        break
                    r ^= lsb
                
                if found_swap:
                    break
            
            if not found_swap:
                # Try double swap: remove two nodes, add one
                found_double = False
                for _ in range(10):
                    u_idx = random.randint(0, N - 1)
                    u_bit = 1 << u_idx
                    if current & u_bit:
                        continue
                    
                    # Try removing two nodes
                    r1 = current
                    while r1 and not found_double:
                        lsb1 = r1 & (-r1)
                        v1_idx = lsb1.bit_length() - 1
                        v1_bit = 1 << v1_idx
                        
                        r2 = current & ~v1_bit
                        while r2 and not found_double:
                            lsb2 = r2 & (-r2)
                            v2_idx = lsb2.bit_length() - 1
                            v2_bit = 1 << v2_idx
                            
                            new_current = current & ~v1_bit & ~v2_bit
                            if (masks[u_idx] & new_current) == 0:
                                # Valid double swap (net -1 size, might enable future additions)
                                # Only accept if it allows adding at least 2 more nodes later
                                # For now, accept with low probability
                                if random.random() < 0.3:
                                    current = new_current | u_bit
                                    current_sz -= 1
                                    found_double = True
                            r2 ^= lsb2
                        r1 ^= lsb1
                
                if not found_double:
                    break
        
        # Phase 3: Final greedy addition after swaps
        while True:
            if time.time() > deadline:
                break
            
            addable = []
            for i in range(N):
                bit = 1 << i
                if not (current & bit) and (masks[i] & current) == 0:
                    addable.append(i)
            
            if not addable:
                break
            
            best_add = min(addable, key=lambda i: (masks[i] & ~current).bit_count())
            current |= (1 << best_add)
            current_sz += 1
            
            if current_sz > best_local_sz:
                best_local = current
                best_local_sz = current_sz
            if current_sz > best_size:
                best_size = current_sz
                best_set = current
        
        return best_local, best_local_sz

    # Run local search
    best_set, best_size = local_search(best_set, best_size)
    
    # 5. Branch and Bound with better pruning
    if time.time() < deadline - 3.0:
        import sys
        sys.setrecursionlimit(10000)
        
        def compute_upper_bound(remaining):
            """Compute an upper bound on the maximum independent set size in remaining."""
            if remaining == 0:
                return 0
            
            # Simple upper bound: number of nodes
            count = remaining.bit_count()
            
            # Better bound: use greedy coloring on the conflict graph induced by remaining
            # For simplicity, use a simple bound: count / (1 + avg_degree)
            # But let's use a simpler approach: just return count for now
            # A tighter bound would require more computation
            
            return count
        
        def branch_and_bound(remaining, current_size, current_set, depth=0):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return
            
            # Pruning
            if current_size + remaining.bit_count() <= best_size:
                return
            
            if remaining == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                return
            
            # Heuristic: Pick a node to branch on - use max degree in remaining
            best_node = -1
            max_deg = -1
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (masks[idx] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_node = idx
                r ^= lsb
                
            if best_node == -1:
                return
            
            bit = 1 << best_node
            
            # Branch 1: Include best_node (usually more promising)
            new_remaining_1 = remaining & ~(masks[best_node] | bit)
            branch_and_bound(new_remaining_1, current_size + 1, current_set | bit, depth + 1)
            
            # Branch 2: Exclude best_node
            if time.time() > deadline:
                return
            new_remaining_2 = remaining & ~bit
            branch_and_bound(new_remaining_2, current_size, current_set, depth + 1)
        
        # Start branch and bound from the current best set as a starting point
        # This gives us a good initial lower bound
        initial_remaining = (1 << N) - 1
        branch_and_bound(initial_remaining, 0, 0)
    
    # 6. Final polish: try to add any remaining valid nodes
    if time.time() < deadline - 1.0:
        for i in range(N):
            bit = 1 << i
            if not (best_set & bit) and (masks[i] & best_set) == 0:
                best_set |= bit
                best_size += 1
    
    # Convert best_set bitset back to list of values
    result = []
    for i in range(N):
        if best_set & (1 << i):
            result.append(candidates[i])
            
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# To match the interface exactly, I should wrap it or just define it here.
# The prompt asks to keep function names. The current program has `construct`.
# I will place the code inside the block or define it globally.
# The seed program defines `construct` globally. I will do the same.
# EVOLVE-BLOCK-END
