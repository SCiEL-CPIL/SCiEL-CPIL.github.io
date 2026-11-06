"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)

    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)
    n = len(valid_nodes)
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}

    # Build conflict bitsets
    conflict_bits = [0] * n
    for i, node in enumerate(valid_nodes):
        for s in squares:
            j = s - node
            if 1 <= j <= N and j != node and j in valid_set:
                j_idx = node_to_idx[j]
                conflict_bits[i] |= (1 << j_idx)

    # Greedy min-degree independent set as initial lower bound
    remaining = (1 << n) - 1
    selected_bits = 0

    while remaining:
        best_node = -1
        best_degree = n + 1
        for i in range(n):
            if (remaining >> i) & 1:
                deg = bin(conflict_bits[i] & remaining).count('1')
                if deg < best_degree:
                    best_degree = deg
                    best_node = i
        selected_bits |= (1 << best_node)
        remaining &= ~(1 << best_node)
        remaining &= ~conflict_bits[best_node]

    best_size = bin(selected_bits).count('1')
    best_solution_bits = selected_bits

    # Local search: greedy add
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        for i in range(n):
            if (best_solution_bits >> i) & 1:
                continue
            if conflict_bits[i] & best_solution_bits:
                continue
            best_solution_bits |= (1 << i)
            best_size += 1
            improved = True

    # Branch and bound with greedy upper bound
    # Use a recursive B&B that tries to prove optimality or find better solutions
    
    # Precompute degree for ordering
    degrees = [bin(conflict_bits[i]).count('1') for i in range(n)]
    
    # Sort vertices by degree descending for better branching
    order = sorted(range(n), key=lambda x: -degrees[x])
    order_rank = [0] * n
    for rank, idx in enumerate(order):
        order_rank[idx] = rank
    
    # B&B state
    best_found = best_size
    best_solution = best_solution_bits
    nodes_visited = [0]
    
    def greedy_upper_bound(candidates):
        """Compute a greedy independent set size as upper bound."""
        if candidates == 0:
            return 0
        count = 0
        rem = candidates
        while rem:
            # Pick vertex with minimum degree in remaining
            best_v = -1
            best_deg = n + 1
            # Iterate only over set bits
            temp = rem
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (conflict_bits[v] & rem).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            
            count += 1
            rem &= ~(1 << best_v)
            rem &= ~conflict_bits[best_v]
        return count

    def branch_and_bound(candidates, current_size, current_bits, depth):
        nonlocal best_found, best_solution
        
        if time.time() > deadline - 5.0:
            return
        
        nodes_visited[0] += 1
        
        if candidates == 0:
            if current_size > best_found:
                best_found = current_size
                best_solution = current_bits
            return
        
        # Upper bound: current_size + greedy IS of candidates
        # Optimization: If we have time, use greedy bound. 
        # Greedy bound is expensive. Let's use a simpler bound first if needed, 
        # but the prompt asks to keep the design. 
        # We will use the greedy bound but it is the bottleneck.
        # To speed up, we can use a tighter heuristic bound if available, 
        # but let's stick to the requested design and just optimize the loops.
        
        ub = current_size + greedy_upper_bound(candidates)
        if ub <= best_found:
            return
        
        # Pick vertex with minimum degree in candidates for branching
        best_v = -1
        best_deg = n + 1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (conflict_bits[v] & candidates).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_candidates = candidates & ~(1 << best_v) & ~conflict_bits[best_v]
        branch_and_bound(new_candidates, current_size + 1, current_bits | (1 << best_v), depth + 1)
        
        # Branch 2: exclude best_v
        branch_and_bound(candidates & ~(1 << best_v), current_size, current_bits, depth + 1)

    # Run B&B with time limit
    if time.time() < deadline - 10.0:
        # Start B&B from the complement of our current solution
        # Actually, let's run B&B to find the maximum independent set
        # We'll use the current solution as a lower bound
        
        # For efficiency, we only explore if we can potentially beat best_found
        # Start with all valid nodes as candidates
        all_candidates = (1 << n) - 1
        
        # Run limited B&B
        branch_and_bound(all_candidates, 0, 0, 0)
        
        if best_found > best_size:
            best_size = best_found
            best_solution_bits = best_solution

    # Extract result
    result = []
    for i in range(n):
        if (best_solution_bits >> i) & 1:
            result.append(valid_nodes[i])

    result.sort()

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to the initial greedy solution
        result = []
        for i in range(n):
            if (selected_bits >> i) & 1:
                result.append(valid_nodes[i])
        result.sort()

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
