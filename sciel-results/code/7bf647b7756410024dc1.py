"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    deadline = t0 + 75.0  # safety margin before 90s kill
    
    n = 256
    # Build conflict bitsets: conflict[i] is a 256-bit int where bit j is set
    # iff (i+1)*(j+1) % ((i+1)+(j+1)) == 0 for i != j
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        bits = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                bits |= (1 << j)
        conflict[i] = bits
    
    # Initial solution: all odd numbers (indices where value is odd)
    # Odd numbers: 1,3,5,...,255 -> indices 0,2,4,...,254
    init_mask = 0
    for i in range(n):
        if (i + 1) % 2 == 1:  # odd value
            init_mask |= (1 << i)
    best_size = bin(init_mask).count('1')
    best_mask = init_mask
    
    # Greedy refinement: try to add elements one by one
    def greedy_from(mask, order):
        result = mask
        for i in order:
            if not (result >> i) & 1:
                if (conflict[i] & result) == 0:
                    result |= (1 << i)
        return result
    
    # Try greedy with different orderings to improve initial solution
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        # Random-ish ordering based on degree (fewer conflicts first)
        degrees = [bin(conflict[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: (degrees[i], -i))
        improved = greedy_from(0, order)
        if bin(improved).count('1') > best_size:
            best_size = bin(improved).count('1')
            best_mask = improved
    
    # Branch and bound for maximum independent set
    # State: available mask (candidates that can still be added)
    # We process vertices in order, deciding include/exclude
    
    # Order vertices by degree (ascending) for better pruning
    degrees = [bin(conflict[i]).count('1') for i in range(n)]
    vertex_order = sorted(range(n), key=lambda i: degrees[i])
    # Reverse: process high-degree vertices first for tighter bounds
    # Actually for MIS, we want to branch on vertices that constrain the most
    
    # Use a recursive B&B with bitsets
    # available: bitmask of vertices still available to add
    # current_mask: bitmask of vertices already selected
    # current_size: number selected so far
    
    global best_size, best_mask, deadline
    
    def bnb(available, current_mask, current_size):
        nonlocal best_size, best_mask, deadline
        
        if time.time() > deadline - 2:
            return
        
        # Upper bound: current_size + popcount(available)
        upper = current_size + bin(available).count('1')
        if upper <= best_size:
            return
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Pick a vertex to branch on: the one with highest degree in available
        # Find vertex with max conflicts within available
        avail_list = []
        tmp = available
        while tmp:
            lb = tmp & (-tmp)
            idx = lb.bit_length() - 1
            avail_list.append(idx)
            tmp ^= lb
        
        if not avail_list:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Pick vertex with max degree in available subgraph
        best_v = avail_list[0]
        best_deg = -1
        for v in avail_list:
            deg_in_avail = bin(conflict[v] & available).count('1')
            if deg_in_avail > best_deg:
                best_deg = deg_in_avail
                best_v = v
        
        # Branch 1: include best_v
        new_avail = available & ~(1 << best_v)
        new_avail &= ~conflict[best_v]
        bnb(new_avail, current_mask | (1 << best_v), current_size + 1)
        
        if time.time() > deadline - 2:
            return
        
        # Branch 2: exclude best_v
        new_avail = available & ~(1 << best_v)
        bnb(new_avail, current_mask, current_size)
    
    # Start B&B from scratch with all vertices available
    all_avail = (1 << n) - 1
    bnb(all_avail, 0, 0)
    
    # Final greedy polish: try to add any remaining non-conflicting elements
    result = best_mask
    for i in range(n):
        if time.time() > deadline - 1:
            break
        if not (result >> i) & 1:
            if (conflict[i] & result) == 0:
                result |= (1 << i)
    
    # Convert mask to list
    result_list = [i + 1 for i in range(n) if (result >> i) & 1]
    
    if not result_list:
        result_list = [1]
    
    return result_list
# EVOLVE-BLOCK-END
