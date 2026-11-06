import time
import random

def construct():
    N = 256
    nodes = list(range(1, N + 1))
    
    # Precompute conflict masks
    # masks[i] is a bitmask where bit j is set if i conflicts with j
    masks = [0] * (N + 1)
    degrees = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # Check if (i*j) % (i+j) == 0
            if (i * j) % (i + j) == 0:
                # Set bits
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
                degrees[i] += 1
                degrees[j] += 1
                
    # Bitmask for each node itself (for checking if node is in forbidden set)
    # Actually, we just check if bit i is in forbidden.
    
    best_set = []
    best_size = 0
    
    # Deadline guard
    deadline = time.time() + 55.0  # 5s margin
    
    # Helper to evaluate a greedy pass
    def run_greedy(order):
        forbidden = 0
        selected = []
        for node in order:
            # Check if node is forbidden (conflicts with any selected)
            # bit node is 1 << node
            if (forbidden >> node) & 1:
                continue
            # Select node
            selected.append(node)
            # Update forbidden mask with neighbors of node
            forbidden |= masks[node]
        return selected

    # Pre-computed orders
    # 1. Random
    # 2. Degree ascending
    # 3. Degree descending
    # 4. Randomized degree
    
    # We will generate random orders on the fly for most iterations
    # But we can also use fixed good orders
    
    # Initial run with degree ascending
    order_deg_asc = sorted(nodes, key=lambda x: degrees[x])
    s = run_greedy(order_deg_asc)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s.copy()
        
    order_deg_desc = sorted(nodes, key=lambda x: -degrees[x])
    s = run_greedy(order_deg_desc)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s.copy()
        
    # Iterated Greedy with Randomization
    # Use random.shuffle on a copy of nodes
    
    # To speed up, we can also try "Local Search" on the best set found
    # But let's maximize iterations of greedy first.
    
    while time.time() < deadline:
        # Strategy 1: Pure Random
        order = nodes.copy()
        random.shuffle(order)
        s = run_greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s.copy()
            
        # Strategy 2: Randomized Degree
        # Add small random noise to degree
        order = sorted(nodes, key=lambda x: degrees[x] + random.random())
        s = run_greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s.copy()
            
        # Strategy 3: Random Subset of High Degree first? No, usually low degree first.
        # Let's try sorting by degree, but shuffling within same degree buckets
        # This is effectively done by noise.
        
        # Occasionally try a local search kick on the best set?
        # For now, rely on the speed of bitmask greedy.
        
    return best_set
