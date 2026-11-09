"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Returns a subset of [1, 512] with lcm(a, b) <= 512 for all pairs,
    maximizing cardinality.
    
    Strategy:
    1. Build a compatibility graph where edge (a,b) exists iff lcm(a,b) <= 512.
    2. Use Branch and Bound maximum clique algorithm to find the largest clique.
    3. Use a deadline guard to ensure we return within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    n = len(nodes)
    
    # Build adjacency using bitmasks for speed
    # adj[i] is a bitmask of nodes compatible with node i
    adj = [0] * n
    for i in range(n):
        a = nodes[i]
        mask = 0
        for j in range(i + 1, n):
            b = nodes[j]
            if a * b // math.gcd(a, b) <= 512:
                mask |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = mask
    
    # Greedy coloring for upper bound in branch and bound
    # We'll use a simple greedy coloring on the candidate set
    
    best_clique = []
    best_size = 0
    
    # Initial greedy lower bound
    greedy_set = []
    candidates = list(range(n))
    while candidates:
        # Pick node with highest degree in remaining candidates
        best_node = None
        best_deg = -1
        for node in candidates:
            deg = bin(adj[node] & ((1 << n) - 1)).count('1')
            # Count only neighbors in candidates
            cand_mask = 0
            for c in candidates:
                cand_mask |= (1 << c)
            deg = bin(adj[node] & cand_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_node = node
        if best_node is None:
            break
        greedy_set.append(best_node)
        # Remove best_node and its non-neighbors
        new_candidates = []
        for c in candidates:
            if c == best_node:
                continue
            if (adj[best_node] >> c) & 1:
                new_candidates.append(c)
        candidates = new_candidates
    
    best_clique = greedy_set[:]
    best_size = len(best_clique)
    
    # Branch and Bound
    # We use a recursive approach with bitmasks
    
    def popcount(x):
        return bin(x).count('1')
    
    def greedy_color(candidates_mask):
        """Greedy coloring to get upper bound on clique size in candidates."""
        colors = []
        uncolored = candidates_mask
        color_idx = 0
        while uncolored:
            # Pick a node from uncolored
            # Use simple strategy: pick first set bit
            node = (uncolored & -uncolored).bit_length() - 1
            # Assign this color to node and all its non-neighbors in uncolored
            # Actually for upper bound, we want to color such that each color class is independent
            # Simpler: just count how many colors needed with greedy
            # Remove node from uncolored
            uncolored &= ~(1 << node)
            color_idx += 1
            # For a tighter bound, we could do proper greedy coloring
            # But for speed, let's use a simpler approach: just return popcount as bound
            # Actually, let's do proper greedy coloring
            break  # This simple version just returns 1, let's improve
        
        # Better greedy coloring:
        colors = {}
        uncolored = candidates_mask
        next_color = 0
        while uncolored:
            # Pick node with highest degree in uncolored subgraph
            best_node = None
            best_deg = -1
            temp = uncolored
            while temp:
                node = (temp & -temp).bit_length() - 1
                temp &= ~(1 << node)
                deg = popcount(adj[node] & uncolored)
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
            
            if best_node is None:
                break
            
            # Try to assign existing color
            assigned = False
            for c in range(next_color):
                # Check if best_node is adjacent to any node with color c
                # We need to track color classes
                pass
            
            # Simplified: just use popcount as upper bound
            # This is valid but weak
            uncolored &= ~(1 << best_node)
            next_color += 1
        
        return next_color
    
    # Even simpler and tighter: use the fact that in a clique, all nodes must be mutually adjacent
    # Upper bound = size of maximum independent set in complement, but that's hard
    # Let's use a different approach: sort candidates by degree and use that as bound
    
    def bnb(candidates_mask, current_clique, current_size):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_clique = current_clique[:]
            return
        
        # Upper bound: current_size + popcount(candidates_mask)
        # This is valid since we can add at most all remaining candidates
        ub = current_size + popcount(candidates_mask)
        if ub <= best_size:
            return
        
        # Pick a node to branch on
        # Use the node with highest degree in candidates for better pruning
        node = None
        best_deg = -1
        temp = candidates_mask
        while temp:
            n_node = (temp & -temp).bit_length() - 1
            temp &= ~(1 << n_node)
            deg = popcount(adj[n_node] & candidates_mask)
            if deg > best_deg:
                best_deg = deg
                node = n_node
        
        if node is None:
            return
        
        # Branch 1: include node
        new_clique = current_clique + [node]
        # New candidates: intersection of current candidates (minus node) with neighbors of node
        new_candidates = (candidates_mask & ~(1 << node)) & adj[node]
        bnb(new_candidates, new_clique, current_size + 1)
        
        if time.time() > deadline:
            return
        
        # Branch 2: exclude node
        new_candidates = candidates_mask & ~(1 << node)
        bnb(new_candidates, current_clique, current_size)
    
    # Start BnB
    all_mask = (1 << n) - 1
    bnb(all_mask, [], 0)
    
    if best_size == 0:
        # Fallback: return divisors of 360
        L = 360
        divisors = []
        for i in range(1, int(L**0.5) + 1):
            if L % i == 0:
                divisors.append(i)
                if i != L // i:
                    divisors.append(L // i)
        return list(set(divisors))
    
    return [nodes[i] for i in best_clique]
# EVOLVE-BLOCK-END
