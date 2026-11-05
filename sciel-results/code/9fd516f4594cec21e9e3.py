import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges_total = len(all_edges)
    
    # Precompute all edge indices
    edge_set = set(all_edges)
    
    # Precompute popcount for all possible 20-bit values
    popcount_table = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    def popcount(x):
        return popcount_table[x]
    
    def count_edges(masks):
        return sum(popcount(m) for m in masks) // 2
    
    # Precompute all triples for faster violation checking
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def get_violation(masks):
        """Find one K3,3 violation or return None."""
        for i, j, k in triples:
            common = masks[i] & masks[j] & masks[k]
            if popcount(common) >= 3:
                return (i, j, k, common)
        return None
    
    def is_valid(masks):
        return get_violation(masks) is None
    
    def try_add_edge(masks, u, v):
        """Try to add edge (u,v), return True if valid."""
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        if is_valid(masks):
            return True
        else:
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            return False
    
    def try_remove_edge(masks, u, v):
        """Remove edge (u,v)."""
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
    
    def greedy_fill(masks):
        """Greedily add all possible edges that maintain validity."""
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (masks[i] & (1 << j)):
                        if try_add_edge(masks, i, j):
                            improved = True
        return masks
    
    def gq22_initialization():
        """
        Construct a K3,3-free graph using GQ(2,2) collinearity structure.
        GQ(2,2) has 15 points and 15 lines, each line has 3 points, each point on 3 lines.
        The collinearity graph (points connected if on same line) is K3,3-free.
        We extend to 20 vertices by adding 5 more vertices.
        """
        # GQ(2,2) structure: points as (x,y,z) in GF(3)^3 with x+y+z=0
        # Actually, let's use a known explicit construction
        # GQ(2,2) = W(2) = dual of GQ(2,2)
        # Points: 15 points, Lines: 15 lines, each line has 3 points
        
        # Explicit GQ(2,2) point set and line set
        # Using the standard representation
        points = list(range(15))
        
        # Lines of GQ(2,2) (each line is a set of 3 points)
        # This is the unique GQ(2,2)
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14],
            [0, 3, 6], [1, 4, 7], [2, 5, 8], [9, 12, 15 % 15], [10, 13, 11],
            [0, 4, 8], [1, 5, 6], [2, 3, 7], [9, 13, 11], [10, 12, 14]
        ]
        
        # Wait, that's not quite right. Let me use a proper GQ(2,2) construction.
        # GQ(2,2) can be constructed from the 3-dimensional vector space over GF(3)
        # Points are 1-dimensional subspaces (lines through origin) in GF(3)^3
        # There are (3^3 - 1)/(3-1) = 13 points... that's not 15.
        
        # Actually GQ(2,2) has s=2, t=2, so (s+1)(st+1) = 3*5 = 15 points
        # Let me use a different explicit construction
        
        # Use the standard GQ(2,2) = the collinearity graph of the icosahedron? No.
        # Let me just use a known K3,3-free graph on 15 vertices with many edges.
        
        # Actually, let's use a simpler approach: start with a known dense K3,3-free structure
        # and extend it.
        
        # Alternative: Use the incidence graph of projective plane of order 2 (Fano plane)
        # The Fano plane has 7 points and 7 lines. Its incidence graph is K3,3-free?
        # No, the incidence graph of the Fano plane is the Heawood graph, which IS K3,3-free
        # and has 14 vertices and 21 edges.
        
        # Let's use a combination approach:
        # Start with a structured base and add vertices strategically.
        
        masks = [0] * n
        
        # Base: Use a 10-vertex structure that is K3,3-free and dense
        # The complement of the Petersen graph is K3,3-free? Let's check.
        # Petersen graph: 10 vertices, 15 edges, 3-regular
        # Complement: 10 vertices, 25 edges, 6-regular
        # Is complement of Petersen K3,3-free? The Petersen graph is K3,3-free,
        # but its complement might not be.
        
        # Let's use a bipartite construction instead:
        # Split 20 vertices into two sets of 10
        # Use a sparse bipartite graph between them plus intra-edges
        
        # Strategy: Use a (10,10) bipartite graph that avoids K3,3
        # A bipartite graph with parts of size 10 is K3,3-free iff
        # no 3 vertices in one part share 3 common neighbors in the other.
        
        # Use a cyclic construction: connect i in left to i, i+1, i+2 mod 10 in right
        # This gives each left vertex degree 3, total 30 edges
        # Check: do any 3 left vertices share 3 common right neighbors?
        # Left vertices i, j, k share right neighbors that are in intersection of
        # {i,i+1,i+2}, {j,j+1,j+2}, {k,k+1,k+2} mod 10
        # For 3 consecutive: {0,1,2}, {1,2,3}, {2,3,4} -> intersection = {2}, size 1
        # For {0,1,3}: {0,1,2}, {1,2,3}, {3,4,5} -> intersection = empty
        # Seems K3,3-free in bipartite sense.
        
        for i in range(10):
            for offset in range(3):
                j = (i + offset) % 10
                masks[i] |= (1 << (j + 10))
                masks[j + 10] |= (1 << i)
        
        # Add intra-left edges carefully
        # Left vertices 0-9: add edges that don't create K3,3
        # Since we have bipartite structure, intra-left edges need care
        for i in range(10):
            for j in range(i+1, 10):
                # Check if adding (i,j) creates K3,3
                # i and j already have common neighbors in right part
                common_right = masks[i] & masks[j]
                if popcount(common_right) < 3:
                    # Safe to add
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        
        # Similarly for right vertices 10-19
        for i in range(10):
            for j in range(i+1, 10):
                vi = i + 10
                vj = j + 10
                common_left = masks[vi] & masks[vj]
                if popcount(common_left) < 3:
                    masks[vi] |= (1 << vj)
                    masks[vj] |= (1 << vi)
        
        return masks
    
    def structured_bipartite_init():
        """Alternative structured initialization."""
        masks = [0] * n
        
        # Use a different cyclic pattern
        # Connect left i to right i, i+1, i+3 mod 10
        for i in range(10):
            for offset in [0, 1, 3]:
                j = (i + offset) % 10
                masks[i] |= (1 << (j + 10))
                masks[j + 10] |= (1 << i)
        
        # Add some intra-edges
        for i in range(10):
            for j in range(i+1, 10):
                common_right = masks[i] & masks[j]
                if popcount(common_right) < 3:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        
        for i in range(10):
            for j in range(i+1, 10):
                vi = i + 10
                vj = j + 10
                common_left = masks[vi] & masks[vj]
                if popcount(common_left) < 3:
                    masks[vi] |= (1 << vj)
                    masks[vj] |= (1 << vi)
        
        return masks
    
    def random_dense_init(p):
        """Random dense initialization."""
        masks = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        return masks
    
    def random_sparse_init(p):
        """Random sparse initialization."""
        masks = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        return masks
    
    # Simulated annealing
    best_edges = 0
    best_masks = [0] * n
    
    # Multiple restarts with different strategies
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Choose initialization strategy
        strategy = restart % 5
        if strategy == 0:
            masks = gq22_initialization()
        elif strategy == 1:
            masks = structured_bipartite_init()
        elif strategy == 2:
            p = 0.70 + 0.05 * (restart % 3)
            masks = random_dense_init(p)
        elif strategy == 3:
            p = 0.35 + 0.05 * (restart % 3)
            masks = random_sparse_init(p)
        else:
            # Mixed: start with structured, then random perturb
            masks = gq22_initialization()
            # Remove some random edges
            for _ in range(10):
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j and (masks[i] & (1 << j)):
                    try_remove_edge(masks, i, j)
        
        # Repair: remove edges until valid
        max_repair = 500
        for _ in range(max_repair):
            if time.time() > deadline - 10:
                break
            viol = get_violation(masks)
            if viol is None:
                break
            i, j, k, common_mask = viol
            # Get 3 common neighbors
            neighbors = []
            m = common_mask
            while len(neighbors) < 3 and m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                neighbors.append(idx)
                m ^= lsb
            
            # Remove a random edge from the K3,3
            u = random.choice([i, j, k])
            v = random.choice(neighbors)
            try_remove_edge(masks, u, v)
        
        if not is_valid(masks):
            continue
        
        # Greedy fill
        greedy_fill(masks)
        
        current_edges = count_edges(masks)
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
        
        # Simulated annealing phase
        T = 1.5
        T_min = 0.005
        cooling_rate = 0.9997
        max_sa_steps = 100000
        
        # Precompute edge lists for faster SA moves
        for step in range(max_sa_steps):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            # Random move: either add an edge, remove an edge, or swap
            move_type = random.random()
            
            if move_type < 0.35:
                # Add a random missing edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not missing:
                    break
                u, v = random.choice(missing)
                old_state = (masks[u], masks[v])
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
                if is_valid(masks):
                    new_edges = count_edges(masks)
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        masks[u] = old_state[0]
                        masks[v] = old_state[1]
                else:
                    masks[u] = old_state[0]
                    masks[v] = old_state[1]
                    
            elif move_type < 0.60:
                # Remove a random existing edge
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                if not existing:
                    break
                u, v = random.choice(existing)
                old_state = (masks[u], masks[v])
                try_remove_edge(masks, u, v)
                
                new_edges = count_edges(masks)
                delta = new_edges - current_edges
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_masks = masks[:]
                else:
                    masks[u] = old_state[0]
                    masks[v] = old_state[1]
                    
            else:
                # Swap: remove one edge, add another
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not existing or not missing:
                    break
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                    
                old_state = (masks[u1], masks[v1], masks[u2], masks[v2])
                try_remove_edge(masks, u1, v1)
                masks[u2] |= (1 << v2)
                masks[v2] |= (1 << u2)
                
                if is_valid(masks):
                    new_edges = count_edges(masks)
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        masks[u1] = old_state[0]
                        masks[v1] = old_state[1]
                        masks[u2] = old_state[2]
                        masks[v2] = old_state[3]
                else:
                    masks[u1] = old_state[0]
                    masks[v1] = old_state[1]
                    masks[u2] = old_state[2]
                    masks[v2] = old_state[3]
            
            T *= cooling_rate
        
        # Final greedy fill
        greedy_fill(masks)
        current_edges = count_edges(masks)
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
    
    # Final polish: try more aggressive local search on best solution
    if best_edges > 0 and time.time() < deadline - 10:
        masks = best_masks[:]
        current_edges = best_edges
        
        # Multiple rounds of greedy fill with different orderings
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Try adding edges in different orders
            order = list(range(n))
            random.shuffle(order)
            for i_idx in range(n-1):
                for j_idx in range(i_idx+1, n):
                    i = order[i_idx]
                    j = order[j_idx]
                    if not (masks[i] & (1 << j)):
                        if try_add_edge(masks, i, j):
                            current_edges += 1
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
    
    # Convert best_masks to edge list
    edges = []
    for i in range(n):
        m = best_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
            
    return edges
