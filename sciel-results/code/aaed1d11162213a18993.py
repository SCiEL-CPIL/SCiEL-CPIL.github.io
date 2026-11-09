import time
import numpy as np

N = 512
DEADLINE = 55  # 5 seconds margin for safety

def construct():
    t0 = time.time()
    
    # 1. Pre-compute all 4-APs and build the conflict graph
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # If a set contains all 4, it's invalid.
    # Conflict graph G=(V, E) where V=[1..N].
    # Edge (u, v) exists if u and v appear together in some 4-AP.
    # We want Maximum Independent Set in G.
    
    # Using bitmasks for speed.
    # N=512 fits in 8 uint64s.
    # Let's use a list of 8 integers, each holding 64 bits.
    # Or simpler: Python integers are arbitrary precision, but bit operations on 512 bits are fast.
    # Let's use a list of integers representing bitsets for neighbors.
    
    neighbors = [0] * (N + 1)
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            
            # Add edges between all pairs in this AP
            # (x1,x2), (x1,x3), (x1,x4), (x2,x3), (x2,x4), (x3,x4)
            pairs = [(x1, x2), (x1, x3), (x1, x4), (x2, x3), (x2, x4), (x3, x4)]
            for u, v in pairs:
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                
    # 2. Greedy Lower Bound Initialization
    # We need a good initial independent set to set the lower bound (LB) for B&B.
    # Strategy: Greedy by degree (pick vertex with min degree among available, add to set, remove neighbors).
    # Or simple greedy: iterate 1..N, add if no conflict with current set.
    # Let's do a few variations to get a strong LB.
    
    def greedy_init(order):
        in_set = [False] * (N + 1)
        size = 0
        # We need a fast way to check conflicts.
        # Since we are building the set, we can just check against the set so far.
        # But checking against the set so far is O(|set|) per candidate.
        # Better: maintain a "blocked" bitmask.
        blocked = 0
        for x in order:
            if not in_set[x]:
                # Check if x is blocked
                if not (blocked & (1 << x)):
                    in_set[x] = True
                    size += 1
                    # Add neighbors of x to blocked
                    blocked |= neighbors[x]
        return size, in_set
    
    # Try increasing order
    size1, set1 = greedy_init(range(1, N + 1))
    # Try decreasing order
    size2, set2 = greedy_init(range(N, 0, -1))
    # Try random orders (deterministic seed)
    import random
    rng = random.Random(42)
    best_lb = 0
    best_set = []
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)), # odds
        list(range(2, N + 1, 2)), # evens
    ]
    
    for _ in range(20):
        o = list(range(1, N + 1))
        rng.shuffle(o)
        orders.append(o)
        
    for o in orders:
        if time.time() - t0 > DEADLINE - 10:
            break
        s, st = greedy_init(o)
        if s > best_lb:
            best_lb = s
            best_set = st
            
    # 3. Branch and Bound for MIS
    # State: 
    #   candidates: bitmask of vertices that can still be added.
    #   current_set: bitmask of vertices in the current independent set.
    #   current_size: len(current_set)
    #   
    # Pruning:
    #   if current_size + popcount(candidates) <= best_lb: return
    #   if candidates is empty: update best_lb if current_size > best_lb
    
    # To speed up, we need efficient popcount. Python int.bit_count() is fast.
    
    # Variable Ordering:
    # Choose a vertex from candidates.
    # Heuristic: Minimum Remaining Values (MRV) is for CSP, here it's like "highest degree in candidate subgraph" or "lowest degree"?
    # In MIS, picking a high-degree vertex is often good because excluding it removes many options, but including it is risky.
    # However, standard B&B for MIS often picks the vertex with the highest degree in the remaining graph to prune more effectively if we branch on "include" vs "exclude".
    # Actually, let's branch:
    #   Branch 1: Include v. New candidates = candidates & ~neighbors[v] & ~(1<<v).
    #   Branch 2: Exclude v. New candidates = candidates & ~(1<<v).
    #   Order: Try Include first if it looks promising? Or Exclude?
    #   Usually, "Include" reduces the candidate set more, so it might lead to a smaller tree if the graph is dense.
    #   But we want a LARGE set.
    #   Let's try "Include" first.
    
    # To pick v: 
    #   Simple: pick the first set bit.
    #   Better: pick the vertex with the highest degree within the candidate subgraph.
    #   Calculating degree for all candidates is expensive.
    #   Approximation: pick vertex with highest degree in the original graph among candidates?
    #   Or just pick the vertex with the smallest index?
    #   Let's try picking the vertex with the highest degree in the original graph first, as it constrains the most.
    
    # Precompute degrees
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = neighbors[i].bit_count()
        
    # Sort vertices by degree descending for a static ordering if dynamic is too slow.
    # But dynamic MRV-like (highest degree in subgraph) is better.
    # Let's implement a simple dynamic picker:
    # Find v in candidates such that (neighbors[v] & candidates).bit_count() is max.
    
    def pick_vertex(candidates):
        # Iterate through set bits
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            # Get lowest set bit
            lsb = c & -c
            v = lsb.bit_length() - 1
            # Calculate degree of v in subgraph
            deg = (neighbors[v] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            c &= c - 1
        return best_v

    best_size_global = best_lb
    best_set_global = best_set.copy()
    
    # B&B Stack
    # Stack items: (candidates_mask, current_set_mask, current_size)
    # We start with all vertices as candidates?
    # No, we initialized with a greedy set.
    # We can start B&B from the state where the greedy set is already chosen?
    # It's complex to resume B&B from an arbitrary partial solution.
    # Easier: Start B&B from scratch with LB = best_lb.
    # The initial candidates is all bits 1..N set.
    
    initial_candidates = 0
    for i in range(1, N + 1):
        initial_candidates |= (1 << i)
        
    stack = [(initial_candidates, 0, 0)]
    
    while stack:
        if time.time() - t0 > DEADLINE:
            break
            
        candidates, current_set, current_size = stack.pop()
        
        # Prune 1: Upper bound
        if current_size + candidates.bit_count() <= best_size_global:
            continue
            
        if candidates == 0:
            if current_size > best_size_global:
                best_size_global = current_size
                # Extract set
                s = []
                m = current_set
                while m:
                    lsb = m & -m
                    s.append(lsb.bit_length() - 1)
                    m &= m - 1
                best_set_global = s
            continue
            
        # Pick vertex
        v = pick_vertex(candidates)
        
        # Branch 1: Include v
        # New candidates: remove v and its neighbors
        new_candidates_1 = candidates & ~(neighbors[v] | (1 << v))
        # We can prune this branch if current_size + 1 + new_candidates_1.bit_count() <= best_size_global
        if current_size + 1 + new_candidates_1.bit_count() > best_size_global:
            stack.append((new_candidates_1, current_set | (1 << v), current_size + 1))
            
        # Branch 2: Exclude v
        new_candidates_2 = candidates & ~(1 << v)
        # Prune this branch
        if current_size + new_candidates_2.bit_count() > best_size_global:
            stack.append((new_candidates_2, current_set, current_size))
            
    # Convert best_set_global to sorted list
    return sorted(best_set_global)

# Dummy call to ensure no syntax errors or immediate crashes in logic flow if run as script
if __name__ == "__main__":
    res = construct()
    print(len(res))
