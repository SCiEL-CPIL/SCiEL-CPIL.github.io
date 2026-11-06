import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115
    
    n = 16
    vertices = list(range(n))
    
    # Pre-compute all 8-subsets
    all_subsets = list(combinations(vertices, 8))
    num_subsets = len(all_subsets)
    
    # Pre-compute canonical Q3 masks on 8 vertices (0..7)
    # Edge index in K8: map (u,v) with u<v to 0..27
    # We can use a simple mapping: 
    # 0: (0,1), 1: (0,2), ..., 27: (6,7)
    # Or just generate all 8! permutations of 0..7, construct edge sets, store as frozenset of tuples or bitmask.
    
    # Bitmask for K8 edges:
    # Index for edge (u, v) where u < v:
    # We can define a function to get the bit index.
    # Or just pre-compute the 840 masks.
    
    def edge_bit_index(u, v):
        if u > v: u, v = v, u
        # 8 vertices, 28 edges
        # Number of edges before u: u*7 - u(u+1)/2 ?
        # Let's just use a lookup or formula.
        # Simpler: iterate 0..27 and check? No, pre-compute.
        # For canonical generation, we just need to generate the masks.
        pass

    # Generate canonical Q3 masks
    # Q3 edges on 0..7:
    # 0-1, 0-2, 0-4, 1-3, 2-3, 4-5, 4-6, 5-7, 6-7, 1-5? No.
    # Standard Q3 embedding:
    # 0: 000, 1: 001, 2: 010, 3: 011, 4: 100, 5: 101, 6: 110, 7: 111
    # Edges:
    # (0,1), (0,2), (0,4)
    # (1,3), (1,5)
    # (2,3), (2,6)
    # (4,5), (4,6)
    # (3,7), (5,7), (6,7)
    
    q3_edges_canonical = [
        (0,1), (0,2), (0,4),
        (1,3), (1,5),
        (2,3), (2,6),
        (4,5), (4,6),
        (3,7), (5,7), (6,7)
    ]
    
    # Generate all 840 unique masks
    canonical_masks = set()
    # Permutations of 0..7
    # To avoid generating 40320 and deduping, we can just generate all and add to set.
    # 40320 is small.
    from itertools import permutations
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges_canonical:
            pu, pv = p[u], p[v]
            if pu > pv: pu, pv = pv, pu
            # Calculate bit index for (pu, pv) in K8
            # We need a consistent mapping.
            # Let's define bit_index(u, v) for u<v in 0..7
            # Total edges 28.
            # index = 0
            # for i in range(u):
            #    index += 7 - i  # edges from i to i+1..7
            # index += v - u - 1 # edges from u to u+1..v-1
            # This is O(1) if precomputed or simple formula.
            # Formula: index(u,v) = u*7 - u(u+1)//2 + v - u - 1
            # Let's verify:
            # u=0, v=1: 0 + 0 = 0. Correct.
            # u=0, v=2: 0 + 1 = 1. Correct.
            # u=1, v=2: 1*7 - 1 + 0 = 6.
            # Edges: (0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3, (0,5)=4, (0,6)=5, (0,7)=6, (1,2)=7?
            # My formula gave 6 for (1,2).
            # Let's re-evaluate.
            # Edges starting with 0: (0,1)..(0,7) -> 7 edges. Indices 0..6.
            # Edges starting with 1: (1,2)..(1,7) -> 6 edges. Indices 7..12.
            # So index(1,2) should be 7.
            # Formula: u*7 - u(u+1)//2 + v - u - 1
            # u=1: 7 - 1 + 2 - 1 - 1 = 6. Incorrect.
            
            # Correct formula for edge (u,v) with u<v in K_n:
            # Number of edges with first vertex < u: sum_{i=0}^{u-1} (n - 1 - i)
            # For n=8: sum_{i=0}^{u-1} (7 - i)
            # u=0: 0
            # u=1: 7
            # u=2: 7+6=13
            # u=3: 13+5=18
            # u=4: 18+4=22
            # u=5: 22+3=25
            # u=6: 25+2=27
            # u=7: 27+1=28 (out of bounds)
            
            # Offset for u:
            # offsets = [0, 7, 13, 18, 22, 25, 27, 28]
            # index = offsets[u] + (v - u - 1)
            
            # Let's use this.
            offsets = [0, 7, 13, 18, 22, 25, 27, 28]
            idx = offsets[pu] + (pv - pu - 1)
            mask |= (1 << idx)
        canonical_masks.add(mask)
        
    canonical_masks = list(canonical_masks)
    # Should be 840
    
    def get_local_mask(adj, subset):
        # subset is a tuple of 8 vertices
        # Map subset vertices to 0..7
        # We need the adjacency between them.
        # Create a 8x8 adjacency matrix or just the 28-bit mask.
        local_mask = 0
        # subset[i] is the global vertex corresponding to local i
        for i in range(8):
            ui = subset[i]
            for j in range(i+1, 8):
                uj = subset[j]
                if (adj[ui] >> uj) & 1:
                    offsets = [0, 7, 13, 18, 22, 25, 27, 28]
                    idx = offsets[i] + (j - i - 1)
                    local_mask |= (1 << idx)
        return local_mask

    def find_q3(adj):
        # Shuffle subset indices
        indices = list(range(num_subsets))
        random.shuffle(indices)
        for idx in indices:
            subset = all_subsets[idx]
            local_mask = get_local_mask(adj, subset)
            # Check if local_mask contains any canonical mask
            for cm in canonical_masks:
                if (local_mask & cm) == cm:
                    return subset, cm
        return None

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    best_edges = 0
    best_adj = None
    
    for restart in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(42 + restart)
        
        # Start with K16
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        # Delete edges until Q3-free
        while True:
            if time.time() - start_time > time_limit:
                break
            res = find_q3(adj)
            if res is None:
                break
            subset, cm = res
            # Pick a random edge from the Q3 to remove
            # We need to map the bits in cm back to global edges in subset
            # cm has 12 bits set.
            # We can iterate through the 28 possible local edges, check if bit is set in cm, 
            # and if so, pick one.
            edges_to_remove = []
            offsets = [0, 7, 13, 18, 22, 25, 27, 28]
            for i in range(8):
                for j in range(i+1, 8):
                    idx = offsets[i] + (j - i - 1)
                    if (cm >> idx) & 1:
                        ui = subset[i]
                        uj = subset[j]
                        edges_to_remove.append((ui, uj))
            
            # Remove one random edge
            u, v = random.choice(edges_to_remove)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        # Local Search: Try to add edges
        if time.time() - start_time > time_limit:
            break
            
        current_edges = count_edges(adj)
        
        for _ in range(500): # Try 500 additions
            if time.time() - start_time > time_limit:
                break
            u = random.randint(0, 15)
            v = random.randint(0, 15)
            if u == v:
                continue
            if not ((adj[u] >> v) & 1):
                # Try adding
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check if Q3-free
                if find_q3(adj) is None:
                    current_edges += 1
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
