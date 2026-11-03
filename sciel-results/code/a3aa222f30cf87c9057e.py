"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            # Ensure we don't generate the zero vector (not possible with m>0 and 1/-1, but good habit)
            # Also, to exploit symmetry, we can restrict to a canonical form.
            # However, for MIS on the full graph, we need all vectors.
            # But we can solve MIS on the "quotient" graph where u ~ -u.
            # Let's generate all, then build the graph.
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def _generate_seed_vectors():
    """Generate a small valid set as a fallback."""
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def construct():
    deadline = time.time() + 115  # Safety margin
    
    m = 4
    max_ip_allowed = 2
    
    candidates = _generate_shell(m)
    N = len(candidates)
    
    # Precompute the Gram matrix
    G = candidates @ candidates.T
    
    # Conflict graph: edge (i, j) exists if G[i, j] > max_ip_allowed
    # We want Maximum Independent Set (MIS) in this graph.
    # Equivalently, Maximum Clique in the complement graph.
    # Let's work with the conflict graph and find MIS.
    
    # Bitset representation for speed
    # N=5280. We can use Python integers as bitsets if N <= 1000 or so?
    # 5280 bits is 660 bytes. Python ints handle this fine.
    # Or use numpy boolean arrays? Python ints with bit operations are very fast in C.
    
    # Let's use Python integers for bitsets.
    # adj[i] is an integer where bit j is 1 if i and j conflict.
    
    adj = [0] * N
    for i in range(N):
        # Get conflicts for i
        # mask = G[i, :] > max_ip_allowed
        # We need to convert boolean array to int bitset
        # This is a bit slow to do in a loop.
        # Let's vectorize the creation of the adjacency matrix first.
        pass
        
    # Vectorized creation of adjacency mask
    conflict_mask = G > max_ip_allowed
    # Diagonal is True (G[i,i]=4 > 2). We don't care about self-loops for MIS logic usually, 
    # but it helps pruning if we treat self as conflict? 
    # Standard MIS: no self loops. 
    # Let's zero out diagonal.
    np.fill_diagonal(conflict_mask, False)
    
    # Convert to list of bitsets
    # np.packbits is one way, but we need big-endian or consistent ordering.
    # Simpler: just keep the boolean matrix and use numpy operations for propagation?
    # Or use the "Tomita" style algorithm with bitsets.
    
    # Let's try a recursive branch and bound with numpy boolean arrays for simplicity first,
    # but optimize the "most constrained" heuristic.
    
    # Actually, Python integer bitsets are much faster for set operations (AND, OR, NOT).
    # Let's convert conflict_mask to list of ints.
    # To convert a row of booleans to an int:
    # int.from_bytes(row.astype(np.uint8).tobytes(), 'big')?
    # Row length 5280. 5280/8 = 660 bytes.
    
    adj_bits = []
    # We need to be careful with bit ordering. 
    # Let's say bit 0 is index 0.
    # np.packbits packs 8 booleans into 1 byte, MSB first? 
    # np.packbits([1,0,0,0,0,0,0,0]) -> 0b10000000 = 128.
    # So index 0 is the most significant bit of the first byte.
    # This makes bit manipulation (shifting) annoying.
    
    # Alternative: Just use numpy boolean arrays and optimize the search loop.
    # Given N=5280, and time limit 115s.
    # A good MIS solver can handle this.
    
    # Let's implement a simple branch and bound MIS.
    # Heuristic: Pick vertex with max degree in current subgraph.
    
    # To make it fast, we'll use a recursive function.
    # State: 
    #   P: set of candidates (boolean array or list of indices)
    #   X: set of excluded vertices (not needed for standard MIS, just P shrinks)
    #   S: current solution set
    
    # We'll use a list of active indices for P.
    # And maintain a subgraph induced by P.
    
    # Optimization: Use the "P" set as a list of indices.
    # When we pick a vertex v:
    #   Branch 1: Include v. New P = P \ {v} \ N(v)
    #   Branch 2: Exclude v. New P = P \ {v}
    
    # To speed up, we can precompute neighbors as lists or bitsets.
    # Let's use bitsets (Python ints) for the adjacency.
    
    # Converting numpy bool matrix to list of Python ints:
    # We can do this row by row.
    # For each row, we want an int where bit i is 1 if conflict.
    # Let's use a helper to convert a boolean array to an int.
    
    def bool_to_int(b):
        # b is np array of bool
        # We want index 0 to be bit 0 (LSB) for easy masking?
        # Or index 0 to be MSB?
        # Let's stick to standard: index i corresponds to bit i.
        # We can iterate? No, too slow.
        # Use np.packbits with bitorder='little'?
        # np.packbits(b, bitorder='little') -> 8 bits per byte, LSB first.
        # Then we can just concatenate bytes?
        # int.from_bytes(data, 'little')
        # This maps index 0 to bit 0, index 1 to bit 1, etc.
        # Perfect.
        packed = np.packbits(b, bitorder='little')
        return int.from_bytes(packed, 'little')
        
    adj_ints = [bool_to_int(conflict_mask[i]) for i in range(N)]
    
    # Also precompute the full mask of all vertices
    full_mask = (1 << N) - 1
    
    # We need a way to get the degree of a vertex in the current subgraph P.
    # degree(v, P) = popcount(adj[v] & P)
    
    # We need a fast popcount. Python 3.10+ has int.bit_count().
    
    best_size = 0
    best_set = []
    
    # To avoid deep recursion limits, we might need to be careful, 
    # but MIS on 5000 nodes with good pruning shouldn't go too deep.
    
    # We also want to use a lower bound to prune.
    # Greedy lower bound on P:
    # Pick a vertex, remove it and its neighbors, repeat.
    
    def greedy_lb(P_mask):
        # P_mask is an int
        # We need to iterate through vertices in P.
        # This is O(N) per call if we scan all bits.
        # We can optimize by just returning a simple bound like log2(|P|) or something?
        # Or a fast greedy:
        # Pick a random vertex? No, deterministic is better for pruning.
        # Pick vertex with max degree?
        
        # Simple bound: 
        # If P is empty, 0.
        # Else, at least 1.
        
        # Better: 
        # We can maintain a global best and prune if current_size + greedy_lb(P) <= best_size.
        
        # Let's implement a fast greedy:
        # 1. Get list of indices in P? That's expensive.
        # 2. Just pick the lowest set bit?
        
        # Let's use a very fast bound: 
        # The size of the maximum independent set is at least the size of the largest 
        # "clique" in the complement? No.
        
        # Let's just use a simple heuristic: 
        # While P is not empty:
        #   Pick v with max degree in P?
        #   This requires scanning P.
        
        # For now, let's use a weak bound to get the code working, then optimize.
        # Weak bound: 1 if P != 0 else 0.
        # This will be very slow.
        
        # Let's try a slightly better one:
        # Count bits in P. If count <= best_size - current_size, prune.
        # This is just popcount(P).
        
        # Let's implement a proper greedy bound.
        # We'll pick vertices in order of index for simplicity first.
        count = 0
        p = P_mask
        while p:
            # Get lowest set bit
            lsb = p & -p
            idx = lsb.bit_length() - 1
            # Remove idx and its neighbors
            p = p & ~(adj_ints[idx] | lsb)
            count += 1
        return count

    # Recursive MIS
    import sys
    sys.setrecursionlimit(10000)
    
    def mis_rec(P_mask, current_size, current_set):
        nonlocal best_size, best_set
        
        if P_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
            return
        
        # Pruning
        # Upper bound on additional vertices: popcount(P_mask)
        # If current_size + popcount(P_mask) <= best_size: return
        # This is a strong bound.
        if current_size + P_mask.bit_count() <= best_size:
            return
            
        # Heuristic: Pick vertex with max degree in P
        # To find max degree, we need to iterate over set bits in P.
        # This is O(N) per level.
        
        # Optimization: If the number of candidates is small, we can do it.
        # For large N, we might pick a random vertex or the first one.
        # Let's try to pick the one with max degree.
        
        max_deg = -1
        best_v = -1
        
        p = P_mask
        while p:
            lsb = p & -p
            idx = lsb.bit_length() - 1
            deg = (adj_ints[idx] & P_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = idx
            p = p & (p - 1) # Clear lowest bit
            
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        # New P = P \ {best_v} \ N(best_v)
        new_P = P_mask & ~(adj_ints[best_v] | (1 << best_v))
        current_set.append(best_v)
        mis_rec(new_P, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        # New P = P \ {best_v}
        new_P = P_mask & ~(1 << best_v)
        mis_rec(new_P, current_size, current_set)
        
    # Initial P is all vertices
    initial_P = full_mask
    # Note: full_mask = (1<<N)-1.
    
    # Start search
    mis_rec(initial_P, 0, [])
    
    # Check time
    if time.time() > deadline - 5:
         # Return best found so far
         if best_set:
             return candidates[best_set]
         else:
             return _generate_seed_vectors()
    
    # The recursion above might be too slow if it doesn't prune well enough.
    # The "max degree" calculation is O(N) per node in the recursion tree.
    # With N=5280, this is expensive.
    
    # Let's see if we can improve the bound.
    # The bound `current_size + P_mask.bit_count() <= best_size` is good.
    # But finding the max degree vertex is the bottleneck.
    
    # Alternative: Use a random vertex or a fixed order?
    # Fixed order (lowest index) is fast to find (lsb).
    # Let's try a hybrid: 
    # If N is large, pick lsb. 
    # If N is small, pick max degree.
    
    # Actually, let's restart the search with a better heuristic implementation if the first one is slow.
    # But we are already inside the function.
    
    # Let's assume the first run finishes or times out.
    # If it times out, we return the best found.
    
    if best_set:
        return candidates[best_set]
    else:
        return _generate_seed_vectors()

# EVOLVE-BLOCK-END
