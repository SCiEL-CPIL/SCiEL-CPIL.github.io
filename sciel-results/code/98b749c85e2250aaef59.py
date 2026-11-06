import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 512
    # Vertices 0..511 correspond to numbers 1..512
    
    # Precompute compatibility bitsets
    # compat[i] is a bitset of vertices j such that lcm(i+1, j+1) <= 512
    compat = [0] * N
    for i in range(N):
        num_i = i + 1
        bits = 0
        for j in range(N):
            if i == j:
                continue
            num_j = j + 1
            g = gcd(num_i, num_j)
            if (num_i * num_j) // g <= N:
                bits |= (1 << j)
        compat[i] = bits
        
    # Helper: get list of indices from bitset
    def bitset_to_list(bits):
        l = []
        while bits:
            lsb = bits & -bits
            idx = lsb.bit_length() - 1
            l.append(idx)
            bits ^= lsb
        return l

    # Helper: find an edge in bitset B
    # Returns (u, v) or None
    def find_edge_in_bitset(B):
        # Iterate through vertices in B
        temp = B
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp ^= lsb
            # Check neighbors of u in B
            neighbors = compat[u] & B
            # Remove u itself
            neighbors &= ~(1 << u)
            if neighbors:
                # Find one neighbor
                lsb2 = neighbors & -neighbors
                v = lsb2.bit_length() - 1
                return (u, v)
        return None

    # Helper: find a triangle in bitset B
    # Returns (u, v, w) or None
    def find_triangle_in_bitset(B):
        temp = B
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp ^= lsb
            # Neighbors of u in B
            neighbors = compat[u] & B
            neighbors &= ~(1 << u)
            # Find an edge in neighbors
            edge = find_edge_in_bitset(neighbors)
            if edge:
                v, w = edge
                return (u, v, w)
        return None

    # Greedy clique construction
    def greedy_clique(order):
        C = []
        candidates = (1 << N) - 1
        for v in order:
            if not (candidates & (1 << v)):
                continue
            # Check if v is compatible with all in C
            # Actually, if we maintain candidates as intersection of compat of C, 
            # then any v in candidates is compatible with all in C.
            # But we need to update candidates.
            # Standard greedy:
            # Start with all vertices.
            # Pick v, add to C, candidates = candidates & compat[v].
            # This ensures all future picks are compatible with v and previous picks.
            pass
            
        # Correct Greedy:
        C = []
        candidates = (1 << N) - 1
        for v in order:
            if not (candidates & (1 << v)):
                continue
            C.append(v)
            candidates &= compat[v]
            if not candidates:
                break
        return C

    # Local search: 1-for-2 and 2-for-3
    def local_search(C):
        C_set = set(C)
        C_bits = 0
        for v in C:
            C_bits |= (1 << v)
            
        improved = True
        while improved:
            improved = False
            # Try 1-for-2
            for u in C:
                if time.time() > deadline:
                    return C
                rest = [x for x in C if x != u]
                common = (1 << N) - 1
                for x in rest:
                    common &= compat[x]
                # Candidates are in common, not in C
                candidates = common & ~C_bits
                # Find an edge in candidates
                edge = find_edge_in_bitset(candidates)
                if edge:
                    v, w = edge
                    # Update C
                    C.remove(u)
                    C.append(v)
                    C.append(w)
                    C_set = set(C)
                    C_bits = 0
                    for x in C:
                        C_bits |= (1 << x)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try 2-for-3
            n = len(C)
            for i in range(n):
                for j in range(i+1, n):
                    if time.time() > deadline:
                        return C
                    u = C[i]
                    v = C[j]
                    rest = [x for x in C if x != u and x != v]
                    common = (1 << N) - 1
                    for x in rest:
                        common &= compat[x]
                    candidates = common & ~C_bits
                    # Find a triangle in candidates
                    tri = find_triangle_in_bitset(candidates)
                    if tri:
                        a, b, c = tri
                        # Update C
                        C.remove(u)
                        C.remove(v)
                        C.append(a)
                        C.append(b)
                        C.append(c)
                        C_set = set(C)
                        C_bits = 0
                        for x in C:
                            C_bits |= (1 << x)
                        improved = True
                        break
                if improved:
                    break
                    
        return C

    best_clique = []
    
    # Run multiple restarts with random orders
    # Also try a few specific orders (e.g., descending degree)
    
    # Calculate degrees
    degrees = [compat[i].bit_count() for i in range(N)]
    
    # Order 1: Descending degree
    order1 = sorted(range(N), key=lambda x: -degrees[x])
    C1 = greedy_clique(order1)
    C1 = local_search(C1)
    if len(C1) > len(best_clique):
        best_clique = C1
        
    # Order 2: Random
    for _ in range(20):
        if time.time() > deadline:
            break
        order2 = list(range(N))
        random.shuffle(order2)
        C2 = greedy_clique(order2)
        C2 = local_search(C2)
        if len(C2) > len(best_clique):
            best_clique = C2
            
    # Convert back to 1-based values
    return [v + 1 for v in best_clique]

# EVOLVE-BLOCK-END
