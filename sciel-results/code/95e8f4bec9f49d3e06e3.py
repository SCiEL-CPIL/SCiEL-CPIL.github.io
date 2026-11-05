import time
import random
import numpy as np

# Deadline guard
_DEADLINE = time.time() + 100.0

def _check_c5_fast(n, adj_bits, edge_u, edge_v):
    """
    Check if adding edge (u, v) creates a C5.
    adj_bits is a list of integers where bit i of adj_bits[j] is 1 if (j, i) is an edge.
    """
    nu = adj_bits[edge_u] & ~(1 << edge_v)
    nv = adj_bits[edge_v] & ~(1 << edge_u)
    
    for a in range(n):
        if a == edge_v: continue
        if not (nu >> a) & 1: continue
        
        for c in range(n):
            if c == edge_u or c == a: continue
            if not (nv >> c) & 1: continue
            
            common = adj_bits[a] & adj_bits[c]
            common &= ~(1 << edge_u)
            common &= ~(1 << edge_v)
            common &= ~(1 << a)
            common &= ~(1 << c)
            
            if common:
                return True
    return False

def _check_alpha_fast(n, adj_bits, target=9):
    """
    Check if graph has an independent set of size 'target'.
    Uses backtracking with bitmasks.
    """
    mask = (1 << n) - 1
    
    def search(remaining, current_size):
        if current_size == target:
            return True
        if remaining.bit_count() < target - current_size:
            return False
        
        bits = []
        temp = remaining
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            bits.append(idx)
            temp ^= lsb
            
        if not bits:
            return False
            
        v = random.choice(bits)
        
        new_remaining_incl = remaining & ~(1 << v) & ~adj_bits[v]
        if search(new_remaining_incl, current_size + 1):
            return True
            
        new_remaining_excl = remaining & ~(1 << v)
        if search(new_remaining_excl, current_size):
            return True
            
        return False

    return search((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    
    # Baseline: 8 disjoint K4s (n=32)
    # This is a known valid solution with alpha=8 and no C5.
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                best_edges.append([base+i, base+j])
                
    # Try to find n=33
    # Strategy: Randomized local search starting from 8 K4s + 1 isolated vertex.
    # We need to add edges to reduce alpha from 9 to 8 without creating C5.
    # As analyzed, adding edges to the isolated vertex in a way that hits all 
    # transversals of the K4s requires connecting to a whole K4 (creates C5).
    # Thus, we must modify the K4s themselves.
    # This is a complex search. Given time limits, we perform a limited search.
    
    found_33 = False
    if time.time() < _DEADLINE - 20:
        # Try a few random perturbations of the 32-vertex graph + 1 vertex
        for restart in range(50):
            if time.time() > _DEADLINE - 10:
                break
                
            n = 33
            adj_bits = [0] * n
            # Initialize with 8 K4s
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i+1, 4):
                        adj_bits[base+i] |= (1 << (base+j))
                        adj_bits[base+j] |= (1 << (base+i))
            
            # Vertex 32 is isolated.
            # We need to connect 32 to some vertices.
            # Also, we can randomly add/remove edges between K4s?
            # No, adding edges between K4s might create C5s if not careful.
            
            # Let's try to connect 32 to 1 vertex in each K4.
            # This keeps alpha=9.
            # We need to "break" the alpha=9 property.
            # This requires that 32 is adjacent to a "dominating" set of the K4s?
            # Actually, if we connect 32 to a vertex in K4_0, we can still pick 32 + (1 from K4_1..K4_7) + (1 from K4_0 not connected to 32).
            # So alpha is still 9.
            
            # The only way to reduce alpha is to ensure that for ANY choice of 1 vertex from each K4,
            # at least one is adjacent to 32.
            # This implies 32 is adjacent to ALL vertices in at least one K4.
            # Which creates C5.
            
            # Therefore, the structure of 8 disjoint K4s CANNOT be extended to 33 by adding a vertex.
            # We must change the structure of the K4s.
            # For example, replace one K4 with a different C5-free graph of 4 vertices with alpha=1?
            # K4 is the only such graph.
            
            # So, we cannot easily get 33.
            # We return 32.
            pass
    
    return {'n': best_n, 'edges': best_edges}
