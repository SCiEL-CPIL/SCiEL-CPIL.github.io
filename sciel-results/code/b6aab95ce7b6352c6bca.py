"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

N = 16
DEADLINE = 85  # safety margin before 90s kill
MASK_ALL = (1 << N) - 1

def _has_diameter2_bitset(adj_bits):
    """Check if graph has diameter exactly 2 using bitset representation.
    adj_bits[i] is a bitmask of neighbors of vertex i."""
    n = N
    has_dist2 = False
    for a in range(n):
        for b in range(a + 1, n):
            if adj_bits[a] & (1 << b):
                continue
            # Check if a and b share a common neighbor
            # Common neighbors: (adj[a] & adj[b]) minus a and b themselves
            common = adj_bits[a] & adj_bits[b]
            # a is not in adj[a], b is not in adj[b], so common can't contain a or b
            # But we need at least one common neighbor c != a, c != b
            # Since a is not adjacent to itself and b is not adjacent to itself,
            # and a is not in adj[a], b is not in adj[b]:
            # common = adj[a] & adj[b] gives vertices adjacent to both
            # We need common to be non-empty
            if common == 0:
                return False
            has_dist2 = True
    return has_dist2

def _all_edges_essential_bitset(adj_bits):
    """Check if every edge is essential using bitset representation."""
    n = N
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_bits[u] & (1 << v)):
                continue
            # Remove edge (u, v)
            mask_u = adj_bits[u] & ~(1 << v)
            mask_v = adj_bits[v] & ~(1 << u)
            
            # Check if diameter is still <= 2 after removal
            # Only need to check pairs that might be affected
            # Actually, need to check all pairs
            for a in range(n):
                new_adj_a = adj_bits[a]
                if a == u:
                    new_adj_a = mask_u
                elif a == v:
                    new_adj_a = mask_v
                
                for b in range(a + 1, n):
                    if new_adj_a & (1 << b):
                        continue
                    new_adj_b = adj_bits[b]
                    if b == u:
                        new_adj_b = mask_u
                    elif b == v:
                        new_adj_b = mask_v
                    
                    common = new_adj_a & new_adj_b
                    if common == 0:
                        # This edge was essential (removal breaks diameter-2)
                        break
                else:
                    continue
                break
            else:
                # All pairs still have distance <= 2, so this edge was NOT essential
                return False
    return True

def _validate_full_bitset(adj_bits):
    """Full validation: diameter exactly 2 and all edges essential."""
    if not _has_diameter2_bitset(adj_bits):
        return False
    if not _all_edges_essential_bitset(adj_bits):
        return False
    return True

def _edges_from_bitset(adj_bits):
    """Convert bitset adjacency to edge list."""
    n = N
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_bits[u] & (1 << v):
                edges.append([u, v])
    return edges

def _bitset_from_adj(adj):
    """Convert adjacency list of sets to bitset representation."""
    n = N
    bits = [0] * n
    for u in range(n):
        for v in adj[u]:
            bits[u] |= (1 << v)
    return bits

def construct():
    start = time.time()
    best_edges = None
    best_count = -1
    
    def try_expand(adj_bits):
        """Try to add edges while maintaining all-essential property using bitsets."""
        n = N
        adj_bits = list(adj_bits)
        
        # Verify current state is valid
        if not _validate_full_bitset(adj_bits):
            return None
        
        pairs = list(combinations(range(n), 2))
        improved = True
        while improved:
            improved = False
            for (u, v) in pairs:
                if time.time() > DEADLINE:
                    return adj_bits
                if adj_bits[u] & (1 << v):
                    continue
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                if _validate_full_bitset(adj_bits):
                    improved = True
                else:
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
        return adj_bits
    
    # Strategy: Start from K_{8,8} (the current best) and try to expand
    # K_{8,8}: vertices 0-7 connected to vertices 8-15
    adj_bits_k88 = [0] * N
    for u in range(8):
        for v in range(8, 16):
            adj_bits_k88[u] |= (1 << v)
            adj_bits_k88[v] |= (1 << u)
    
    adj_exp = try_expand(adj_bits_k88)
    if adj_exp is not None:
        edges = _edges_from_bitset(adj_exp)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Fallback: return K_{8,8} if nothing better found
    if best_edges is None:
        best_edges = _edges_from_bitset(adj_bits_k88)
    
    # Ensure types are correct
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    return result
# EVOLVE-BLOCK-END
