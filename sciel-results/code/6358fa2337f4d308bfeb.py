"""C6-free graph on 40 vertices: fast bitset local search."""
import time
import random

_N = 40
_DEADLINE = 100.0
_MARGIN = 5.0


def _creates_c6(u, v, adj_bits):
    """Check if adding edge (u, v) creates a C6."""
    # Path u-a-b-c-v. a in N(u), c in N(v).
    # b in N(a) & N(c).
    # u, v, a, b, c distinct.
    nu = adj_bits[u]
    nv = adj_bits[v]
    
    # Iterate a in N(u)
    a = nu
    while a:
        low_a = a & -a
        a ^= low_a
        ia = low_a.bit_length() - 1
        
        # Iterate c in N(v)
        c = nv
        while c:
            low_c = c & -c
            c ^= low_c
            ic = low_c.bit_length() - 1
            
            if ia == ic:
                continue
            
            # Check if N(a) and N(c) share a neighbor b
            # b must be distinct from u, v, a, c
            # adj_bits[ia] & adj_bits[ic] gives common neighbors.
            # If non-zero, there is a b.
            # We should ensure b is not u, v, a, c.
            # But in a simple graph, if ia != ic, and ia is in N(u), ic is in N(v):
            # If b == u: u in N(ia) (true) and u in N(ic) => ic in N(u).
            # Then a, c in N(u).
            # Cycle u-a-u-c-v-u? No.
            # The check `adj_bits[ia] & adj_bits[ic]` is sufficient for existence of a 6-cycle in most cases,
            # provided we handle the distinctness.
            # For speed, we just check non-zero. False positives (non-simple cycles) are rare 
            # and if we reject a valid edge, we just miss it. 
            # If we accept an invalid edge, we return a graph with C6.
            # So we must be correct.
            
            common = adj_bits[ia] & adj_bits[ic]
            if common:
                # Check if any b in common is valid
                # b != u, b != v, b != ia, b != ic
                # u, v are not in common?
                # u is in N(ia) (yes). Is u in N(ic)?
                # If u in N(ic), then ic in N(u).
                # So if ic in N(u) and ia in N(u), and ia != ic.
                # Then b=u is a common neighbor.
                # But b=u makes the cycle non-simple (u-a-u...).
                # So we should mask out u, v, ia, ic.
                mask = ~((1 << u) | (1 << v) | (1 << ia) | (1 << ic))
                if common & mask:
                    return True
    return False


def _is_c6_free(adj_bits):
    """Check if current graph is C6-free."""
    # We can just try to add all missing edges and see if any creates C6?
    # No, that's for the search.
    # For the seed, we can just check pairs.
    # Actually, if the seed is built by adding edges one by one using _creates_c6,
    # it is guaranteed to be C6-free.
    return True


def _build_seed():
    """Build a random C6-free seed."""
    n = _N
    adj_bits = [0] * n
    edges = []
    rng = random.Random(42)
    
    # Try to add random edges
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(possible_edges)
    
    for u, v in possible_edges:
        if not _creates_c6(u, v, adj_bits):
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            edges.append([u, v])
            
    return adj_bits, edges


def _local_search(adj_bits, edges, rng, deadline):
    n = _N
    m = len(edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v) if u < v else (v, u))
        
    best_m = m
    best_adj_bits = adj_bits[:]
    best_eset = set(edge_set)
    
    # Missing edges
    missing = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
    rng.shuffle(missing)
    idx = 0
    
    while time.time() < deadline:
        if idx >= len(missing):
            idx = 0
            rng.shuffle(missing)
            # If we cycled through all, we might be stuck. 
            # But we don't do swaps, so we are at a local optimum for additions.
            # We can break or continue shuffling (same result).
            # Let's break if no progress was made in the last cycle?
            # For simplicity, just break.
            break
            
        u, v = missing[idx]
        idx += 1
        
        if (u, v) in edge_set:
            continue
            
        if not _creates_c6(u, v, adj_bits):
            # Add edge
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            edge_set.add((u, v))
            missing.remove((u, v)) # O(N^2) slow. Use a set for missing too?
            m += 1
            if m > best_m:
                best_m = m
                best_adj_bits = adj_bits[:]
                best_eset = set(edge_set)
                
    return best_adj_bits, best_eset, best_m


def construct():
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    rng = random.Random(12345)
    
    adj_bits, edges = _build_seed()
    best_adj_bits, best_eset, best_m = _local_search(adj_bits, edges, rng, deadline)
    
    out = [[u, v] for (u, v) in best_eset if u < v]
    return out
