import time
import random
import numpy as np

# Deadline guard - evaluator kills at 150s, use 135s as safety margin
_DEADLINE = time.time() + 135.0

def _check_c5_exists(n, adj_bits):
    """Check if graph contains a C5 (5-cycle) using bitmask approach."""
    for u in range(n):
        for v in range(u+1, n):
            if not (adj_bits[u] >> v) & 1:
                continue
            # u-v is an edge. Need path u-a-b-c-v with all distinct.
            nu = adj_bits[u] & ~(1 << v)
            nv = adj_bits[v] & ~(1 << u)
            for a in range(n):
                if a == u or a == v:
                    continue
                if not (nu >> a) & 1:
                    continue
                for c in range(n):
                    if c == u or c == v or c == a:
                        continue
                    if not (nv >> c) & 1:
                        continue
                    # Need b in N(a) ∩ N(c) \ {u, v, a, c}
                    common = adj_bits[a] & adj_bits[c]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    common &= ~(1 << a)
                    common &= ~(1 << c)
                    if common:
                        return True
    return False

def _has_independent_set(n, adj_bits, target=9):
    """Check if graph has an independent set of size target using backtracking."""
    mask = (1 << n) - 1
    
    def search(remaining, current_size):
        if current_size == target:
            return True
        if remaining.bit_count() < target - current_size:
            return False
        if not remaining:
            return False
        
        # Pick vertex with highest degree in remaining for better pruning
        best_v = -1
        best_deg = -1
        temp = remaining
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg_in_rem = (adj_bits[idx] & remaining).bit_count()
            if deg_in_rem > best_deg:
                best_deg = deg_in_rem
                best_v = idx
            temp ^= lsb
        
        v = best_v
        # Include v: new remaining excludes v and its neighbors
        new_rem_incl = remaining & ~(1 << v) & ~adj_bits[v]
        if search(new_rem_incl, current_size + 1):
            return True
        # Exclude v
        new_rem_excl = remaining & ~(1 << v)
        if search(new_rem_excl, current_size):
            return True
        return False
    
    return search((1 << n) - 1, 0)

def _try_paley_like(n):
    """
    Try a Paley-like construction on n vertices.
    For prime p ≡ 1 (mod 4), the Paley graph has edges between u,v 
    iff (v-u) is a quadratic residue mod p.
    We try to find a subset of Z_n with a connection set S that gives
    a C5-free graph with small independence number.
    """
    # For n=34, try various connection sets
    # A graph on Z_n where (u,v) is an edge iff (v-u) mod n in S (or (u-v) mod n in S)
    # S must be symmetric: if d in S then n-d in S
    # 0 not in S
    
    for n in [33, 34, 35]:
        if time.time() > _DEADLINE - 20:
            break
        
        # Try random symmetric subsets S of {1, 2, ..., n-1}
        # where if d in S then n-d in S
        # |S| should be around n/2 for reasonable density
        
        for trial in range(200):
            if time.time() > _DEADLINE - 10:
                break
            
            # Generate random symmetric set
            # Pairs: (1, n-1), (2, n-2), ..., (floor(n/2), ...)
            pairs = []
            for d in range(1, (n//2) + 1):
                if d != n - d:
                    pairs.append((d, n-d))
                else:
                    pairs.append((d,))
            
            # Randomly select pairs
            random.shuffle(pairs)
            # Select about 40-60% of pairs
            frac = 0.4 + 0.2 * random.random()
            num_pairs = int(len(pairs) * frac)
            S = set()
            for p in pairs[:num_pairs]:
                for d in p:
                    S.add(d)
            
            # Build adjacency
            adj_bits = [0] * n
            for u in range(n):
                for v in range(n):
                    if u == v:
                        continue
                    diff = (v - u) % n
                    if diff in S:
                        adj_bits[u] |= (1 << v)
            
            # Check no self-loops
            for u in range(n):
                adj_bits[u] &= ~(1 << u)
            
            # Quick check: no C5
            if _check_c5_exists(n, adj_bits):
                continue
            
            # Check alpha < 9
            if not _has_independent_set(n, adj_bits, 9):
                # Valid! Extract edges
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj_bits[u] >> v) & 1:
                            edges.append([u, v])
                return n, edges
    
    return None

def _try_modular_construction(n):
    """
    Try a modular arithmetic construction where edges are defined by
    difference sets. More structured than random Paley-like.
    """
    if time.time() > _DEADLINE - 15:
        return None
    
    # For n=34: try connection sets based on quadratic non-residues mod various primes
    # or other structured sets
    
    for n in [33, 34, 35]:
        if time.time() > _DEADLINE - 10:
            break
        
        # Try: S = {d : d mod p is a quadratic non-residue} for various p
        # Or try specific structured sets
        
        # For n=34, try S based on mod 17 (since 34 = 2*17)
        if n == 34:
            # Quadratic residues mod 17: {1, 2, 4, 8, 16, 15, 13, 9}
            # Non-residues: {3, 5, 6, 7, 10, 11, 12, 14}
            # Map to Z_34: use both d and 34-d
            qr_17 = set([1, 2, 4, 8, 16, 15, 13, 9])
            nonres_17 = set([3, 5, 6, 7, 10, 11, 12, 14])
            
            # Try S = {d in Z_34 : d mod 17 in nonres_17}
            S = set()
            for d in range(1, 34):
                if d % 17 in nonres_17:
                    S.add(d)
            
            adj_bits = [0] * n
            for u in range(n):
                for v in range(n):
                    if u == v:
                        continue
                    diff = (v - u) % n
                    if diff in S:
                        adj_bits[u] |= (1 << v)
            
            for u in range(n):
                adj_bits[u] &= ~(1 << u)
            
            if not _check_c5_exists(n, adj_bits):
                if not _has_independent_set(n, adj_bits, 9):
                    edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if (adj_bits[u] >> v) & 1:
                                edges.append([u, v])
                    return n, edges
        
        # Try other structured sets for n=33 (p=33, not prime, but try mod 11)
        if n == 33:
            # Quadratic residues mod 11: {1, 3, 4, 5, 9}
            # Non-residues: {2, 6, 7, 8, 10}
            nonres_11 = set([2, 6, 7, 8, 10])
            S = set()
            for d in range(1, 33):
                if d % 11 in nonres_11:
                    S.add(d)
            
            adj_bits = [0] * n
            for u in range(n):
                for v in range(n):
                    if u == v:
                        continue
                    diff = (v - u) % n
                    if diff in S:
                        adj_bits[u] |= (1 << v)
            
            for u in range(n):
                adj_bits[u] &= ~(1 << u)
            
            if not _check_c5_exists(n, adj_bits):
                if not _has_independent_set(n, adj_bits, 9):
                    edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if (adj_bits[u] >> v) & 1:
                                edges.append([u, v])
                    return n, edges
    
    return None

def _try_local_search_from_k4s(target_n):
    """
    Try to extend the 8 K4s construction to target_n vertices using local search.
    Start with 8 K4s (32 vertices) + (target_n - 32) isolated vertices.
    Add edges to reduce alpha below 9 while maintaining C5-free property.
    """
    if time.time() > _DEADLINE - 15:
        return None
    
    n = target_n
    base_n = 32
    extra = n - base_n
    
    # Base: 8 disjoint K4s
    adj_bits = [0] * n
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                adj_bits[base+i] |= (1 << (base+j))
                adj_bits[base+j] |= (1 << (base+i))
    
    # Vertices 32..n-1 are isolated
    
    # We need to add edges involving the extra vertices (and possibly between K4s)
    # such that alpha < 9 and no C5 is created.
    
    # Strategy: For each extra vertex, connect it to a subset of K4 vertices.
    # The constraint is that we can't connect to all 4 of any K4 (creates K5 -> C5).
    # But we CAN connect to 3 of a K4, or 2 of two different K4s, etc.
    
    # The key insight from earlier analysis: to reduce alpha from 9 to 8,
    # the extra vertex must be adjacent to at least one vertex in EVERY
    # transversal (one from each K4). This requires being adjacent to all 4
    # of at least one K4, which is impossible (creates C5).
    
    # BUT: if we have multiple extra vertices, we can split the "hitting" responsibility.
    # Also, we can add edges BETWEEN K4s to change the structure.
    
    # For now, try a greedy approach: add random edges and check validity.
    
    for restart in range(100):
        if time.time() > _DEADLINE - 5:
            break
        
        # Reset
        adj_bits = [0] * n
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    adj_bits[base+i] |= (1 << (base+j))
                    adj_bits[base+j] |= (1 << (base+i))
        
        # Add random edges involving extra vertices
        # Each extra vertex gets 2-3 random neighbors from the K4s
        for ev in range(base_n, n):
            # Choose 2-3 random vertices from K4s to connect to
            num_conn = 2 + random.randint(0, 1)
            candidates = list(range(base_n))
            random.shuffle(candidates)
            for c in candidates[:num_conn]:
                # Check if adding this edge creates a C5
                adj_bits[ev] |= (1 << c)
                adj_bits[c] |= (1 << ev)
        
        # Check C5
        if _check_c5_exists(n, adj_bits):
            continue
        
        # Check alpha
        if not _has_independent_set(n, adj_bits, 9):
            edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_bits[u] >> v) & 1:
                        edges.append([u, v])
            return n, edges
    
    return None

def construct():
    start_time = time.time()
    
    # First, try the known 32-vertex construction (8 disjoint K4s)
    # This is guaranteed valid
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                edges.append([base+i, base+j])
    
    best_n = 32
    best_edges = edges
    
    # Try to find larger constructions
    # 1. Try modular/Paley-like constructions
    result = _try_modular_construction(34)
    if result is not None:
        n_found, edges_found = result
        if n_found > best_n:
            best_n = n_found
            best_edges = edges_found
    
    # 2. Try random Paley-like
    result = _try_paley_like(34)
    if result is not None:
        n_found, edges_found = result
        if n_found > best_n:
            best_n = n_found
            best_edges = edges_found
    
    # 3. Try local search from K4s for n=33, 34, 35
    for target_n in [33, 34, 35]:
        if time.time() > _DEADLINE - 10:
            break
        result = _try_local_search_from_k4s(target_n)
        if result is not None:
            n_found, edges_found = result
            if n_found > best_n:
                best_n = n_found
                best_edges = edges_found
    
    return {'n': best_n, 'edges': best_edges}
