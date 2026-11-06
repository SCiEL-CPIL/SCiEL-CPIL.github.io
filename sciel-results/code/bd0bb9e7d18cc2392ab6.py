"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def _has_k222_fast(adj, n):
    """Check if graph with adjacency bitmasks adj on n vertices contains K2,2,2 as a subgraph.
    
    K2,2,2 has two parts of size 2. A subgraph K2,2,2 exists iff there are two disjoint
    pairs (a,b) and (c,d) such that all 4 cross-edges exist.
    Equivalently: there exist vertices a,b with |N(a)∩N(b)| >= 4, and within that common
    neighborhood there exist two vertices c,d that share at least 2 common neighbors in the
    common neighborhood of a,b.
    """
    for a in range(n):
        for b in range(a + 1, n):
            if not ((adj[a] >> b) & 1):
                continue
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            if cn_mask.bit_count() < 4:
                continue
            # Check if any two vertices in cn_mask share >= 2 common neighbors within cn_mask
            m = cn_mask
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                m ^= lsb
                # For each other vertex v in cn_mask (v > u to avoid duplicates), check common neighbors
                m2 = cn_mask & ~(1 << u) & m  # remaining vertices after u
                # Actually we need to check pairs, let's iterate differently
                m2 = cn_mask & ~(1 << u)
                while m2:
                    lsb2 = m2 & (-m2)
                    v = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    if v <= u:
                        continue
                    common_uv = adj[u] & adj[v] & cn_mask
                    if common_uv.bit_count() >= 2:
                        return True
    return False

def _has_k222_v2(adj, n):
    """Faster K2,2,2 detection using precomputed common-neighbor counts."""
    for a in range(n):
        for b in range(a + 1, n):
            if not ((adj[a] >> b) & 1):
                continue
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            cnt = cn_mask.bit_count()
            if cnt < 4:
                continue
            # Need to check if there exist c,d in cn_mask with |N(c)∩N(d)∩cn_mask| >= 2
            # Iterate over pairs in cn_mask
            m = cn_mask
            while m:
                lsb = m & (-m)
                c = lsb.bit_length() - 1
                m ^= lsb
                # Check pairs (c, d) where d is in remaining m
                m2 = m
                while m2:
                    lsb2 = m2 & (-m2)
                    d = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    common_cd = adj[c] & adj[d] & cn_mask
                    if common_cd.bit_count() >= 2:
                        return True
    return False

def _is_valid(adj, n):
    """Check both color classes are K2,2,2-free."""
    if _has_k222_v2(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222_v2(comp, n):
        return False
    return True

def _build_paley(q):
    """Build Paley graph on q vertices (q prime, q ≡ 1 mod 4)."""
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _paley_colors(adj, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _count_k222_violations(adj, n):
    """Count the number of K2,2,2 subgraphs in the graph (for optimization)."""
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            if not ((adj[a] >> b) & 1):
                continue
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            if cn_mask.bit_count() < 4:
                continue
            m = cn_mask
            while m:
                lsb = m & (-m)
                c = lsb.bit_length() - 1
                m ^= lsb
                m2 = m
                while m2:
                    lsb2 = m2 & (-m2)
                    d = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    common_cd = adj[c] & adj[d] & cn_mask
                    bc = common_cd.bit_count()
                    if bc >= 2:
                        count += bc * (bc - 1) // 2
    return count

def _local_search(adj, n, max_iters=5000, time_limit=100):
    """Local search to minimize K2,2,2 violations in both color classes."""
    start = time.time()
    best_violations = _count_k222_violations(adj, n) + _count_k222_violations(_complement(adj, n), n)
    best_adj = adj[:]
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        if best_violations == 0:
            break
        
        # Pick a random edge to flip
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        
        # Flip the edge
        adj[i] ^= (1 << j)
        adj[j] ^= (1 << i)
        
        new_violations = _count_k222_violations(adj, n) + _count_k222_violations(_complement(adj, n), n)
        
        if new_violations < best_violations:
            best_violations = new_violations
            best_adj = adj[:]
        else:
            # Revert
            adj[i] ^= (1 << j)
            adj[j] ^= (1 << i)
    
    return best_adj, best_violations

def _complement(adj, n):
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    return comp

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # First try Paley graph on 29 vertices
    q = 29
    adj = _build_paley(q)
    if _is_valid(adj, q):
        colors = _paley_colors(adj, q)
        return {'n': q, 'colors': colors}
    
    # Try to find a valid 30-vertex coloring using local search from Paley 29 + 1 vertex
    # Start with Paley 29 and try to add a 30th vertex
    if time.time() < deadline - 5:
        # Try adding vertex 29 to the Paley 29 graph
        n30 = 30
        adj30 = adj[:] + [0]
        # Try all possible connections for vertex 29
        best_adj30 = None
        best_score = float('inf')
        
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Start from Paley 29 and add vertex 29 with random connections
            a30 = adj[:] + [0]
            for i in range(29):
                if random.random() < 0.5:
                    a30[i] |= (1 << 29)
                    a30[29] |= (1 << i)
            
            score = _count_k222_violations(a30, 30) + _count_k222_violations(_complement(a30, 30), 30)
            if score < best_score:
                best_score = score
                best_adj30 = a30[:]
            
            if score == 0:
                break
        
        if best_adj30 is not None and best_score == 0:
            colors = []
            for i in range(30):
                for j in range(i + 1, 30):
                    if (best_adj30[i] >> j) & 1:
                        colors.append(0)
                    else:
                        colors.append(1)
            return {'n': 30, 'colors': colors}
        
        # Try local search from the best 30-vertex candidate
        if best_adj30 is not None and time.time() < deadline - 10:
            adj30_opt, viol = _local_search(best_adj30, 30, max_iters=10000, time_limit=60)
            if viol == 0:
                colors = []
                for i in range(30):
                    for j in range(i + 1, 30):
                        if (adj30_opt[i] >> j) & 1:
                            colors.append(0)
                        else:
                            colors.append(1)
                return {'n': 30, 'colors': colors}
    
    # Fallback: return Paley 29 if valid, otherwise Paley 17
    if _is_valid(adj, 29):
        colors = _paley_colors(adj, 29)
        return {'n': 29, 'colors': colors}
    
    q = 17
    adj = _build_paley(q)
    colors = _paley_colors(adj, q)
    return {'n': q, 'colors': colors}
# EVOLVE-BLOCK-END
