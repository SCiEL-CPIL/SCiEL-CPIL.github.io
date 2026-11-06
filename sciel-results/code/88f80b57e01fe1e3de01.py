"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def _has_k222_v2(adj, n):
    """Check if graph with adjacency bitmasks adj on n vertices contains K2,2,2 as a subgraph."""
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
    """Count the number of K2,2,2 subgraphs in the graph."""
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

def _complement(adj, n):
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    return comp

def _count_both(adj, n):
    """Count total violations in both color classes."""
    return _count_k222_violations(adj, n) + _count_k222_violations(_complement(adj, n), n)

def _local_search(adj, n, time_limit=100):
    """Local search to minimize K2,2,2 violations in both color classes using simulated annealing."""
    start = time.time()
    cur_viol = _count_both(adj, n)
    best_viol = cur_viol
    best_adj = adj[:]
    T = max(1.0, cur_viol * 0.5)
    T_min = 0.01
    cool_rate = 0.999
    
    for it in range(2000000):
        if time.time() - start > time_limit:
            break
        if best_viol == 0:
            break
        
        # Pick a random edge to flip
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        
        # Flip the edge
        adj[i] ^= (1 << j)
        adj[j] ^= (1 << i)
        
        new_viol = _count_both(adj, n)
        delta = new_viol - cur_viol
        
        if delta < 0 or random.random() < (2.718281828 ** (-delta / T)):
            cur_viol = new_viol
            if new_viol < best_viol:
                best_viol = new_viol
                best_adj = adj[:]
        else:
            # Revert
            adj[i] ^= (1 << j)
            adj[j] ^= (1 << i)
        
        T = max(T_min, T * cool_rate)
    
    return best_adj, best_viol

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    # First try Paley graph on 29 vertices
    q = 29
    adj = _build_paley(q)
    if _is_valid(adj, q):
        colors = _paley_colors(adj, q)
        return {'n': q, 'colors': colors}
    
    # Try to find a valid 30-vertex coloring
    if time.time() < deadline - 5:
        n30 = 30
        best_adj30 = None
        best_score = float('inf')
        
        # Phase 1: Try structured initializations
        # Paley 29 + vertex connected to all (worst case for color 0)
        # Paley 29 + vertex connected to none (worst case for color 1)
        # Paley 29 + vertex with balanced connections
        
        # Try: vertex 29 connected to vertices that are NOT adjacent in Paley (complement pattern)
        # and vertex 29 connected to vertices that ARE adjacent (same pattern)
        init_candidates = []
        
        # Candidate 1: connect to all
        a1 = adj[:] + [0]
        for i in range(29):
            a1[i] |= (1 << 29)
            a1[29] |= (1 << i)
        init_candidates.append(a1)
        
        # Candidate 2: connect to none
        a2 = adj[:] + [0]
        init_candidates.append(a2)
        
        # Candidate 3: connect to even-indexed vertices
        a3 = adj[:] + [0]
        for i in range(29):
            if i % 2 == 0:
                a3[i] |= (1 << 29)
                a3[29] |= (1 << i)
        init_candidates.append(a3)
        
        # Candidate 4: connect to vertices 0-14
        a4 = adj[:] + [0]
        for i in range(15):
            a4[i] |= (1 << 29)
            a4[29] |= (1 << i)
        init_candidates.append(a4)
        
        # Candidate 5: connect based on Paley adjacency pattern (copy vertex 0's pattern)
        a5 = adj[:] + [0]
        for i in range(29):
            if (adj[0] >> i) & 1:
                a5[i] |= (1 << 29)
                a5[29] |= (1 << i)
        init_candidates.append(a5)
        
        # Candidate 6: connect to non-neighbors of vertex 0
        a6 = adj[:] + [0]
        for i in range(1, 29):
            if not ((adj[0] >> i) & 1):
                a6[i] |= (1 << 29)
                a6[29] |= (1 << i)
        init_candidates.append(a6)
        
        # Add some random initializations
        for r in range(30):
            ar = adj[:] + [0]
            for i in range(29):
                if random.random() < 0.5:
                    ar[i] |= (1 << 29)
                    ar[29] |= (1 << i)
            init_candidates.append(ar)
        
        # Score all initializations
        for cand in init_candidates:
            if time.time() > deadline - 5:
                break
            score = _count_both(cand, n30)
            if score < best_score:
                best_score = score
                best_adj30 = cand[:]
            if score == 0:
                break
        
        if best_adj30 is not None and best_score == 0:
            colors = []
            for i in range(n30):
                for j in range(i + 1, n30):
                    if (best_adj30[i] >> j) & 1:
                        colors.append(0)
                    else:
                        colors.append(1)
            return {'n': n30, 'colors': colors}
        
        # Phase 2: Local search from best candidate
        if best_adj30 is not None and time.time() < deadline - 10:
            remaining_time = deadline - 10 - time.time()
            adj30_opt, viol = _local_search(best_adj30, n30, time_limit=remaining_time)
            if viol == 0:
                colors = []
                for i in range(n30):
                    for j in range(i + 1, n30):
                        if (adj30_opt[i] >> j) & 1:
                            colors.append(0)
                        else:
                            colors.append(1)
                return {'n': n30, 'colors': colors}
    
    # Fallback: return Paley 29 if valid, otherwise Paley 17
    if _is_valid(adj, 29):
        colors = _paley_colors(adj, 29)
        return {'n': 29, 'colors': colors}
    
    q = 17
    adj = _build_paley(q)
    colors = _paley_colors(adj, q)
    return {'n': q, 'colors': colors}
# EVOLVE-BLOCK-END
