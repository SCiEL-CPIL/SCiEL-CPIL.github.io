"""K4-free graph with no independent 6-set via randomized local search."""
import time
import random
from itertools import combinations

# Deadline guard: return before 150s wall-clock
_DEADLINE = time.time() + 140  # safety margin of 10s

def _check_k4(adj, n):
    """Check if graph has a K4. adj[i] is a bitset of neighbors of i."""
    for a in range(n):
        na = adj[a]
        # iterate neighbors of a
        b = (na >> (a + 1)) & ((1 << (n - a - 1)) - 1)
        # simpler: just iterate all pairs
    # brute force with bitsets
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            # common neighbors of a and b, with index > b
            common = (adj[a] & adj[b]) & ~((1 << (b + 1)) - 1)
            # need two more vertices c, d both in common and adjacent to each other
            c_bit = common & ~(1 << (b + 1))  # skip b itself
            # iterate over c in common
            c_list = []
            tmp = common
            while tmp:
                c = (tmp & -tmp).bit_length() - 1
                if c > b:
                    c_list.append(c)
                tmp &= tmp - 1
            for i in range(len(c_list)):
                c = c_list[i]
                # need d in common, d > c, and c~d
                for j in range(i + 1, len(c_list)):
                    d = c_list[j]
                    if (adj[c] >> d) & 1:
                        return True
    return False

def _check_indep6(adj, n):
    """Check if graph has an independent set of size 6.
    Independent set = clique in complement.
    complement_adj[i] = bits of non-neighbors of i (excluding i)."""
    for i in range(n):
        # non-neighbors of i, excluding i
        all_mask = (1 << n) - 1
        non_neigh = all_mask & ~(adj[i] | (1 << i))
        # build complement adjacency
    # Build complement adj
    all_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = all_mask & ~(adj[i] | (1 << i))
    # Find clique of size 6 in complement
    for a in range(n):
        ca = comp_adj[a]
        # b > a, b in ca
        b_list = []
        tmp = ca
        while tmp:
            b = (tmp & -tmp).bit_length() - 1
            if b > a:
                b_list.append(b)
            tmp &= tmp - 1
        for i in range(len(b_list)):
            b = b_list[i]
            # c > b, c in ca and cb
            common1 = ca & comp_adj[b] & ~((1 << (b + 1)) - 1)
            c_list = []
            tmp = common1
            while tmp:
                c = (tmp & -tmp).bit_length() - 1
                if c > b:
                    c_list.append(c)
                tmp &= tmp - 1
            for j in range(len(c_list)):
                c = c_list[j]
                # d > c, d in common1 and cc
                common2 = common1 & comp_adj[c] & ~((1 << (c + 1)) - 1)
                d_list = []
                tmp = common2
                while tmp:
                    d = (tmp & -tmp).bit_length() - 1
                    if d > c:
                        d_list.append(d)
                    tmp &= tmp - 1
                for k in range(len(d_list)):
                    d = d_list[k]
                    # e > d, e in common2 and cd
                    common3 = common2 & comp_adj[d] & ~((1 << (d + 1)) - 1)
                    e_list = []
                    tmp = common3
                    while tmp:
                        e = (tmp & -tmp).bit_length() - 1
                        if e > d:
                            e_list.append(e)
                        tmp &= tmp - 1
                    for l in range(len(e_list)):
                        e = e_list[l]
                        # f > e, f in common3 and ce
                        common4 = common3 & comp_adj[e] & ~((1 << (e + 1)) - 1)
                        tmp = common4
                        while tmp:
                            f = (tmp & -tmp).bit_length() - 1
                            if f > e:
                                return True
                            tmp &= tmp - 1
    return False

def _is_valid(adj, n):
    """Check both constraints."""
    if _check_k4(adj, n):
        return False
    if _check_indep6(adj, n):
        return False
    return True

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        tmp = adj[i] & ~((1 << (i + 1)) - 1)
        while tmp:
            j = (tmp & -tmp).bit_length() - 1
            edges.append([i, j])
            tmp &= tmp - 1
    return edges

def _random_graph(n, p):
    """Generate a random graph with edge probability p."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _local_search(adj, n, max_steps=5000):
    """Local search: flip edges to satisfy constraints.
    Returns (adj, valid) where valid indicates if constraints are met."""
    best_adj = adj[:]
    best_violations = _count_violations(adj, n)
    
    for step in range(max_steps):
        if time.time() > _DEADLINE:
            break
        # Pick a random vertex and flip a random edge incident to it
        u = random.randrange(n)
        # Get all possible neighbors
        candidates = [v for v in range(n) if v != u]
        if not candidates:
            break
        v = random.choice(candidates)
        
        # Flip edge (u, v)
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        viol = _count_violations(adj, n)
        if viol < best_violations:
            best_violations = viol
            best_adj = adj[:]
            if viol == 0:
                return adj, True
        else:
            # Revert with some probability (simulated annealing-like)
            if random.random() > 0.1:
                # Revert
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    return best_adj, best_violations == 0

def _count_violations(adj, n):
    """Count number of K4s and independent 6-sets (approximate via sampling for speed)."""
    # For speed, use a fast check
    if _check_k4(adj, n):
        return 1
    if _check_indep6(adj, n):
        return 1
    return 0

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 140
    
    random.seed(42)
    
    best_n = 0
    best_edges = []
    
    # Try increasing n from 35 down, but start with reasonable attempts
    # The known record is 35, so try to find valid graphs at n=35, 34, etc.
    
    for n in range(35, 16, -1):
        if time.time() > _DEADLINE - 5:
            break
        
        # Try multiple random restarts
        found = False
        for attempt in range(50):
            if time.time() > _DEADLINE - 5:
                break
            
            # Start with a random graph with moderate density
            # For K4-free, we want sparse-ish; for no indep-6, we want dense-ish
            # Balance: p around 0.3-0.5
            p = 0.3 + 0.2 * random.random()
            adj = _random_graph(n, p)
            
            # Quick check
            if _is_valid(adj, n):
                best_n = n
                best_edges = _adj_to_edges(adj, n)
                found = True
                break
            
            # Local search
            adj2, valid = _local_search(adj, n, max_steps=2000)
            if valid:
                best_n = n
                best_edges = _adj_to_edges(adj2, n)
                found = True
                break
        
        if found:
            break
    
    if best_n == 0:
        # Fallback: return the Paley graph on 17 vertices
        squares = {x * x % 17 for x in range(1, 17)}
        edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        return {'n': 17, 'edges': edges}
    
    return {'n': best_n, 'edges': best_edges}
