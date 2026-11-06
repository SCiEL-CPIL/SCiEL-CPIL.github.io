import time
import random
from itertools import combinations

def _count_violations(adj, n):
    """
    Count total K2,2,2 violations in both color classes.
    """
    full = (1 << n) - 1
    
    def count_in_graph(adj_list):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_list[u] >> v) & 1:
                    continue
                common_mask = adj_list[u] & adj_list[v]
                common_mask &= ~(1 << u)
                common_mask &= ~(1 << v)
                
                cn_list = []
                m = common_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    cn_list.append(idx)
                    m ^= lsb
                
                L = len(cn_list)
                for i in range(L):
                    b1 = cn_list[i]
                    for j in range(i + 1, L):
                        b2 = cn_list[j]
                        common_b1_b2 = adj_list[b1] & adj_list[b2] & common_mask
                        if common_b1_b2.bit_count() >= 2:
                            count += 1
        return count
    
    c0 = count_in_graph(adj)
    
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    
    c1 = count_in_graph(comp)
    
    return c0 + c1

def _find_one_violation_edge(adj, n):
    """
    Find one violating edge. Returns (u, v) or None.
    """
    full = (1 << n) - 1
    
    def find_in_graph(adj_list):
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_list[u] >> v) & 1:
                    continue
                common_mask = adj_list[u] & adj_list[v]
                common_mask &= ~(1 << u)
                common_mask &= ~(1 << v)
                
                cn_list = []
                m = common_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    cn_list.append(idx)
                    m ^= lsb
                
                L = len(cn_list)
                for i in range(L):
                    b1 = cn_list[i]
                    for j in range(i + 1, L):
                        b2 = cn_list[j]
                        common_b1_b2 = adj_list[b1] & adj_list[b2] & common_mask
                        if common_b1_b2.bit_count() >= 2:
                            return (u, v)
        return None
    
    result = find_in_graph(adj)
    if result:
        return result
    
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    
    return find_in_graph(comp)

def _paley_29():
    """Return Paley graph on 29 vertices."""
    q = 29
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

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 29
    best_adj = _paley_29()
    
    # Try multiple initializations for vertex 29
    init_patterns = [
        lambda i: random.random() < 0.5,      # random
        lambda i: True,                        # all connected
        lambda i: False,                       # no connections
        lambda i: i % 2 == 0,                  # even indices
        lambda i: i % 3 == 0,                  # multiples of 3
        lambda i: i < 15,                      # first half
    ]
    
    for attempt, init_fn in enumerate(init_patterns):
        if time.time() > deadline - 10:
            break
        
        n = 30
        adj = [0] * n
        
        # Copy Paley 29
        for i in range(29):
            adj[i] = best_adj[i]
        
        # Initialize vertex 29
        for i in range(29):
            if init_fn(i):
                adj[i] |= (1 << 29)
                adj[29] |= (1 << i)
        
        # Simulated annealing
        T = 1.0
        cooling = 0.9997
        min_T = 0.003
        max_iters = 40000
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Count violations
            total_violations = _count_violations(adj, n)
            
            if total_violations == 0:
                # Valid coloring found for n=30
                best_n = n
                best_adj = adj[:]
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (best_adj[i] >> j) & 1:
                            colors.append(0)
                        else:
                            colors.append(1)
                return {'n': n, 'colors': colors}
            
            # Find a violating edge to flip
            viol = _find_one_violation_edge(adj, n)
            if viol is None:
                break
            
            u, v = viol
            
            # Flip the edge
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_violations = _count_violations(adj, n)
            delta = new_violations - total_violations
            
            if delta <= 0 or (T > min_T and random.random() < (2.718281828 ** (-delta / max(T, 0.001)))):
                # Accept
                T *= cooling
            else:
                # Reject: undo
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                T *= cooling
        
        # Local search phase: greedy improvements
        for ls_iter in range(3000):
            if time.time() > deadline - 5:
                break
            
            total_violations = _count_violations(adj, n)
            if total_violations == 0:
                best_n = n
                best_adj = adj[:]
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (best_adj[i] >> j) & 1:
                            colors.append(0)
                        else:
                            colors.append(1)
                return {'n': n, 'colors': colors}
            
            # Try flipping a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_violations = _count_violations(adj, n)
            if new_violations < total_violations:
                pass  # Accept improvement
            else:
                # Reject
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Fallback to n=29
    n = 29
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
