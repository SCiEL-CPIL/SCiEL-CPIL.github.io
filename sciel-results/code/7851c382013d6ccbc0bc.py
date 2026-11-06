import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Primes p ≡ 3 mod 4
    primes = [3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]
    
    for n in range(31, 68):
        if time.time() - start_time > deadline - 10:
            break
            
        # Check if n is a candidate for Paley
        if n in primes:
            orient = _pale_tournament_orient(n)
            adj = _orient_to_adj(orient, n)
            if _is_valid(adj, n):
                return {'n': n, 'orientation': orient}
                
        # Try local search
        # Initialize with Paley if possible, else random
        if n in primes:
            init_adj = _orient_to_adj(_pale_tournament_orient(n), n)
        else:
            init_adj = None # Random in _solve
            
        result_orient = _local_search(n, init_adj, max_time=deadline - (time.time() - start_time) - 5)
        if result_orient is not None:
            return {'n': n, 'orientation': result_orient}

    # Fallback
    return {'n': 67, 'orientation': _pale_tournament_orient(67)}

def _pale_tournament_orient(p):
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    n = p
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % p in squares:
                orient.append(0)
            else:
                orient.append(1)
    return orient

def _orient_to_adj(orient, n):
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orient[idx] == 0:
                adj[i] |= (1 << j)
            else:
                adj[j] |= (1 << i)
            idx += 1
    return adj

def _is_valid(adj, n):
    # Check if any 4-set is strongly connected (no dominator)
    for a in range(n-3):
        for b in range(a+1, n-2):
            for c in range(b+1, n-1):
                for d in range(c+1, n):
                    if _is_sc_4(a, b, c, d, adj):
                        return False
    return True

def _is_sc_4(a, b, c, d, adj):
    deg_a = ((adj[a] >> b) & 1) + ((adj[a] >> c) & 1) + ((adj[a] >> d) & 1)
    if deg_a == 3: return False
    deg_b = ((adj[b] >> a) & 1) + ((adj[b] >> c) & 1) + ((adj[b] >> d) & 1)
    if deg_b == 3: return False
    deg_c = ((adj[c] >> a) & 1) + ((adj[c] >> b) & 1) + ((adj[c] >> d) & 1)
    if deg_c == 3: return False
    deg_d = ((adj[d] >> a) & 1) + ((adj[d] >> b) & 1) + ((adj[d] >> c) & 1)
    if deg_d == 3: return False
    return True

def _local_search(n, init_adj, max_time=10.0):
    start = time.time()
    
    if init_adj is None:
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                else:
                    adj[j] |= (1 << i)
    else:
        adj = init_adj[:]
        
    def get_first_violation():
        for a in range(n-3):
            for b in range(a+1, n-2):
                for c in range(b+1, n-1):
                    for d in range(c+1, n):
                        if _is_sc_4(a, b, c, d, adj):
                            return (a, b, c, d)
        return None

    def count_violations():
        count = 0
        for a in range(n-3):
            for b in range(a+1, n-2):
                for c in range(b+1, n-1):
                    for d in range(c+1, n):
                        if _is_sc_4(a, b, c, d, adj):
                            count += 1
        return count

    current_viol = count_violations()
    best_viol = current_viol
    best_adj = adj[:]
    
    if current_viol == 0:
        return _adj_to_orient(adj, n)

    while time.time() - start < max_time:
        viol = get_first_violation()
        if viol is None:
            return _adj_to_orient(adj, n)
            
        a, b, c, d = viol
        pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        random.shuffle(pairs)
        
        improved = False
        for u, v in pairs:
            # Flip
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] |= (1 << u)
            else:
                adj[v] &= ~(1 << u)
                adj[u] |= (1 << v)
                
            new_viol = count_violations()
            
            if new_viol < best_viol:
                best_viol = new_viol
                best_adj = adj[:]
                improved = True
                break
            else:
                # Revert
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[v] &= ~(1 << u)
                    adj[u] |= (1 << v)
                    
        if not improved:
            # Random walk
            for _ in range(5):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[v] &= ~(1 << u)
                    adj[u] |= (1 << v)
                    
                new_viol = count_violations()
                
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_adj = adj[:]
                    break
                else:
                    # Revert
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[v] &= ~(1 << u)
                        adj[u] |= (1 << v)
                        
        if best_viol == 0:
            break
            
    if best_viol == 0:
        return _adj_to_orient(best_adj, n)
    return None

def _adj_to_orient(adj, n):
    orient = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                orient.append(0)
            else:
                orient.append(1)
    return orient
