import time
import random

def _has_k4(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                common = adj[u] & adj[v]
                for a in common:
                    for b in common:
                        if a < b and b in adj[a]:
                            return True
    return False

def _has_ind6(adj, n):
    # Branch and bound for Ind6
    def search(idx, current_set, size):
        if size == 6:
            return True
        if idx == n:
            return False
        if n - idx + size < 6:
            return False
        
        # Try include idx
        can_add = True
        for v in current_set:
            if idx in adj[v]:
                can_add = False
                break
        if can_add:
            current_set.add(idx)
            if search(idx + 1, current_set, size + 1):
                return True
            current_set.remove(idx)
            
        # Try exclude idx
        if search(idx + 1, current_set, size):
            return True
        return False
    return search(0, set(), 0)

def _is_valid(adj, n):
    if _has_k4(adj, n):
        return False
    if _has_ind6(adj, n):
        return False
    return True

def _paley(n):
    if n < 2 or n % 4 != 1:
        return None
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return None
    squares = {(x * x) % n for x in range(1, n)}
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                adj[i].add(j)
                adj[j].add(i)
    return adj

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110
    
    candidates = [37, 29, 17]
    for n in candidates:
        if time.time() - start_time > time_limit:
            break
        adj = _paley(n)
        if adj is None:
            continue
        if _is_valid(adj, n):
            edges = [[i, j] for i in range(n) for j in adj[i] if i < j]
            return {'n': n, 'edges': edges}
            
    # Fallback to 17
    adj = _paley(17)
    edges = [[i, j] for i in range(17) for j in adj[i] if i < j]
    return {'n': 17, 'edges': edges}
# EVOLVE-BLOCK-END
