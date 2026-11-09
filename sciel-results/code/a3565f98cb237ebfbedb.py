import time
import random

def _alpha_bnb_fast(adj, n, limit=10):
    """
    Fast branch-and-bound to check if alpha >= limit.
    Returns True if alpha >= limit (fail), False if alpha < limit (pass).
    """
    found = [False]
    
    def bnb(cand, sz):
        if found[0]:
            return
        if sz >= limit:
            found[0] = True
            return
        if sz + cand.bit_count() < limit:
            return
        if cand == 0:
            return
        
        # Heuristic: Pick vertex with max degree in cand
        best_v = -1
        best_deg = -1
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return

        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        bnb(cand & ~v_bit & ~adj[best_v], sz + 1)
        
        if found[0]:
            return
            
        # Branch 2: Exclude best_v
        bnb(cand & ~v_bit, sz)

    bnb((1 << n) - 1, 0)
    return found[0]

def _is_valid_cayley(S, n):
    """
    Check if Cayley graph on Z_n with symmetric connection set S is triangle-free 
    and has alpha < 10.
    """
    S_set = set(S)
    
    # Check sum-free (triangle-free)
    for s1 in S_set:
        for s2 in S_set:
            if (s1 + s2) % n in S_set:
                return None

    # Build adjacency
    adj = [0] * n
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                adj[i] |= (1 << j)
    
    # Check independence number
    if _alpha_bnb_fast(adj, n, limit=10):
        return None
        
    # Return edges
    edges = []
    for u in range(n):
        v_bits = adj[u]
        while v_bits:
            lsb = v_bits & -v_bits
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            v_bits ^= lsb
    return {'n': n, 'edges': edges}

def _try_cayley_search(n):
    """
    Try to find a valid Cayley graph on Z_n.
    """
    candidates = []
    
    # 1. Specific high-degree candidates
    if n == 40:
        # S = {s | s%5 in {1,4}}
        S1 = set(s for s in range(40) if s % 5 in (1, 4))
        candidates.append(S1)
        # S = {s | s%8 in {1,3,5,7}}? 1+1=2 not in. 1+3=4 not in. 1+5=6 not in. 1+7=0 not in.
        # 3+3=6 not in. 3+5=0. 3+7=2. 5+5=2. 5+7=4. 7+7=6.
        # Looks sum-free.
        S2 = set(s for s in range(40) if s % 8 in (1, 3, 5, 7))
        candidates.append(S2)
        
    if n == 39:
        # Try similar modular sets
        # k=13, R={1, 12}
        S1 = set(s for s in range(39) if s % 13 in (1, 12))
        candidates.append(S1)
        
    # 2. General structured candidates
    for k in range(2, n):
        for r in range(1, k // 2 + 1):
            if r == k - r:
                R = {r}
            else:
                R = {r, k - r}
            
            S = set()
            for i in range(n):
                if i % k in R:
                    S.add(i)
            S.discard(0)
            if S:
                candidates.append(S)
                
    # Sort by size (larger degree is better for small alpha)
    candidates.sort(key=lambda x: len(x), reverse=True)
    
    for S in candidates:
        result = _is_valid_cayley(S, n)
        if result:
            return result
            
    # 3. Random search
    rng = random.Random(n)
    for _ in range(20):
        S = set()
        for i in range(1, n // 2 + 1):
            if rng.random() < 0.6: # Bias towards larger sets
                S.add(i)
                S.add((-i) % n)
        S.discard(0)
        if not S:
            continue
        result = _is_valid_cayley(S, n)
        if result:
            return result
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110

    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
            
        result = _try_cayley_search(n)
        if result:
            return result
            
        if n == 26:
            edges_fallback = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
            return {'n': n, 'edges': edges_fallback}

    n = 26
    edges_fallback = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges_fallback}
