"""Cayley-graph construction for 4-regular girth-10 graph using group theory."""
import math
import time
import itertools
from collections import deque

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    deadline = time.time() + 135  # safety margin before 150s kill
    t0 = time.time()
    
    best_n = None
    best_edges = None
    
    # Strategy 1: Try cyclic groups Z_m - most promising for small n
    for m in range(243, 501):
        if time.time() > deadline:
            break
        result = _try_cyclic_fast(m, g)
        if result is not None:
            n, edges = result
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
                if n <= 250:
                    break
    
    # Strategy 2: Try product groups Z_a x Z_b with limited search
    if time.time() < deadline:
        for a in range(3, 25):
            for b in range(a, 25):
                if time.time() > deadline:
                    break
                n = a * b
                if n < 243 or (best_n is not None and n >= best_n):
                    continue
                result = _try_product_ab_limited(a, b, g)
                if result is not None:
                    n2, edges = result
                    if best_n is None or n2 < best_n:
                        best_n = n2
                        best_edges = edges
    
    # Strategy 3: Try Z_a x Z_b x Z_c for small factors
    if time.time() < deadline and best_n is None or best_n > 280:
        for a in range(2, 12):
            for b in range(a, 12):
                for c in range(b, 12):
                    if time.time() > deadline:
                        break
                    n = a * b * c
                    if n < 243 or (best_n is not None and n >= best_n):
                        continue
                    result = _try_product_abc(a, b, c, g)
                    if result is not None:
                        n2, edges = result
                        if best_n is None or n2 < best_n:
                            best_n = n2
                            best_edges = edges
    
    if best_n is not None:
        return best_n, best_edges
    
    return _construct_fallback(4, g)


def _try_cyclic_fast(m, g):
    """Try Cayley graph of Z_m with generating set {a, -a, b, -b} using efficient girth check."""
    seen_pairs = set()
    
    for a in range(1, m // 2 + 1):
        for b in range(a + 1, m // 2 + 1):
            if a == b or a == m - b or b == m - a:
                continue
            if (2 * a) % m == 0 or (2 * b) % m == 0:
                continue
            if math.gcd(math.gcd(a, b), m) != 1:
                continue
            
            key = (min(a, m-a), min(b, m-b))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            
            S = [a, m - a, b, m - b]
            
            if _has_short_nb_cycle_cyclic_fast(m, S, g):
                continue
            
            edge_set = set()
            for i in range(m):
                for s in [a, b]:
                    j = (i + s) % m
                    if i < j:
                        edge_set.add((i, j))
                    else:
                        edge_set.add((j, i))
            edges = sorted(edge_set)
            
            deg = [0] * m
            for i, j in edges:
                deg[i] += 1
                deg[j] += 1
            if any(d != 4 for d in deg):
                continue
            
            return m, edges
    
    return None


def _has_short_nb_cycle_cyclic_fast(m, S, g):
    """Fast check for non-backtracking closed walks of length < g in Z_m with step set S.
    
    Uses BFS from each starting step, tracking (position, last_step_index).
    Prunes early when distance reaches g-1.
    """
    S_neg = [(-s) % m for s in S]
    
    for s1_idx, s1 in enumerate(S):
        visited = set()
        queue = deque()
        start_pos = s1 % m
        queue.append((start_pos, s1_idx, 1))
        visited.add((start_pos, s1_idx))
        
        while queue:
            pos, prev_idx, dist = queue.popleft()
            if dist >= g - 1:
                continue
            neg_prev = S_neg[prev_idx]
            for next_idx, s_next in enumerate(S):
                if s_next == neg_prev:
                    continue
                new_pos = (pos + s_next) % m
                new_state = (new_pos, next_idx)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if new_pos == 0 and (dist + 1) < g:
                    if s_next != S_neg[s1_idx]:
                        return True
                queue.append((new_pos, next_idx, dist + 1))
    
    return False


def _try_product_ab_limited(a, b, g):
    """Try Cayley graph of Z_a x Z_b with limited generating set search."""
    candidate_pairs = []
    seen = set()
    
    def norm(v):
        x, y = v
        x = min(x, a - x) if x <= a//2 else a - x
        y = min(y, b - y) if y <= b//2 else b - y
        return (x, y)
    
    for x1 in range(0, a):
        for y1 in range(0, b):
            if x1 == 0 and y1 == 0:
                continue
            for x2 in range(0, a):
                for y2 in range(0, b):
                    if x2 == 0 and y2 == 0:
                        continue
                    if (x2 == x1 and y2 == y1) or (x2 == (-x1) % a and y2 == (-y1) % b):
                        continue
                    # Check that 2*s is not identity
                    if (2*x1) % a == 0 and (2*y1) % b == 0:
                        continue
                    if (2*x2) % a == 0 and (2*y2) % b == 0:
                        continue
                    # Canonical form
                    n1 = norm((x1, y1))
                    n2 = norm((x2, y2))
                    key = (min(n1, n2), max(n1, n2))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_pairs.append(((x1, y1), (x2, y2)))
                    if len(candidate_pairs) >= 500:
                        break
                if len(candidate_pairs) >= 500:
                    break
            if len(candidate_pairs) >= 500:
                break
    
    for (x1, y1), (x2, y2) in candidate_pairs:
        S = [(x1, y1), ((-x1) % a, (-y1) % b), (x2, y2), ((-x2) % a, (-y2) % b)]
        
        if not _is_connected_product(a, b, S):
            continue
        
        if _has_short_nb_cycle_product(a, b, S, g):
            continue
        
        n = a * b
        edge_set = set()
        for i in range(a):
            for j in range(b):
                v = i * b + j
                for (dx, dy) in [S[0], S[2]]:
                    ni = (i + dx) % a
                    nj = (j + dy) % b
                    w = ni * b + nj
                    if v < w:
                        edge_set.add((v, w))
                    else:
                        edge_set.add((w, v))
        edges = sorted(edge_set)
        
        deg = [0] * n
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        if any(d != 4 for d in deg):
            continue
        
        return n, edges
    
    return None


def _try_product_abc(a, b, c, g):
    """Try Cayley graph of Z_a x Z_b x Z_c with limited generating set search."""
    candidate_pairs = []
    seen = set()
    
    def norm3(v):
        x, y, z = v
        x = min(x, a - x) if x <= a//2 else a - x
        y = min(y, b - y) if y <= b//2 else b - y
        z = min(z, c - z) if z <= c//2 else c - z
        return (x, y, z)
    
    # Generate a few candidate pairs
    for x1 in range(0, a):
        for y1 in range(0, b):
            for z1 in range(0, c):
                if x1 == 0 and y1 == 0 and z1 == 0:
                    continue
                if (2*x1)%a == 0 and (2*y1)%b == 0 and (2*z1)%c == 0:
                    continue
                for x2 in range(0, a):
                    for y2 in range(0, b):
                        for z2 in range(0, c):
                            if x2 == 0 and y2 == 0 and z2 == 0:
                                continue
                            if (2*x2)%a == 0 and (2*y2)%b == 0 and (2*z2)%c == 0:
                                continue
                            if (x2==x1 and y2==y1 and z2==z1) or \
                               (x2==(-x1)%a and y2==(-y1)%b and z2==(-z1)%c):
                                continue
                            n1 = norm3((x1,y1,z1))
                            n2 = norm3((x2,y2,z2))
                            key = (min(n1,n2), max(n1,n2))
                            if key in seen:
                                continue
                            seen.add(key)
                            candidate_pairs.append(((x1,y1,z1),(x2,y2,z2)))
                            if len(candidate_pairs) >= 200:
                                break
                        if len(candidate_pairs) >= 200:
                            break
                    if len(candidate_pairs) >= 200:
                        break
                if len(candidate_pairs) >= 200:
                    break
            if len(candidate_pairs) >= 200:
                break
        if len(candidate_pairs) >= 200:
            break
    
    for (x1,y1,z1),(x2,y2,z2) in candidate_pairs:
        S = [(x1,y1,z1), ((-x1)%a,(-y1)%b,(-z1)%c), (x2,y2,z2), ((-x2)%a,(-y2)%b,(-z2)%c)]
        
        if not _is_connected_abc(a,b,c,S):
            continue
        
        if _has_short_nb_cycle_abc(a,b,c,S,g):
            continue
        
        n = a*b*c
        edge_set = set()
        for i in range(a):
            for j in range(b):
                for l in range(c):
                    v = (i*b+j)*c+l
                    for (dx,dy,dz) in [S[0],S[2]]:
                        ni=(i+dx)%a; nj=(j+dy)%b; nl=(l+dz)%c
                        w=(ni*b+nj)*c+nl
                        if v<w: edge_set.add((v,w))
                        else: edge_set.add((w,v))
        edges=sorted(edge_set)
        
        deg=[0]*n
        for i,j in edges:
            deg[i]+=1; deg[j]+=1
        if any(d!=4 for d in deg):
            continue
        
        return n, edges
    
    return None


def _is_connected_product(a, b, S):
    visited = set()
    queue = deque()
    start = (0, 0)
    visited.add(start)
    queue.append(start)
    while queue:
        i, j = queue.popleft()
        for dx, dy in S:
            ni = (i + dx) % a
            nj = (j + dy) % b
            if (ni, nj) not in visited:
                visited.add((ni, nj))
                queue.append((ni, nj))
    return len(visited) == a * b


def _is_connected_abc(a,b,c,S):
    visited=set()
    queue=deque()
    start=(0,0,0)
    visited.add(start)
    queue.append(start)
    while queue:
        i,j,l=queue.popleft()
        for dx,dy,dz in S:
            ni=(i+dx)%a; nj=(j+dy)%b; nl=(l+dz)%c
            if (ni,nj,nl) not in visited:
                visited.add((ni,nj,nl))
                queue.append((ni,nj,nl))
    return len(visited)==a*b*c


def _has_short_nb_cycle_product(a, b, S, g):
    S_neg = [((-s[0]) % a, (-s[1]) % b) for s in S]
    
    for s1_idx, s1 in enumerate(S):
        visited = set()
        queue = deque()
        start_pos = (s1[0] % a, s1[1] % b)
        queue.append((start_pos, s1_idx, 1))
        visited.add((start_pos, s1_idx))
        
        while queue:
            pos, prev_idx, dist = queue.popleft()
            if dist >= g - 1:
                continue
            neg_prev = S_neg[prev_idx]
            for next_idx, s_next in enumerate(S):
                if s_next == neg_prev:
                    continue
                new_pos = ((pos[0] + s_next[0]) % a, (pos[1] + s_next[1]) % b)
                new_state = (new_pos, next_idx)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if new_pos == (0, 0) and (dist + 1) < g:
                    if s_next != S_neg[s1_idx]:
                        return True
                queue.append((new_pos, next_idx, dist + 1))
    
    return False


def _has_short_nb_cycle_abc(a,b,c,S,g):
    S_neg=[((-s[0])%a,(-s[1])%b,(-s[2])%c) for s in S]
    for s1_idx,s1 in enumerate(S):
        visited=set()
        queue=deque()
        start_pos=(s1[0]%a,s1[1]%b,s1[2]%c)
        queue.append((start_pos,s1_idx,1))
        visited.add((start_pos,s1_idx))
        while queue:
            pos,prev_idx,dist=queue.popleft()
            if dist>=g-1:
                continue
            neg_prev=S_neg[prev_idx]
            for next_idx,s_next in enumerate(S):
                if s_next==neg_prev:
                    continue
                new_pos=((pos[0]+s_next[0])%a,(pos[1]+s_next[1])%b,(pos[2]+s_next[2])%c)
                new_state=(new_pos,next_idx)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if new_pos==(0,0,0) and (dist+1)<g:
                    if s_next!=S_neg[s1_idx]:
                        return True
                queue.append((new_pos,next_idx,dist+1))
    return False


def _construct_fallback(k, g):
    """Fallback construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
