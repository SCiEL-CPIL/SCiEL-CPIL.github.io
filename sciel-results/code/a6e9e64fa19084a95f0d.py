import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Try PSL(2,q) random search for small n
    # Moore bound
    n_min = _moore_bound(k, g)
    
    # Try prime powers q such that q+1 >= n_min
    # We want small n, so start from low q
    # Limit q to avoid huge search spaces if n_min is large?
    # But n_min is ~428.
    # We can try q up to 2000.
    
    # Generate candidate q values
    candidates_q = []
    # Primes and prime powers
    # Simple sieve for primes up to 2000
    limit_q = 2000
    is_prime = [True] * (limit_q + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit_q)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_q + 1, i):
                is_prime[j] = False
                
    for q in range(2, limit_q + 1):
        # Check if q is prime power
        # For simplicity, just check primes and small prime powers
        if is_prime[q]:
            candidates_q.append(q)
        else:
            # Check if q is p^e
            is_pp = False
            for p in range(2, int(math.isqrt(q)) + 1):
                if q % p == 0:
                    val = q
                    while val % p == 0:
                        val //= p
                    if val == 1:
                        is_pp = True
                        break
            if is_pp:
                candidates_q.append(q)
                
    # Sort by q+1 (n) to try smallest n first
    candidates_q = sorted(set(candidates_q), key=lambda x: x+1)
    
    best_result = None
    
    for q in candidates_q:
        if time.time() > deadline - 5:
            break
            
        n = q + 1
        if n < n_min:
            continue
            
        # Try to find a valid set of k generators in PGL(2,q)
        # We need k=5.
        # S must be symmetric, no identity, no fixed points for any s,
        # and no fixed points for s_i s_j^{-1}.
        
        # Heuristic: Try random sets
        attempts = 500
        for _ in range(attempts):
            if time.time() > deadline - 2:
                break
                
            S = _generate_random_symmetric_set(q, k)
            if S is None:
                continue
                
            # Check validity
            if _is_valid_psl_graph(S, q, k, g):
                # Construct edges
                edges = []
                # Map points to 0..n-1
                # Points are 0..q-1 and q (infinity)
                for x in range(n):
                    px = x if x < q else q
                    for s in S:
                        y = _action_on_point(px, s[0], s[1], s[2], s[3], q)
                        if y is None:
                            continue # Should not happen if valid
                        py = y if y < q else q
                        if px < py:
                            edges.append((px, py))
                        elif px > py:
                            edges.append((py, px))
                        # If px == py, it's a loop, but we checked no fixed points
                        # Wait, _is_valid_psl_graph checks no fixed points.
                        # So px != py.
                        
                # Verify girth one last time? _is_valid_psl_graph did it.
                return n, edges
                
    # 2. Fallback to LUW
    if k == 5 and g == 9:
        # Try d=4 (n=1250)
        n, edges = _construct_luw(k, 4)
        if _girth_at_least(n, edges, g):
            return n, edges
        
        # Try d=5 (n=6250)
        n, edges = _construct_luw(k, 5)
        return n, edges
        
    # General fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _construct_luw(q, d)

def _moore_bound(k, g):
    if k <= 1: return 0
    if g == 1: return 1
    r = g // 2
    if g % 2 == 1:
        s = 1
        term = k
        for _ in range(r - 1):
            s += term
            term *= (k - 1)
        return s
    else:
        s = 0
        term = 1
        for _ in range(r):
            s += term
            term *= (k - 1)
        return 2 * s

def _generate_random_symmetric_set(q, k):
    """
    Generate a random symmetric set S of size k in PGL(2,q).
    Returns list of tuples (a,b,c,d) or None if failed.
    """
    # We need to pick k elements.
    # Since k=5 is small, we can pick 2 pairs of inverses and 1 involution?
    # Or just pick 5 random elements and check if they form a symmetric set?
    # That's hard.
    # Better: Pick 2 random elements A, B. Add A, A^-1, B, B^-1.
    # Then pick 1 involution C. Add C.
    # Total 5 elements.
    # Ensure all distinct.
    
    S = []
    used = set()
    
    # Helper to add element and its inverse
    def add_pair(M):
        inv = _matrix_inverse(M, q)
        # Canonical form for hashing?
        # Just use tuple. But M and lambda*M are same.
        # Normalize M so that a!=0 or (a==0 and b!=0) and a=1?
        # Simpler: just check equality by action? No, too slow.
        # We'll just check if M or inv is in S using a robust equality.
        
        if _is_same_matrix(M, inv, q) and len(S) < k:
             # It's an involution
             if not _is_in_list(M, S, q):
                 S.append(M)
             return
        else:
             if not _is_in_list(M, S, q):
                 S.append(M)
             if not _is_in_list(inv, S, q):
                 S.append(inv)
                 
    # Try to generate
    for _ in range(20):
        S = []
        # Generate 2 pairs
        for _ in range(2):
            M = _random_pgl_element(q)
            add_pair(M)
            if len(S) > k:
                S = []
                break
        if len(S) > k:
            continue
            
        # Generate 1 involution
        # Find a random element M such that M^2 = I (or scalar)
        # M = [[a, b], [c, d]]
        # M^2 = I => a^2+bc=1, ab+bd=0, ac+cd=0, bc+d^2=1
        # This is hard to generate randomly.
        # Alternative: Just pick a random M and check if M^2 is scalar.
        # Probability is low.
        # Alternative: Use a known involution structure?
        # Or just pick a random M and use it, but then we need its inverse to be in S.
        # If we pick M, we add M and M^-1. That's 2 elements.
        # We need 5.
        # So 2 pairs (4) + 1 element.
        # The 5th element must be its own inverse (involution) OR we pick a 3rd pair and drop one?
        # No, size must be exactly k.
        # So we need 2 pairs and 1 involution.
        
        # Finding an involution:
        # In PGL(2,q), involutions are conjugate to [[0,1],[1,0]] or [[1,1],[0,1]]?
        # [[0,1],[1,0]] is an involution.
        # We can take a random P in PGL(2,q) and compute P * [[0,1],[1,0]] * P^-1.
        
        P = _random_pgl_element(q)
        J = (0, 1, 1, 0)
        C = _matrix_mul(P, J, q)
        C = _matrix_mul(C, _matrix_inverse(P, q), q)
        
        # Check if C is involution (it should be)
        # Add C
        if not _is_in_list(C, S, q):
            S.append(C)
        else:
            # Collision, retry
            continue
            
        if len(S) == k:
            # Check no identity
            for M in S:
                if _is_identity(M, q):
                    S = []
                    break
            if len(S) == k:
                return S
                
    return None

def _is_same_matrix(M1, M2, q):
    a1,b1,c1,d1 = M1
    a2,b2,c2,d2 = M2
    # M1 = lambda M2
    # Check if a1*c2 == c1*a2, a1*d2 == d1*a2, etc.
    # If M2 is zero, false.
    if a2 == 0 and b2 == 0 and c2 == 0 and d2 == 0:
        return a1 == 0 and b1 == 0 and c1 == 0 and d1 == 0
    if a2 != 0:
        if (a1 * d2 - d1 * a2) % q != 0: return False
        if (a1 * c2 - c1 * a2) % q != 0: return False
        if (a1 * b2 - b1 * a2) % q != 0: return False
    elif b2 != 0:
        if (b1 * d2 - d1 * b2) % q != 0: return False
        if (b1 * c2 - c1 * b2) % q != 0: return False
        if (b1 * a2 - a1 * b2) % q != 0: return False
    elif c2 != 0:
        if (c1 * d2 - d1 * c2) % q != 0: return False
        if (c1 * b2 - b1 * c2) % q != 0: return False
        if (c1 * a2 - a1 * c2) % q != 0: return False
    elif d2 != 0:
        if (d1 * b2 - b1 * d2) % q != 0: return False
        if (d1 * c2 - c1 * d2) % q != 0: return False
        if (d1 * a2 - a1 * d2) % q != 0: return False
    else:
        return False
    return True

def _is_in_list(M, S, q):
    for s in S:
        if _is_same_matrix(M, s, q):
            return True
    return False

def _is_identity(M, q):
    a,b,c,d = M
    # M = I => a=d=1, b=c=0 (up to scalar)
    # So a*d - b*c != 0 and b=0, c=0, a=d.
    if b == 0 and c == 0 and a == d:
        return True
    return False

def _random_pgl_element(q):
    while True:
        a = random.randint(0, q-1)
        b = random.randint(0, q-1)
        c = random.randint(0, q-1)
        d = random.randint(0, q-1)
        det = (a*d - b*c) % q
        if det != 0:
            return (a,b,c,d)

def _matrix_mul(M1, M2, q):
    a1,b1,c1,d1 = M1
    a2,b2,c2,d2 = M2
    return ((a1*a2 + b1*c2) % q,
            (a1*b2 + b1*d2) % q,
            (c1*a2 + d1*c2) % q,
            (c1*b2 + d1*d2) % q)

def _matrix_inverse(M, q):
    a,b,c,d = M
    det = (a*d - b*c) % q
    inv_det = pow(det, -1, q)
    return ((d*inv_det) % q,
            (-b*inv_det) % q,
            (-c*inv_det) % q,
            (a*inv_det) % q)

def _action_on_point(x, a, b, c, d, q):
    if x == q:
        if c == 0:
            return q
        else:
            inv_c = pow(c, -1, q)
            return (a * inv_c) % q
    else:
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        else:
            inv_den = pow(den, -1, q)
            return (num * inv_den) % q

def _is_valid_psl_graph(S, q, k, g):
    n = q + 1
    # 1. Check no fixed points for any s in S
    for s in S:
        a,b,c,d = s
        # cx^2 + (d-a)x - b = 0
        # Discriminant D = (d-a)^2 + 4bc
        # If D is square, fixed points exist.
        D = ((d-a)*(d-a) + 4*b*c) % q
        # Check if D is quadratic residue
        # Legendre symbol or just pow(D, (q-1)//2, q)
        if D == 0:
            return False
        if pow(D, (q-1)//2, q) == 1:
            return False
            
    # 2. Check no fixed points for s_i s_j^{-1}
    for i in range(len(S)):
        for j in range(len(S)):
            if i == j: continue
            M = _matrix_mul(S[i], _matrix_inverse(S[j], q), q)
            a,b,c,d = M
            D = ((d-a)*(d-a) + 4*b*c) % q
            if D == 0:
                return False
            if pow(D, (q-1)//2, q) == 1:
                return False
                
    # 3. Check girth
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for x in range(n):
        px = x if x < q else q
        for s in S:
            y = _action_on_point(px, s[0], s[1], s[2], s[3], q)
            if y is None: continue
            py = y if y < q else q
            if px != py:
                adj[px].append(py)
                
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Girth check using non-backtracking BFS
    if not _girth_at_least_fast(n, adj, g):
        return False
        
    return True

def _girth_at_least_fast(n, adj, g):
    # Same as _girth_at_least but takes adj list
    for start in range(n):
        dist = {}
        queue = [(start, -1, 0)]
        dist[(start, -1)] = 0
        head = 0
        while head < len(queue):
            u, prev_u, depth = queue[head]
            head += 1
            if depth >= g - 1:
                continue
            for v in adj[u]:
                if v == prev_u:
                    continue
                new_depth = depth + 1
                state = (v, u)
                if state in dist:
                    if v == start and new_depth >= 2:
                        if new_depth < g:
                            return False
                else:
                    dist[state] = new_depth
                    queue.append((v, u, new_depth))
                    if v == start and new_depth >= 2:
                        if new_depth < g:
                            return False
    return True

def _construct_luw(q, d):
    # ... (copy from current) ...
    pass

def _girth_at_least(n, edges, g):
    # ... (copy from current) ...
    pass
