import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (fast, early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- LUW construction for cubic graphs ----------------
    def build_luw_cubic(p, time_limit):
        # p must be prime, p % 4 == 1
        if p % 4 != 1:
            return None
        
        # Generate all trace-0 matrices in SL(2, p)
        # M = [[a, b], [c, -a]] with a^2 + b*c = 1
        # a can be 0..p-1
        # If a == 0, b*c = 1 => b=1, c=1 or b=-1, c=-1 (2 matrices)
        # If a != 0, b*c = 1 - a^2
        # Let val = 1 - a^2. If val == 0, b=1, c=0 or b=0, c=1 (2 matrices)
        # If val != 0, we need pairs (b,c) such that b*c = val.
        # There are p-1 such pairs.
        
        mats = []
        for a in range(p):
            val = (1 - a*a) % p
            if val == 0:
                # b*c = 0 => (1,0) or (0,1)
                mats.append((a, 1, 0, -a % p))
                mats.append((a, 0, 1, -a % p))
            else:
                # Find b such that b != 0, then c = val * b^-1
                # Iterate b from 1 to p-1
                for b in range(1, p):
                    c = (val * pow(b, p-2, p)) % p
                    mats.append((a, b, c, -a % p))
        
        # Deduplicate? 
        # [[a,b],[c,-a]] and [[-a,-b],[-c,a]] are inverses (and negatives).
        # In PSL(2,p), M and -M are the same.
        # So we can divide by 2 roughly.
        
        # We need to pick 3 generators A, B, C.
        # Heuristic: Pick A, B, C such that AB, AC, BC have high orders.
        # To save time, we just pick 3 random ones or specific ones and check girth.
        # Better: Pick A = [[0,1],[-1,0]] (a=0, b=1, c=-1 -> c=p-1)
        # Wait, a=0, val=1. b=1, c=1. M=[[0,1],[1,0]].
        # Let's fix A = [[0,1],[1,0]] (trace 0, det -1? No, det = -1. We need det 1.)
        # M = [[a,b],[c,-a]], det = -a^2 - bc. We need -a^2 - bc = 1 => a^2 + bc = -1.
        # My previous derivation: a^2 + bc = 1.
        # If M = [[a,b],[c,d]], det = ad-bc = 1. If d=-a, det = -a^2 - bc = 1 => a^2+bc = -1.
        # So val = -1 - a^2.
        
        mats = []
        for a in range(p):
            val = (-1 - a*a) % p
            if val == 0:
                mats.append((a, 1, 0, -a % p))
                mats.append((a, 0, 1, -a % p))
            else:
                for b in range(1, p):
                    c = (val * pow(b, p-2, p)) % p
                    mats.append((a, b, c, -a % p))
        
        # Identify unique elements in PSL(2,p)
        # M ~ -M
        seen = set()
        unique_mats = []
        for m in mats:
            # Canonical form: make first non-zero entry positive
            # Or just use tuple
            key = m
            if key[0] < 0 or (key[0]==0 and key[1]<0) or (key[0]==0 and key[1]==0 and key[2]<0):
                key = (-m[0]%p, -m[1]%p, -m[2]%p, -m[3]%p)
            if key not in seen:
                seen.add(key)
                unique_mats.append(m)
        
        if len(unique_mats) < 3:
            return None
            
        # Select 3 generators
        # Simple heuristic: take first 3? No, likely low girth.
        # Try a few random triples or structured ones.
        # Given time constraints, we'll try the first few and check girth.
        # To improve, we can sort by "distance" from identity.
        
        # For now, just try a few combinations.
        # A common choice:
        # A = [[0,1],[-1,0]] -> a=0, b=1, c=-1. val = -1. c = -1 * 1 = -1. OK.
        # B = [[1,0],[0,-1]] -> a=1, b=0, c=0. val = -2. c=0. OK.
        # C = [[0,1],[1,0]] -> a=0, b=1, c=1. val = -1. c = -1. Wait.
        # If a=0, val=-1. b=1, c=-1. So [[0,1],[-1,0]].
        # If b=-1, c=1. So [[0,-1],[1,0]].
        
        # Let's just pick 3 distinct ones from the list and check.
        # To maximize chance, pick ones with different 'a' values.
        
        candidates = []
        for i in range(min(10, len(unique_mats))):
            candidates.append(unique_mats[i])
            
        # Try combinations of 3 from candidates
        # This is O(C(10,3)) = 120 checks.
        # Each check is O(n^2) roughly. n ~ p. p ~ 500. n ~ 1000.
        # 120 * 1000^2 is too slow.
        
        # We must be smarter.
        # Just pick 3 and check. If fail, try next 3.
        
        # Let's pick A, B, C such that they are "far" apart.
        # For simplicity, I'll just return the first valid one found by checking a few fixed triples.
        
        triples = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (0, 1, 4),
            (0, 2, 4),
            (0, 3, 4),
            (1, 2, 4),
            (1, 3, 4),
            (2, 3, 4),
            (0, 1, 5),
            (0, 2, 5),
            (0, 3, 5),
            (0, 4, 5),
            (1, 2, 5),
            (1, 3, 5),
            (1, 4, 5),
            (2, 3, 5),
            (2, 4, 5),
            (3, 4, 5),
        ]
        
        for t in triples:
            if len(unique_mats) <= max(t):
                continue
            A = unique_mats[t[0]]
            B = unique_mats[t[1]]
            C = unique_mats[t[2]]
            
            # Build graph
            def mul(a, b):
                return ((a[0]*b[0] + a[1]*b[2]) % p, (a[0]*b[1] + a[1]*b[3]) % p,
                        (a[2]*b[0] + a[3]*b[2]) % p, (a[2]*b[1] + a[3]*b[3]) % p)
            
            def inv(a):
                # det = 1
                return (a[3], -a[1]%p, -a[2]%p, a[0])
            
            I = (1, 0, 0, 1)
            gens = [A, B, C]
            
            index = {I: 0}
            order = [I]
            edges = []
            
            for i in range(len(order)):
                v = order[i]
                vi = inv(v)
                for w in gens:
                    u = mul(vi, w)
                    if u not in index:
                        index[u] = len(order)
                        order.append(u)
                    if index[v] < index[u]:
                        edges.append((index[v], index[u]))
                        
            n = len(order)
            # Check if simple (no self loops, no multi edges)
            # LUW graphs are simple if generators are distinct and not identity.
            # Girth check
            gc = check_girth(n, edges, g, time_limit)
            if gc == -1:
                return None
            if gc >= g:
                return n, edges
                
        return None

    # ---------------- main ----------------
    if k != 3:
        return (0, []) # Fallback, task is k=3

    # Find primes p % 4 == 1
    # Start with p such that 2(p-1) is not too big, but big enough for girth 16.
    # Moore bound for k=3, g=16: 1 + 3 * (1 + 2 + ... + 2^7) = 1 + 3 * 255 = 766.
    # So n >= 766.
    # p ~ n/2 ~ 383.
    # Try p from 300 to 600.
    
    best = None
    best_n = float('inf')
    
    # Primes p % 4 == 1
    primes = []
    for x in range(300, 600):
        if x % 4 != 1:
            continue
        is_prime = True
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(x)
            
    for p in primes:
        if remaining() < 10:
            break
        res = build_luw_cubic(p, deadline - 5)
        if res is None:
            continue
        n, edges = res
        if n < best_n:
            best_n = n
            best = (n, edges)
            
    if best is not None:
        return best
        
    # If no good LUW found, return a trivial valid graph? No, must be 3-regular girth 16.
    # If we fail, we might return a large LUW that we didn't check fully?
    # Better to just return the first one we can build and hope it's valid?
    # The check_girth is strict.
    
    # Fallback: Return a known valid graph if we had one, but we don't.
    # We must return *something*.
    # If we didn't find one, we might timeout.
    # Let's try one more large prime just in case.
    if remaining() > 5:
        p = 541
        res = build_luw_cubic(p, deadline - 2)
        if res:
            return res
            
    return (0, [])
