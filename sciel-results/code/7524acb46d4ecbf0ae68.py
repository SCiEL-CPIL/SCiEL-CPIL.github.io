# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def check_c4_safe(adj, n, u, v):
        """Check if adding edge (u,v) creates a C4.
        A C4 would be u-x-y-v-u where x in N(u), y in N(v), x!=y, x~y."""
        nu = adj[u]
        nv = adj[v]
        # Need to find x in nu, y in nv, x != y, such that x ~ y
        # Iterate over smaller set
        if len(nu) > len(nv):
            nu, nv = nv, nu
        for x in nu:
            for y in nv:
                if x == y:
                    continue
                if y in adj[x]:
                    return True
        return False
    
    def build_polarity_graph(n):
        """Build a polarity graph of a projective plane as a C4-free seed.
        For projective plane of order q, the polarity graph has q^2+q+1 vertices
        and is C4-free with regular degree q+1.
        We use this as a starting point and extend to n vertices."""
        # Find largest q such that q^2+q+1 <= n
        q = 1
        while (q+1)**2 + (q+1) + 1 <= n:
            q += 1
        
        plane_n = q*q + q + 1
        
        # Build polarity graph of projective plane of order q
        # Points and lines are both indexed 0..q^2+q
        # A point (i,j) is represented as i*q + j for i,j in 0..q
        # Actually, let's use a simpler construction:
        # The projective plane PG(2,q) has points that can be represented as
        # 1-dimensional subspaces of F_q^3.
        # For simplicity, we'll use a known polarity construction.
        
        # For q=2: PG(2,2) has 7 points, polarity graph is Fano plane complement
        # For q=3: PG(2,3) has 13 points
        # For q=4: PG(2,4) has 21 points
        # For q=5: PG(2,5) has 31 points
        # For q=6: not a prime power
        # For q=7: PG(2,7) has 57 points (too large for n<=47)
        
        # Let's build the polarity graph directly using coordinates
        # Points of PG(2,q) can be represented as equivalence classes of nonzero vectors in F_q^3
        # We'll enumerate all 1-dimensional subspaces
        
        def mod(a, m):
            return a % m
        
        def is_prime_power(q):
            # Check if q is a prime power
            if q < 2:
                return False
            p = 2
            while p * p <= q:
                if q % p == 0:
                    # Check if q is a power of p
                    temp = q
                    while temp % p == 0:
                        temp //= p
                    if temp == 1:
                        return True
                    else:
                        p += 1
                        break
                else:
                    p += 1
            # Check if q is prime
            for i in range(2, int(q**0.5)+1):
                if q % i == 0:
                    return False
            return True
        
        if not is_prime_power(q) or q < 2:
            # Fallback: use a random C4-free graph
            return None, 0
        
        # Build finite field F_q
        # For small q, we can use simple modular arithmetic if q is prime
        # For prime powers, we need GF(p^k)
        # Let's handle primes first, then extend
        
        # Find prime base and exponent
        p = None
        k = 1
        for candidate_p in range(2, q+1):
            temp = q
            while temp % candidate_p == 0:
                temp //= candidate_p
                k += 1
            if temp == 1:
                p = candidate_p
                break
        
        if p is None:
            return None, 0
        
        # Build GF(p^k)
        # For k=1, just use Z_p
        if k == 1:
            def gf_add(a, b):
                return (a + b) % p
            
            def gf_mul(a, b):
                return (a * b) % p
            
            def gf_inv(a):
                if a == 0:
                    return 0
                for i in range(1, p):
                    if gf_mul(a, i) == 1:
                        return i
                return 0
        else:
            # Need to build GF(p^k) - this is complex
            # For now, skip non-prime q
            return None, 0
        
        # Enumerate points of PG(2,q) as 1-dimensional subspaces of F_q^3
        points = []
        seen = set()
        
        for x in range(q):
            for y in range(q):
                # Point (x, y, 1)
                pt = (x, y, 1)
                key = tuple(sorted(pt))
                if key not in seen:
                    seen.add(key)
                    points.append(pt)
        
        for x in range(q):
            # Point (x, 1, 0)
            pt = (x, 1, 0)
            key = tuple(sorted(pt))
            if key not in seen:
                seen.add(key)
                points.append(pt)
        
        # Point (1, 0, 0)
        pt = (1, 0, 0)
        key = tuple(sorted(pt))
        if key not in seen:
            seen.add(key)
            points.append(pt)
        
        # We should have q^2 + q + 1 points
        if len(points) != plane_n:
            return None, 0
        
        # Define a polarity: point P=(x,y,z) is orthogonal to line L if x*x' + y*y' + z*z' = 0 (for simplicity)
        # Actually, a standard polarity in PG(2,q) maps point (x,y,z) to the line
        # -y*x + x*y + (x^2+y^2)*z = 0 (for odd q) or similar
        # Let's use a simpler approach: define adjacency based on a quadratic form
        
        # For the polarity graph, two points P and Q are adjacent if P lies on the polar line of Q
        # Let's use the standard polarity: P=(x1,y1,z1) is orthogonal to Q=(x2,y2,z2) if
        # x1*x2 + y1*y2 + z1*z2 = 0 (symmetric bilinear form)
        
        adj = [set() for _ in range(plane_n)]
        
        for i in range(plane_n):
            for j in range(i+1, plane_n):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                # Check orthogonality
                if gf_add(gf_add(gf_mul(x1, x2), gf_mul(y1, y2)), gf_mul(z1, z2)) == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # This gives us a C4-free graph on plane_n vertices
        # Now we need to extend to n vertices
        # Add n - plane_n isolated vertices and connect them optimally
        
        extended_adj = [set() for _ in range(n)]
        for i in range(plane_n):
            extended_adj[i] = set(adj[i])
        
        return extended_adj, plane_n
    
    def local_search(adj, n, min_deg_req, time_limit, rng):
        """Local search to maximize minimum degree while maintaining C4-freeness."""
        t0 = time.time()
        
        def min_degree():
            return min(len(adj[i]) for i in range(n))
        
        # Precompute common neighbors count for fast C4 check
        def get_common_neighbors_count(u, v):
            """Count common neighbors of u and v."""
            nu = adj[u]
            nv = adj[v]
            if len(nu) > len(nv):
                nu, nv = nv, nu
            count = 0
            for x in nu:
                if x in nv:
                    count += 1
            return count
        
        while time.time() - t0 < time_limit:
            md = min_degree()
            if md >= min_deg_req:
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append([u, v])
                return edge_list
            
            # Find vertex with minimum degree
            target = min(range(n), key=lambda i: len(adj[i]))
            
            # Try to add an edge from target to a non-neighbor
            candidates = [v for v in range(n) if v != target and v not in adj[target]]
            rng.shuffle(candidates)
            
            added = False
            for v in candidates:
                # Check if adding edge (target, v) creates a C4
                # This happens if there exist x in N(target), y in N(v) with x != y and x~y
                # Equivalently, if |N(target) ∩ N(v)| >= 1, we need to check more carefully
                # Actually, the C4 check is: does there exist x in N(target)\{v}, y in N(v)\{target}
                # such that x~y and x != y?
                
                nu = adj[target]
                nv = adj[v]
                
                # For each x in nu, for each y in nv, if x != y and x~y, then C4
                # But we can optimize: if x is in nv, then x is a common neighbor,
                # and we need to check if x has another neighbor in nv (other than v and target)
                
                # Simple check:
                c4_found = False
                for x in nu:
                    if x == v:
                        continue
                    for y in nv:
                        if x == y or y == target:
                            continue
                        if y in adj[x]:
                            c4_found = True
                            break
                    if c4_found:
                        break
                
                if not c4_found:
                    adj[target].add(v)
                    adj[v].add(target)
                    added = True
                    break
            
            if not added:
                # Try to remove an edge to unblock
                # Remove an edge that doesn't involve target
                removable = [(u, v) for u in range(n) for v in adj[u] if u < v and u != target and v != target]
                if removable:
                    u, v = rng.choice(removable)
                    adj[u].discard(v)
                    adj[v].discard(u)
        
        return None
    
    def greedy_build(n, min_deg_req, time_limit, strategy_seed):
        """Build a C4-free graph with min degree >= min_deg_req using polarity graph seed + local search."""
        t0 = time.time()
        rng = random.Random(strategy_seed)
        
        # Try to use polarity graph as seed
        seed_adj, plane_n = build_polarity_graph(n)
        
        if seed_adj is not None:
            adj = [set(s) for s in seed_adj]
            # Local search to improve
            result = local_search(adj, n, min_deg_req, time_limit, rng)
            if result is not None:
                return {'n': n, 'edges': result}
            # If local search didn't reach min_deg_req, still return what we have
            # if it's better than nothing
            md = min(len(adj[i]) for i in range(n))
            if md >= min_deg_req:
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append([u, v])
                return {'n': n, 'edges': edge_list}
        
        # Fallback: greedy construction
        adj = [set() for _ in range(n)]
        
        def add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)
        
        stuck_count = 0
        repair_budget = 200
        
        while time.time() - t0 < time_limit:
            md = min(len(adj[i]) for i in range(n))
            if md >= min_deg_req:
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append([u, v])
                return {'n': n, 'edges': edge_list}
            
            target = min(range(n), key=lambda i: len(adj[i]))
            
            candidates = [v for v in range(n) if v != target and v not in adj[target]]
            rng.shuffle(candidates)
            
            added = False
            for v in candidates:
                if not check_c4_safe(adj, n, target, v):
                    add_edge(target, v)
                    added = True
                    stuck_count = 0
                    break
            
            if not added:
                stuck_count += 1
                if stuck_count > 10:
                    if repair_budget > 0:
                        repair_budget -= 1
                        removable = [(u, v) for u in range(n) for v in adj[u] if u < v and u != target and v != target]
                        if removable:
                            u, v = rng.choice(removable)
                            adj[u].discard(v)
                            adj[v].discard(u)
                        stuck_count = 0
                    else:
                        break
        
        return None
    
    best_result = None
    best_n = 0
    
    # Try from n=47 down to n=40
    time_budgets = {
        47: 25.0,
        46: 30.0,
        45: 15.0,
        44: 10.0,
        43: 8.0,
        42: 6.0,
        41: 5.0,
        40: 4.0,
        39: 3.0,
    }
    
    seed_counter = 0
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline:
            break
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
        
        min_deg_req = max(1, n - 39)
        t_limit = min(time_budgets.get(n, 5.0), remaining - 2)
        if t_limit < 1:
            continue
        
        # Multiple restarts with different seeds
        attempts = max(1, int(t_limit / 3.0))
        for attempt in range(attempts):
            if time.time() - start_time > deadline:
                break
            seed = 12345 + seed_counter * 7919 + attempt * 104729
            seed_counter += 1
            result = greedy_build(n, min_deg_req, t_limit / max(1, attempts - attempt) if attempt < attempts - 1 else 2.0, seed)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                # Found a valid graph for this n, return it (maximize n)
                return result
    
    # Fallback: return best found or a trivial valid graph
    if best_result is not None:
        return best_result
    
    # Trivial fallback: complete graph on small n (but must be C4-free, so n<=3 for K_n)
    # Actually for n=3, min_deg_req = max(1, 3-39) = 1, K3 has min degree 2 >= 1, and K3 has no C4
    return {'n': 3, 'edges': [[0,1],[1,2],[0,2]]}

# EVOLVE-BLOCK-END
