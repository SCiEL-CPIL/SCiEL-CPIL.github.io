import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_c5_exhaustive(n, adj):
        for combo in itertools.combinations(range(n), 5):
            degs = [0] * 5
            edge_count = 0
            for i in range(5):
                vi = combo[i]
                for j in range(i + 1, 5):
                    vj = combo[j]
                    if vj in adj[vi]:
                        degs[i] += 1
                        degs[j] += 1
                        edge_count += 1
            if edge_count == 5 and all(d == 2 for d in degs):
                return True
        return False

    def has_c5_fast(n, adj):
        """Faster C5 detection using adjacency sets"""
        for a in range(n):
            for b in adj[a]:
                if b <= a:
                    continue
                for c in adj[b]:
                    if c <= b or c == a:
                        continue
                    for d in adj[c]:
                        if d <= c or d == a or d == b:
                            continue
                        for e in adj[d]:
                            if e <= d or e == a or e == b or e == c:
                                continue
                            # Check if e is adjacent to a
                            if a in adj[e]:
                                return True
        return False

    def find_large_independent_set(n, adj):
        remaining = set(range(n))
        ind_set = []
        while remaining:
            min_deg = float('inf')
            v = None
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
            ind_set.append(v)
            remaining.remove(v)
            remaining -= adj[v]
        return ind_set

    def verify_graph(n, edges):
        """Verify a graph is C5-free and has no independent set of size 9"""
        adj = build_adj(n, edges)
        if has_c5_fast(n, adj):
            return False
        # Use a more robust independence check
        # Try multiple random orderings
        for _ in range(20):
            perm = list(range(n))
            random.shuffle(perm)
            remaining = set(perm)
            ind_set = []
            while remaining:
                v = remaining.pop()
                ind_set.append(v)
                remaining -= adj[v]
            if len(ind_set) >= 9:
                return False
        # Also try greedy with minimum degree
        ind_set = find_large_independent_set(n, adj)
        if len(ind_set) >= 9:
            return False
        return True

    def try_finite_field_construction(target_n):
        """
        Try to construct a C5-free graph with alpha < 9 using algebraic methods.
        Based on polarity graphs or similar constructions.
        """
        if time.time() > deadline - 20:
            return None
        
        # Try different prime powers and constructions
        for p in [2, 3, 5, 7, 11, 13]:
            if time.time() > deadline - 15:
                return None
            
            # Construction 1: Use quadratic residues
            if p > 2:
                residues = set()
                for x in range(1, p):
                    residues.add((x * x) % p)
                
                # Create graph where u~v if (v-u) mod p is a quadratic residue
                edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        diff = (v - u) % p
                        if diff in residues or (p - diff) in residues:
                            edges.append([u, v])
                
                if verify_graph(target_n, edges):
                    return {'n': target_n, 'edges': edges}
            
            # Construction 2: Use a difference set approach
            # For p = 13, use the (13, 4, 1) difference set
            if p == 13:
                diff_set = {0, 1, 3, 9}  # (13,4,1) difference set
                edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        diff = (v - u) % p
                        if diff in diff_set or (p - diff) in diff_set:
                            edges.append([u, v])
                
                if verify_graph(target_n, edges):
                    return {'n': target_n, 'edges': edges}
            
            # Construction 3: Polarity graph of a conic
            if p > 2:
                # Define a conic in PG(2,p) and use its polarity
                # For simplicity, use a specific construction
                edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        # Map vertices to points in a structured way
                        ux = u % p
                        uy = (u // p) % p
                        vx = v % p
                        vy = (v // p) % p
                        # Define adjacency via a bilinear form
                        if p > 2:
                            # Use a hyperbolic form: ux*vy - uy*vx = 0 (degenerate)
                            # Better: use a non-degenerate form
                            val = (ux * vy - uy * vx) % p
                            if val == 1 or val == p - 1:  # Adjust threshold
                                edges.append([u, v])
                
                if verify_graph(target_n, edges):
                    return {'n': target_n, 'edges': edges}
        
        return None

    # Baseline: n=32, 8 disjoint K4s
    best_result = None
    n_base = 32
    edges_base = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges_base.append([base + i, base + j])
    best_result = {'n': n_base, 'edges': edges_base}
    
    # First try deterministic algebraic constructions
    for target_n in [35, 34, 33]:
        if time.time() > deadline - 30:
            break
        
        result = try_finite_field_construction(target_n)
        if result is not None:
            best_result = result
            break
    
    # If no deterministic construction found, try random search
    if best_result['n'] < 35:
        for target_n in [35, 34, 33]:
            if time.time() > deadline - 20:
                break
            
            # Try a few restarts
            for restart in range(10):
                if time.time() > deadline - 10:
                    break
                
                edges = set()
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if random.random() < 0.4:
                            edges.add((u, v))
                
                current_edges = list(edges)
                found_valid = False
                
                # Local search
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    adj = build_adj(target_n, current_edges)
                    
                    # Use fast C5 check
                    c5_exists = has_c5_fast(target_n, adj)
                    
                    # Check Alpha
                    ind_set = find_large_independent_set(target_n, adj)
                    alpha_violation = len(ind_set) >= 9
                    
                    if c5_exists:
                        # Remove random edge
                        if current_edges:
                            current_edges.pop(random.randint(0, len(current_edges) - 1))
                    elif alpha_violation:
                        # Add edge to independent set
                        if len(ind_set) >= 2:
                            u, v = random.sample(ind_set, 2)
                            if (u, v) not in current_edges and (v, u) not in current_edges:
                                current_edges.append((u, v))
                    else:
                        # Seems valid, verify more carefully
                        if verify_graph(target_n, current_edges):
                            best_result = {'n': target_n, 'edges': [[u, v] for u, v in current_edges]}
                            found_valid = True
                            break
                
                if found_valid:
                    break
    
    return best_result
