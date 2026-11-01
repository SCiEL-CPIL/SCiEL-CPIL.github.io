import time
import random

def construct(k, g):
    # Target: 3-regular, girth g
    # Strategy: Voltage lift of Petersen graph over Z_p
    
    deadline = time.time() + 110  # 110s safety margin
    
    # Petersen graph
    # Vertices 0..9
    # Edges:
    # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    m_edges = len(petersen_edges)
    
    # Precompute non-backtracking closed walks of length 5 to g-1 in Petersen
    # This allows fast girth verification for any voltage assignment
    
    # Adjacency list for Petersen
    adj = [[] for _ in range(n_base)]
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    walks = {l: [] for l in range(5, g)}
    
    # Generate walks
    # A walk is a sequence of vertices v0, v1, ..., vl with v0=vl
    # Non-backtracking: v_{i-1} != v_{i+1}
    # We can use DFS/BFS to find them
    # Since girth is 5, we start from length 5
    
    def find_walks(target_len):
        # Start from all edges
        # State: (current_vertex, prev_vertex, length, path)
        # We only care about closed walks, so we can track the start vertex
        # But to avoid duplicates, we can just generate all and check closure
        # Given small size, this is fine
        
        results = []
        # Start with all directed edges
        for u in range(n_base):
            for v in adj[u]:
                # Walk of length 1: u->v
                # We need to extend to target_len
                # Let's use a recursive helper or iterative
                pass
        
        # Iterative approach
        # State: (current, prev, start, length)
        # We want length = target_len and current == start
        # And non-backtracking: prev != next
        
        # Initialize with length 1
        states = []
        for u in range(n_base):
            for v in adj[u]:
                states.append((v, u, u, 1))
                
        for l in range(2, target_len + 1):
            new_states = []
            for curr, prev, start, curr_len in states:
                for next_v in adj[curr]:
                    if next_v == prev:
                        continue
                    new_states.append((next_v, curr, start, curr_len + 1))
            states = new_states
            
        # Filter closed walks
        for curr, prev, start, curr_len in states:
            if curr == start:
                # Check if it's a valid non-backtracking walk
                # The generation ensured non-backtracking steps
                # But we need to ensure the first step didn't backtrack?
                # The state (v, u, u, 1) means we went u->v.
                # Next step from v cannot go to u.
                # So the walk is u->v->...
                # If it closes, it's a cycle.
                # We need to record the sequence of edges or vertices to compute voltage sum
                # Let's re-generate to keep the path
                pass
                
    # Better: Generate paths and store the sequence of directed edges
    # Since we need voltage sums, we need the sequence of edges.
    
    def generate_closed_walks(max_len):
        # Returns list of (length, list_of_directed_edges)
        # Directed edge is (u, v)
        all_walks = {l: [] for l in range(5, max_len + 1)}
        
        # Start from all directed edges
        # path: list of (u, v)
        # We want to close: last v == first u
        
        # To avoid O(2^L) blowup, we rely on small L (max 15)
        # Number of walks in Petersen is small.
        
        def dfs(curr, prev, start, length, path):
            if length == max_len:
                return
            for next_v in adj[curr]:
                if next_v == prev:
                    continue
                new_path = path + [(curr, next_v)]
                new_length = length + 1
                if new_length >= 5 and next_v == start:
                    # Found a closed walk
                    all_walks[new_length].append(new_path)
                if new_length < max_len:
                    dfs(next_v, curr, start, new_length, new_path)
                    
        for u in range(n_base):
            for v in adj[u]:
                dfs(v, u, u, 1, [(u, v)])
                
        return all_walks

    closed_walks = generate_closed_walks(g - 1)
    
    # Now search for a prime p and voltage assignment
    # Try primes p
    # We want n = 10 * p to be small.
    # Lower bound on p is roughly 51.
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    def find_primitive_root(p):
        if p == 2: return 1
        phi = p - 1
        # Factorize phi
        factors = []
        n = phi
        d = 2
        while d * d <= n:
            if n % d:
                d += 1
            else:
                while n % d == 0:
                    n //= d
                factors.append(d)
        if n > 1:
            factors.append(n)
            
        for g in range(2, p):
            ok = True
            for f in factors:
                if pow(g, phi // f, p) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None

    def check_girth(p, voltages):
        # voltages: dict or list mapping edge index to voltage
        # We need to map (u,v) to voltage
        # Let's use a dictionary for directed edges
        # vol[(u,v)]
        
        vol = {}
        for i, (u, v) in enumerate(petersen_edges):
            vol[(u, v)] = voltages[i]
            vol[(v, u)] = (-voltages[i]) % p
            
        for l in range(5, g):
            for walk in closed_walks[l]:
                s = 0
                for u, v in walk:
                    s = (s + vol[(u, v)]) % p
                if s == 0:
                    return False
        return True

    best_n = float('inf')
    best_edges = None
    best_p = None
    best_voltages = None
    
    # Try primes starting from 53
    # We can also try composite m if we use Z_m, but prime is simpler for primitive root
    # Let's stick to primes for now.
    
    # To speed up, we can try the "powers of primitive root" assignment first
    # If it fails, we can try random assignments for that p
    
    p_candidate = 53
    while p_candidate < 500 and time.time() < deadline:
        if is_prime(p_candidate):
            g_root = find_primitive_root(p_candidate)
            if g_root:
                # Assignment 1: Powers of primitive root
                voltages = [pow(g_root, i, p_candidate) for i in range(m_edges)]
                if check_girth(p_candidate, voltages):
                    best_p = p_candidate
                    best_voltages = voltages
                    break
                    
                # Assignment 2: Random (with fixed seed for reproducibility, but we want to find *a* solution)
                # We can try a few random assignments
                # But deterministic is better for "score" stability?
                # The task says "deterministic voltage lift ... assigned as first 15 powers".
                # So we should stick to that or similar deterministic rules.
                # If powers fail, try shifted powers? g^(i+1)?
                # Or just move to next prime.
                
        p_candidate += 1
        
    if best_p is None:
        # If no prime worked with simple powers, fall back to LUW or a larger prime
        # Let's try a larger prime with random search?
        # Or just use the LUW construction as fallback to ensure validity.
        # But LUW is huge.
        # Let's try to find ANY valid voltage assignment for a small prime.
        
        # Try p from 53 to 200 again, but with random voltages (seeded)
        random.seed(42)
        for p_candidate in range(53, 200):
            if time.time() > deadline - 10:
                break
            if not is_prime(p_candidate):
                continue
            for _ in range(10): # 10 random tries per prime
                voltages = [random.randint(1, p_candidate-1) for _ in range(m_edges)]
                if check_girth(p_candidate, voltages):
                    best_p = p_candidate
                    best_voltages = voltages
                    break
            if best_p:
                break
                
    if best_p is None:
        # Fallback to LUW if voltage lift fails (should not happen)
        # Implement LUW fallback
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # ... LUW code ...
        # For now, return a valid but large graph
        # We can just return the Petersen graph if g<=5, but g=16.
        # Let's use the seed program's LUW logic as a black box fallback if needed.
        # But I will implement a simple LUW inside if needed.
        
        # Actually, let's just use a larger prime if we haven't found one.
        # Try p up to 1000
        for p_candidate in range(200, 1000):
            if time.time() > deadline - 5:
                break
            if not is_prime(p_candidate):
                continue
            g_root = find_primitive_root(p_candidate)
            if g_root:
                voltages = [pow(g_root, i, p_candidate) for i in range(m_edges)]
                if check_girth(p_candidate, voltages):
                    best_p = p_candidate
                    best_voltages = voltages
                    break
                    
    if best_p is None:
        # Absolute fallback: Return a trivial invalid graph? No, must be valid.
        # Use LUW.
        return luw_construct(k, g)

    # Construct the lifted graph
    n = 10 * best_p
    edges = []
    
    # Vertex index: v * p + i
    # v in 0..9, i in 0..p-1
    
    # vol[(u,v)]
    vol = {}
    for i, (u, v) in enumerate(petersen_edges):
        vol[(u, v)] = best_voltages[i]
        vol[(v, u)] = (-best_voltages[i]) % best_p
        
    for u in range(10):
        for v in adj[u]:
            # Edge (u, v) in base
            # For each i in Z_p
            # Connect (u, i) to (v, (i + vol[(u,v)]) % p)
            # We add each undirected edge once.
            # To avoid duplicates, we can just add all directed edges and then deduplicate?
            # Or only add if u < v?
            # The lifted graph is undirected.
            # If we add (u,i) -> (v, j) and (v,j) -> (u,i), they are the same edge.
            # Let's just add all pairs where u < v in the base?
            # No, the voltage depends on direction.
            # But the edge in the lifted graph is undirected.
            # The set of edges is { { (u,i), (v, (i+vol[(u,v)])%p) } | (u,v) in E_base, i in Z_p }
            # This set is symmetric: if we take (v,j), j = i+vol[(u,v)], then i = j - vol[(u,v)] = j + vol[(v,u)].
            # So the edge from v to u is the same.
            # So we can iterate over all directed edges in base, and for each i, add the edge.
            # But we will add each edge twice (once for u->v and once for v->u).
            # We can use a set to deduplicate.
            
            for i in range(best_p):
                j = (i + vol[(u, v)]) % best_p
                u_idx = u * best_p + i
                v_idx = v * best_p + j
                if u_idx < v_idx:
                    edges.append((u_idx, v_idx))
                else:
                    edges.append((v_idx, u_idx))
                    
    # Deduplicate edges just in case (though the logic above should produce unique edges if we are careful)
    # Actually, if we iterate over all directed edges (u,v) and all i, we generate each undirected edge twice.
    # (u,i) to (v,j) is generated when processing (u,v) and i.
    # It is also generated when processing (v,u) and j.
    # So we should just use a set.
    
    edge_set = set()
    for u in range(10):
        for v in adj[u]:
            for i in range(best_p):
                j = (i + vol[(u, v)]) % best_p
                u_idx = u * best_p + i
                v_idx = v * best_p + j
                if u_idx < v_idx:
                    edge_set.add((u_idx, v_idx))
                else:
                    edge_set.add((v_idx, u_idx))
                    
    edges = list(edge_set)
    
    return n, edges

def luw_construct(k, g):
    # Fallback from seed
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
