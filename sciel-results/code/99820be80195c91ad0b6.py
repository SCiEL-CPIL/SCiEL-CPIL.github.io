"""Rewritten program: construct k-regular graph of girth g using random Cayley graphs."""
# EVOLVE-BLOCK-START
import random
import itertools
from collections import deque
import time

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try to build a Cayley graph on a non-abelian group using involutory generators.
    Use product of symmetric groups S_m1 x S_m2 x ... as the group.
    Select k involutory permutations (transpositions or products of disjoint transpositions)
    as generators, then verify girth via BFS from identity.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # Try random Cayley graph on product of symmetric groups
    # For k=5, g=9: we need 5 involutions
    # Group: S_a x S_b x S_c ... where product of sizes gives n
    # Each generator is an involution (order 2) in the group
    
    def perm_mul(p, q):
        """Compose permutations p and q: result[i] = p[q[i]]"""
        return tuple(p[q[i]] for i in range(len(p)))
    
    def perm_inv(p):
        """Inverse of permutation p"""
        inv = [0] * len(p)
        for i in range(len(p)):
            inv[p[i]] = i
        return tuple(inv)
    
    def is_involution(p):
        """Check if permutation is an involution (p^2 = identity)"""
        for i in range(len(p)):
            if p[p[i]] != i:
                return False
        return True
    
    def random_involution(n, rng):
        """Generate a random involution on n elements"""
        # An involution is a product of disjoint transpositions and fixed points
        # Build by randomly pairing up elements
        remaining = list(range(n))
        rng.shuffle(remaining)
        perm = list(range(n))
        i = 0
        while i < len(remaining):
            if i + 1 < len(remaining) and rng.random() < 0.5:
                # Swap remaining[i] and remaining[i+1]
                a, b = remaining[i], remaining[i+1]
                perm[a] = b
                perm[b] = a
                i += 2
            else:
                i += 1
        return tuple(perm)
    
    def group_element_to_tuple(perm_list):
        """Convert list of permutations to a tuple representation"""
        return tuple(perm_list)
    
    def group_mul(elem1, elem2):
        """Multiply two group elements (each is a tuple of permutations)"""
        return tuple(perm_mul(p1, p2) for p1, p2 in zip(elem1, elem2))
    
    def group_identity(parts):
        """Identity element of the group"""
        return tuple(tuple(range(p)) for p in parts)
    
    def group_inv(elem):
        """Inverse of group element"""
        return tuple(perm_inv(p) for p in elem)
    
    def build_cayley_graph(parts, generators, max_nodes=None):
        """
        Build Cayley graph on product of symmetric groups.
        parts: list of integers [m1, m2, ...] for S_m1 x S_m2 x ...
        generators: list of k group elements (involutions)
        Returns (n, edges) or None if exceeds max_nodes
        """
        identity = group_identity(parts)
        
        # BFS from identity to enumerate group elements
        index = {identity: 0}
        order = [identity]
        edges = []
        
        for vertex in order:
            for gen in generators:
                neighbor = group_mul(vertex, gen)
                if neighbor not in index:
                    if max_nodes is not None and len(order) >= max_nodes:
                        return None
                    index[neighbor] = len(order)
                    order.append(neighbor)
            # Add edges (only from first half to avoid duplicates)
            v_idx = index[vertex]
            for gen in generators:
                neighbor = group_mul(vertex, gen)
                n_idx = index[neighbor]
                if v_idx < n_idx:
                    edges.append((v_idx, n_idx))
        
        return len(order), edges
    
    def compute_girth(n, edges, max_girth=None):
        """
        Compute girth of graph using BFS from each vertex.
        Returns the girth (length of shortest cycle).
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_cycle = float('inf')
        
        for start in range(n):
            # BFS from start
            dist = [-1] * n
            parent = [-1] * n
            queue = deque()
            dist[start] = 0
            queue.append(start)
            
            while queue:
                u = queue.popleft()
                if max_girth is not None and dist[u] + 1 > min_cycle:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and parent[v] != u:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if max_girth is not None and min_cycle < max_girth:
                            return min_cycle
        
        return min_cycle if min_cycle < float('inf') else n + 1
    
    def verify_girth(n, edges, required_girth):
        """
        Verify that the graph has girth at least required_girth.
        Uses non-backtracking walks from every vertex.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS tracking (vertex, previous) states
        # to find shortest non-backtracking cycle
        min_cycle = float('inf')
        
        for start in range(n):
            # State: (current_vertex, previous_vertex)
            # dist[state] = length of walk
            dist = {}
            queue = deque()
            
            for nb in adj[start]:
                state = (nb, start)
                dist[state] = 1
                queue.append(state)
            
            while queue:
                curr, prev = queue.popleft()
                d = dist[(curr, prev)]
                
                if d >= required_girth:
                    break
                
                for nb in adj[curr]:
                    if nb == prev:
                        continue  # No backtracking
                    if nb == start:
                        # Found a cycle of length d+1
                        cycle_len = d + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < required_girth:
                            return False
                    else:
                        state = (nb, curr)
                        if state not in dist:
                            dist[state] = d + 1
                            queue.append(state)
            
            if min_cycle < required_girth:
                return False
        
        return True
    
    # Main construction
    rng = random.Random(42)
    
    # For k=5, g=9, try various group structures
    # Product of symmetric groups: S_a x S_b x S_c ...
    # Total vertices = a * b * c * ...
    # We need k=5 involutory generators
    
    # Strategy: try different combinations of symmetric group factors
    # and random involution generators
    
    best_result = None
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for verification
    
    # Try various group sizes
    # For girth 9, degree 5, we expect n to be in the thousands
    # Try products like S_4 x S_4 x S_5, S_3 x S_3 x S_3 x S_5, etc.
    
    # Candidate factorizations
    candidates = []
    
    # Generate candidate factorizations
    for a in range(3, 10):
        for b in range(3, 10):
            for c in range(3, 10):
                n = a * b * c
                if n >= mb and n <= 5000:
                    candidates.append((a, b, c))
    
    # Also try 2-factor and 4-factor products
    for a in range(4, 15):
        for b in range(4, 15):
            n = a * b
            if n >= mb and n <= 5000:
                candidates.append((a, b))
    
    for a in range(3, 8):
        for b in range(3, 8):
            for c in range(3, 8):
                for d in range(3, 8):
                    n = a * b * c * d
                    if n >= mb and n <= 5000:
                        candidates.append((a, b, c, d))
    
    # Sort by n (smaller first)
    candidates = list(set(candidates))
    candidates.sort(key=lambda x: x[0] * x[1] * (x[2] if len(x) > 2 else 1) * (x[3] if len(x) > 3 else 1))
    
    # Limit number of candidates to try
    candidates = candidates[:50]
    
    for parts in candidates:
        if time.time() - start_time > time_limit:
            break
        
        n_expected = 1
        for p in parts:
            n_expected *= p
        
        # Generate random involutory generators
        # Each generator is a tuple of involutions, one per factor
        # At least one component must be non-identity for the generator to be non-trivial
        
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
            
            generators = []
            valid = True
            
            for _ in range(k):
                gen = []
                # Randomly decide which components are non-identity
                non_id_components = set()
                for i in range(len(parts)):
                    if rng.random() < 0.7:
                        non_id_components.add(i)
                
                if not non_id_components:
                    non_id_components.add(rng.randint(0, len(parts) - 1))
                
                for i in range(len(parts)):
                    if i in non_id_components:
                        gen.append(random_involution(parts[i], rng))
                    else:
                        gen.append(tuple(range(parts[i])))
                
                # Check it's not the identity
                if all(gen[i] == tuple(range(parts[i])) for i in range(len(parts))):
                    valid = False
                    break
                
                generators.append(tuple(gen))
            
            if not valid:
                continue
            
            # Build the Cayley graph
            result = build_cayley_graph(parts, generators, max_nodes=5000)
            if result is None:
                continue
            
            n, edges = result
            
            # Verify girth
            if verify_girth(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                    # If we found a good one, we could break early
                    # but let's try a few more to see if we can do better
                    if n <= mb * 2:
                        break
        
        if best_result is not None and best_result[0] <= mb * 2:
            break
    
    # If we found a valid graph, return it
    if best_result is not None:
        return best_result
    
    # Fallback: use L-U-W construction
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
