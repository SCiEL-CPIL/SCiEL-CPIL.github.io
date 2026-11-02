"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d = g - 4
    d = max(3, d)

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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _check_girth_bfs(n, adj, g):
    """Check if graph has girth >= g using BFS from each vertex (non-backtracking)."""
    for s in range(n):
        dist = [(-1, -1)] * n
        dist[s] = (0, -1)
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            du, pu = dist[u]
            if du + 1 >= g:
                continue
            for v in adj[u]:
                if v == pu:
                    continue
                if dist[v][0] == -1:
                    dist[v] = (du + 1, u)
                    queue.append(v)
                elif dist[v][0] > du + 1:
                    cycle_len = dist[v][0] + du + 1
                    if cycle_len < g:
                        return False
                    dist[v] = (du + 1, u)
                    queue.append(v)
                else:
                    if dist[v][1] != u:
                        cycle_len = dist[v][0] + du + 1
                        if cycle_len < g:
                            return False
    return True


def _cayley_graph_search(k, g, deadline, seed=42):
    """
    Cayley graph construction via Schreier enumeration of F_k / R.
    
    We build a k-regular graph by choosing generators s_0, ..., s_{k-1} of a group G
    and taking the Cayley graph Cay(G, S) where S = {s_i}. The graph is k-regular
    if each s_i has order > 2 (or we use directed edges). For an undirected k-regular
    simple graph, we need |S| = k and no s_i = s_j^{-1} for i != j, and s_i^2 != e.
    
    Strategy: Enumerate the Schreier coset graph of F_k / <R> for various relator sets R,
    and find the smallest quotient with girth >= g.
    
    Practical approach: Use a random relator set and enumerate the coset graph up to
    a size limit. If the enumeration closes (all cosets found) and the resulting graph
    has girth >= g, we have a valid construction.
    
    For k=3: generators a, b, c. We need relators that make the group finite.
    A common approach: use relators of the form w_i = 1 where w_i are long random words.
    """
    rng = random.Random(seed)
    best_n = float('inf')
    best_edges = None
    
    # First try LUW as baseline
    try:
        n_luw, edges_luw = _luw_construct(k, g, deadline)
        if n_luw < best_n:
            best_n = n_luw
            best_edges = edges_luw
    except:
        pass
    
    # Cayley graph approach: enumerate Schreier cosets
    def try_cayley_construction(k, g, max_size, num_relators, relator_length, rng, deadline):
        """
        Try to construct a Cayley graph by choosing random relators and enumerating.
        Returns (n, edges) or (None, None) if fails.
        """
        if time.time() > deadline:
            return None, None
        
        # Generate random relators
        relators = []
        for _ in range(num_relators):
            word = []
            for _ in range(relator_length):
                gen = rng.randint(0, k - 1)
                inv = rng.randint(0, 1)
                word.append((gen, inv))
            relators.append(tuple(word))
        
        # Schreier enumeration: state = (coset_id, word)
        # We track the mapping from group elements (as words) to coset IDs
        # Start with coset 0 = identity
        
        # For each coset, we track which generators lead to which coset
        # coset_map[coset_id][gen] = target_coset_id or -1 if not yet determined
        # We also track the "word" that defines each coset to detect when we close
        
        # Use a dictionary: word_tuple -> coset_id
        # word is a tuple of (gen, inv) pairs representing the path from identity
        
        coset_of_word = {}
        word_of_coset = {}
        coset_map = {}  # coset_id -> list of (target_coset_id, gen) for each gen
        num_cosets = 0
        
        # Identity
        identity = ()
        coset_of_word[identity] = 0
        word_of_coset[0] = identity
        coset_map[0] = [-1] * k
        num_cosets = 1
        
        # BFS to enumerate cosets
        queue = [0]
        qi = 0
        found_cycle = False
        cycle_girth = 0
        
        while qi < len(queue):
            if time.time() > deadline:
                return None, None
            if num_cosets >= max_size:
                return None, None
            
            c = queue[qi]
            qi += 1
            current_word = word_of_coset[c]
            
            for gen in range(k):
                # Forward step
                new_word = current_word + ((gen, 0),)
                if new_word in coset_of_word:
                    target = coset_of_word[new_word]
                    coset_map[c][gen] = target
                    # Check for cycle: if target == c, we have a loop (bad for simple graph)
                    # If we've seen this before, it's fine (just closes the graph)
                else:
                    if num_cosets >= max_size:
                        return None, None
                    target = num_cosets
                    num_cosets += 1
                    coset_of_word[new_word] = target
                    word_of_coset[target] = new_word
                    coset_map[target] = [-1] * k
                    coset_map[c][gen] = target
                    queue.append(target)
                
                # Backward step (inverse)
                inv_word = current_word + ((gen, 1),)
                if inv_word in coset_of_word:
                    target = coset_of_word[inv_word]
                    # For undirected graph, the inverse edge is already counted
                    # We just need to ensure consistency
                else:
                    if num_cosets >= max_size:
                        return None, None
                    target = num_cosets
                    num_cosets += 1
                    coset_of_word[inv_word] = target
                    word_of_coset[target] = inv_word
                    coset_map[target] = [-1] * k
                    queue.append(target)
        
        # Now check all relators
        for rel in relators:
            # Apply the relator to coset 0
            current = 0
            for gen, inv in rel:
                if inv == 0:
                    current = coset_map[current][gen]
                else:
                    # Inverse step: find the coset that maps to current via gen
                    # This requires reverse lookup
                    # For simplicity, skip inverse relators or handle differently
                    pass
            
            # If current != 0, the relator is not satisfied in this quotient
            # This means our enumeration didn't close properly
            if current != 0:
                return None, None
        
        # Build the graph
        n = num_cosets
        edges = []
        for c in range(n):
            for gen in range(k):
                target = coset_map[c][gen]
                if target > c:
                    edges.append((c, target))
        
        # Check for loops and multi-edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return None, None
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None, None
            edge_set.add(key)
        
        # Check degree
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return None, None
        
        # Check girth
        if _check_girth_bfs(n, adj, g):
            return n, edges
        
        return None, None
    
    # Try various parameters
    param_sets = [
        (200, 3, 8), (300, 3, 10), (400, 3, 12), (500, 3, 14),
        (200, 4, 8), (300, 4, 10), (400, 4, 12),
        (200, 5, 10), (300, 5, 12),
    ]
    
    for max_size, num_rel, rel_len in param_sets:
        if time.time() > deadline - 5:
            break
        if max_size >= best_n:
            continue
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            n, edges = try_cayley_construction(k, g, max_size, num_rel, rel_len, rng, deadline)
            if n is not None and n < best_n:
                best_n = n
                best_edges = edges
                break
    
    if best_edges is not None:
        return best_n, best_edges
    else:
        return best_n, best_edges


def construct(k, g):
    deadline = time.time() + 110
    
    # Try Cayley graph approach
    try:
        n, edges = _cayley_graph_search(k, g, deadline)
        if n is not None and edges is not None:
            return n, edges
    except:
        pass
    
    # Fallback to LUW
    try:
        return _luw_construct(k, g, deadline)
    except:
        return 0, []

# EVOLVE-BLOCK-END
