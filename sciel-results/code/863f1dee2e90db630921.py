import time
import random
import itertools

# Precompute Q3 edge sets on vertices 0..7
def generate_q3_edge_sets():
    # Standard Q3 edges on 0..7 (binary 000..111)
    base_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            xor = i ^ j
            if xor == 1 or xor == 2 or xor == 4:
                base_edges.append((i, j))
                
    # Generate all permutations of 0..7
    # For each perm, map base_edges to permuted edges
    q3_sets = set()
    for perm in itertools.permutations(range(8)):
        # perm[i] is the vertex in the graph corresponding to Q3 vertex i
        e_set = frozenset(
            (min(perm[i], perm[j]), max(perm[i], perm[j])) 
            for (i, j) in base_edges
        )
        q3_sets.add(e_set)
        
    return list(q3_sets)

Q3_EDGE_SETS = generate_q3_edge_sets()

def has_q3_adj(adj, n=16):
    """Full check: does the graph contain any Q3 subgraph?"""
    for vertices in itertools.combinations(range(n), 8):
        # Build induced edge bitmask on these 8 vertices
        # Map vertices to local indices 0..7
        induced_mask = 0
        for i in range(8):
            u = vertices[i]
            for j in range(i+1, 8):
                v = vertices[j]
                if adj[u] & (1 << v):
                    induced_mask |= (1 << (i*8 + j))
        
        # Check if any Q3 edge set is a subset
        for q3_set in Q3_EDGE_SETS:
            q3_mask = 0
            for (i, j) in q3_set:
                q3_mask |= (1 << (i*8 + j))
            if (q3_mask & induced_mask) == q3_mask:
                return True
    return False

def has_q3_with_edge(adj, new_edge, n=16):
    """Check if adding new_edge creates a Q3. Only check 8-sets containing both endpoints."""
    u, v = new_edge
    # Find all 8-vertex subsets that contain both u and v
    other_vertices = [x for x in range(n) if x != u and x != v]
    
    for subset in itertools.combinations(other_vertices, 6):
        vertices = list(subset) + [u, v]
        # Build induced edge bitmask
        induced_mask = 0
        for i in range(8):
            a = vertices[i]
            for j in range(i+1, 8):
                b = vertices[j]
                if adj[a] & (1 << b):
                    induced_mask |= (1 << (i*8 + j))
        
        # Check if any Q3 edge set is a subset
        for q3_set in Q3_EDGE_SETS:
            q3_mask = 0
            for (i, j) in q3_set:
                q3_mask |= (1 << (i*8 + j))
            if (q3_mask & induced_mask) == q3_mask:
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 16
    # Start with K3,13
    edges = set()
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
    best_edges = set(edges)
    best_count = len(edges)
    
    # Candidates: all edges not in best_edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    candidates = [e for e in all_possible if e not in best_edges]
    
    # Local search with random restarts and edge removal
    while time.time() < deadline:
        # Phase 1: Greedy edge addition
        random.shuffle(candidates)
        improved = False
        for e in candidates[:]:
            if time.time() > deadline:
                break
                
            u, v = e
            # Try adding e
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(adj, e, n):
                # Remove e
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Keep e
                edges.add(e)
                candidates.remove(e)
                best_count = len(edges)
                improved = True
                
        if not improved:
            # Phase 2: Try random edge removal + re-addition to escape local optima
            if edges and random.random() < 0.3:
                # Remove a random edge
                removed = random.choice(list(edges))
                edges.remove(removed)
                u, v = removed
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                candidates.append(removed)
                # Re-run greedy
                random.shuffle(candidates)
                for e in candidates[:]:
                    if time.time() > deadline:
                        break
                    u, v = e
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    if has_q3_with_edge(adj, e, n):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        edges.add(e)
                        candidates.remove(e)
                        best_count = len(edges)
            else:
                # Try a fresh random restart
                if random.random() < 0.1:
                    # Randomly initialize with some random edges
                    edges = set()
                    adj = [0] * n
                    random.shuffle(all_possible)
                    for e in all_possible[:20]:
                        u, v = e
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.add(e)
                    candidates = [e for e in all_possible if e not in edges]
                    best_count = len(edges)
                else:
                    break
    
    # Final verification
    if has_q3_adj(adj, n):
        # Shouldn't happen, but return a safe graph
        return [list(e) for e in best_edges]
        
    return [list(e) for e in edges]
