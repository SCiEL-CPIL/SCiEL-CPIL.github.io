import time
import random
import itertools

# Precompute Q3 edge sets on vertices 0..7
def generate_q3_edge_sets():
    base_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            xor = i ^ j
            if xor == 1 or xor == 2 or xor == 4:
                base_edges.append((i, j))
                
    q3_sets = set()
    for perm in itertools.permutations(range(8)):
        e_set = frozenset(
            (min(perm[i], perm[j]), max(perm[i], perm[j])) 
            for (i, j) in base_edges
        )
        q3_sets.add(e_set)
        
    return list(q3_sets)

Q3_EDGE_SETS = generate_q3_edge_sets()

# Precompute Q3 edge masks for fast subset checking
# Each mask is an integer where bit (i*8+j) is set if edge (i,j) is in the Q3 set
Q3_MASKS = []
for q3_set in Q3_EDGE_SETS:
    mask = 0
    for (i, j) in q3_set:
        mask |= (1 << (i*8 + j))
    Q3_MASKS.append(mask)

def has_q3_adj(adj, n=16):
    """Full check: does the graph contain any Q3 subgraph?"""
    for vertices in itertools.combinations(range(n), 8):
        induced_mask = 0
        for i in range(8):
            u = vertices[i]
            for j in range(i+1, 8):
                v = vertices[j]
                if adj[u] & (1 << v):
                    induced_mask |= (1 << (i*8 + j))
        
        for q3_mask in Q3_MASKS:
            if (q3_mask & induced_mask) == q3_mask:
                return True
    return False

def has_q3_with_edge(adj, new_edge, n=16):
    """Check if adding new_edge creates a Q3. Only check 8-sets containing both endpoints."""
    u, v = new_edge
    other_vertices = [x for x in range(n) if x != u and x != v]
    
    for subset in itertools.combinations(other_vertices, 6):
        vertices = list(subset) + [u, v]
        induced_mask = 0
        for i in range(8):
            a = vertices[i]
            for j in range(i+1, 8):
                b = vertices[j]
                if adj[a] & (1 << b):
                    induced_mask |= (1 << (i*8 + j))
        
        for q3_mask in Q3_MASKS:
            if (q3_mask & induced_mask) == q3_mask:
                return True
    return False

def count_q3(adj, n=16):
    """Count the number of Q3 subgraphs in the graph."""
    count = 0
    for vertices in itertools.combinations(range(n), 8):
        induced_mask = 0
        for i in range(8):
            u = vertices[i]
            for j in range(i+1, 8):
                v = vertices[j]
                if adj[u] & (1 << v):
                    induced_mask |= (1 << (i*8 + j))
        
        for q3_mask in Q3_MASKS:
            if (q3_mask & induced_mask) == q3_mask:
                count += 1
                break  # Count each 8-set at most once
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 16
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
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
    
    candidates = [e for e in all_possible if e not in best_edges]
    
    # Simulated annealing approach
    # State: edges (set), adj (list of bitmasks)
    # Objective: maximize edge count, subject to Q3-free constraint
    # Use penalty for Q3 count to allow temporary violations
    
    current_edges = set(edges)
    current_adj = adj[:]
    current_count = len(edges)
    
    # Temperature schedule
    t0 = 100.0
    t1 = 0.1
    # We'll run for a fixed number of iterations or until deadline
    max_iterations = 50000
    
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Temperature annealing
        progress = iteration / max_iterations
        temperature = t0 * (t1 / t0) ** progress
        
        # Choose a random move: add edge, remove edge, or swap
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random candidate edge
            if not candidates:
                continue
            e = random.choice(candidates)
            u, v = e
            
            # Add edge
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            current_edges.add(e)
            candidates.remove(e)
            
            # Check Q3
            q3_count = count_q3(current_adj, n)
            new_count = len(current_edges)
            
            if q3_count == 0:
                # Accept if Q3-free
                if new_count > best_count:
                    best_count = new_count
                    best_edges = set(current_edges)
            else:
                # Accept with probability exp(-penalty / temperature)
                penalty = 100 * q3_count
                delta = -penalty  # Objective decreases
                if random.random() < min(1.0, exp(-delta / temperature)):
                    # Accept
                    if new_count > best_count and q3_count == 0:
                        best_count = new_count
                        best_edges = set(current_edges)
                else:
                    # Reject: undo
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_edges.remove(e)
                    candidates.append(e)
                    
        elif move_type < 0.8:
            # Try to remove a random edge
            if not current_edges:
                continue
            e = random.choice(list(current_edges))
            u, v = e
            
            # Remove edge
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_edges.remove(e)
            candidates.append(e)
            
            # Always accept removal (it can't create a Q3)
            # But this reduces edge count, so only useful for escape
            
        else:
            # Try a swap: remove one edge, add another
            if not current_edges or not candidates:
                continue
            e_remove = random.choice(list(current_edges))
            e_add = random.choice(candidates)
            
            u1, v1 = e_remove
            u2, v2 = e_add
            
            # Remove first
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_edges.remove(e_remove)
            candidates.append(e_remove)
            
            # Add second
            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            current_edges.add(e_add)
            candidates.remove(e_add)
            
            q3_count = count_q3(current_adj, n)
            new_count = len(current_edges)
            
            if q3_count == 0:
                if new_count > best_count:
                    best_count = new_count
                    best_edges = set(current_edges)
            else:
                penalty = 100 * q3_count
                if random.random() < min(1.0, exp(-penalty / temperature)):
                    pass  # Accept
                else:
                    # Undo both
                    current_adj[u2] &= ~(1 << v2)
                    current_adj[v2] &= ~(1 << u2)
                    current_edges.remove(e_add)
                    candidates.append(e_add)
                    
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
                    current_edges.add(e_remove)
                    candidates.remove(e_remove)
    
    # Final greedy pass to polish
    random.shuffle(candidates)
    for e in candidates[:]:
        if time.time() > deadline - 5:
            break
        u, v = e
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        
        if has_q3_with_edge(current_adj, e, n):
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        else:
            current_edges.add(e)
            candidates.remove(e)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = set(current_edges)
    
    # Final verification
    if has_q3_adj(best_edges_to_adj(best_edges, n), n):
        # Shouldn't happen, but return a safe graph
        return [list(e) for e in edges]
        
    return [list(e) for e in best_edges]

def best_edges_to_adj(edge_set, n):
    adj = [0] * n
    for (u, v) in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def exp(x):
    import math
    return math.exp(x)
