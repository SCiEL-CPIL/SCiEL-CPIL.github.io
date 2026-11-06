import random
import time
import itertools

def construct():
    """
    Constructs a graph with no K5 and no independent set of size 5.
    Uses bitmask-based adjacency with simulated annealing and greedy repair.
    """
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    # Fallback: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    paley_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Try to find graphs with more vertices
    for n in range(42, 37, -1):
        if time.time() > deadline - 5:
            break
        result = find_graph_bitmask(n, deadline)
        if result is not None:
            return result
            
    return q, paley_edges

def find_graph_bitmask(n, deadline):
    """
    Attempts to find a (5,5)-free graph on n vertices using bitmask-based SA.
    Returns (n, edges) if successful, else None.
    """
    # Precompute all 5-combinations for faster iteration
    all_5sets = list(itertools.combinations(range(n), 5))
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize random graph using bitmasks
        masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        
        # Local search with simulated annealing
        T = 2.0
        cooling = 0.995
        min_T = 0.05
        no_improve_count = 0
        best_cost = get_cost_bitmask(masks, n, all_5sets)
        current_cost = best_cost
        
        if current_cost == 0:
            return extract_edges_from_masks(n, masks)
        
        while time.time() < deadline - 10:
            if current_cost == 0:
                return extract_edges_from_masks(n, masks)
            
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Calculate delta cost efficiently
            val = (masks[u] >> v) & 1
            delta = calc_delta_bitmask(masks, u, v, val, n, all_5sets)
            
            # Acceptance criterion
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                # Flip the edge
                if val:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                else:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            # Occasional greedy repair: find a violation and fix it
            if no_improve_count > 100:
                viol = find_violation_bitmask(masks, n, all_5sets)
                if viol is not None:
                    (v0, v1, v2, v3, v4), is_clique = viol
                    vertices = [v0, v1, v2, v3, v4]
                    # Pick a random edge in this violating set to flip
                    u2, v2 = random.choice([(vertices[a], vertices[b]) for a in range(5) for b in range(a + 1, 5)])
                    val2 = (masks[u2] >> v2) & 1
                    delta2 = calc_delta_bitmask(masks, u2, v2, val2, n, all_5sets)
                    if val2:
                        masks[u2] &= ~(1 << v2)
                        masks[v2] &= ~(1 << u2)
                    else:
                        masks[u2] |= (1 << v2)
                        masks[v2] |= (1 << u2)
                    current_cost += delta2
                    if current_cost < best_cost:
                        best_cost = current_cost
                    no_improve_count = 0
    
    return None

def get_cost_bitmask(masks, n, all_5sets):
    """Calculate total cost (#K5 + #independent 5-sets) using bitmasks."""
    count = 0
    for combo in all_5sets:
        i, j, k, l, m = combo
        # Check clique: all 10 edges must exist
        if ((masks[i] >> j) & 1) and ((masks[i] >> k) & 1) and ((masks[i] >> l) & 1) and ((masks[i] >> m) & 1) and \
           ((masks[j] >> k) & 1) and ((masks[j] >> l) & 1) and ((masks[j] >> m) & 1) and \
           ((masks[k] >> l) & 1) and ((masks[k] >> m) & 1) and ((masks[l] >> m) & 1):
            count += 1
        # Check independent: all 10 edges must NOT exist
        if not ((masks[i] >> j) & 1) and not ((masks[i] >> k) & 1) and not ((masks[i] >> l) & 1) and not ((masks[i] >> m) & 1) and \
           not ((masks[j] >> k) & 1) and not ((masks[j] >> l) & 1) and not ((masks[j] >> m) & 1) and \
           not ((masks[k] >> l) & 1) and not ((masks[k] >> m) & 1) and not ((masks[l] >> m) & 1):
            count += 1
    return count

def find_violation_bitmask(masks, n, all_5sets):
    """Find one violation (K5 or independent 5-set) using bitmasks."""
    for combo in all_5sets:
        i, j, k, l, m = combo
        # Check clique
        if ((masks[i] >> j) & 1) and ((masks[i] >> k) & 1) and ((masks[i] >> l) & 1) and ((masks[i] >> m) & 1) and \
           ((masks[j] >> k) & 1) and ((masks[j] >> l) & 1) and ((masks[j] >> m) & 1) and \
           ((masks[k] >> l) & 1) and ((masks[k] >> m) & 1) and ((masks[l] >> m) & 1):
            return combo, True
        # Check independent
        if not ((masks[i] >> j) & 1) and not ((masks[i] >> k) & 1) and not ((masks[i] >> l) & 1) and not ((masks[i] >> m) & 1) and \
           not ((masks[j] >> k) & 1) and not ((masks[j] >> l) & 1) and not ((masks[j] >> m) & 1) and \
           not ((masks[k] >> l) & 1) and not ((masks[k] >> m) & 1) and not ((masks[l] >> m) & 1):
            return combo, False
    return None

def calc_delta_bitmask(masks, u, v, val, n, all_5sets):
    """
    Calculate the change in cost when flipping edge (u, v).
    Only 5-sets containing both u and v are affected.
    """
    # Find all 5-sets containing u and v
    delta = 0
    others = [x for x in range(n) if x != u and x != v]
    
    for a, b, c in itertools.combinations(others, 3):
        S = (u, v, a, b, c)
        
        # Current state
        is_clique_now = True
        is_ind_now = True
        
        # Check all 10 edges in S
        edges = [(u, v), (u, a), (u, b), (u, c), (v, a), (v, b), (v, c), (a, b), (a, c), (b, c)]
        for x, y in edges:
            e = (masks[x] >> y) & 1
            if e:
                is_ind_now = False
            else:
                is_clique_now = False
        
        cost_now = (1 if is_clique_now else 0) + (1 if is_ind_now else 0)
        
        # New state (flip u,v)
        is_clique_new = True
        is_ind_new = True
        
        for x, y in edges:
            if (x, y) == (u, v) or (x, y) == (v, u):
                e = 1 - val  # flipped
            else:
                e = (masks[x] >> y) & 1
            if e:
                is_ind_new = False
            else:
                is_clique_new = False
        
        cost_new = (1 if is_clique_new else 0) + (1 if is_ind_new else 0)
        delta += cost_new - cost_now
    
    return delta

def extract_edges_from_masks(n, masks):
    """Extract edge list from bitmask representation."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (masks[i] >> j) & 1:
                edges.append((i, j))
    return n, edges
