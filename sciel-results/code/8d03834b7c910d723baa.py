import random
import time
import math

N = 18
DEADLINE = 112.0  # Safety margin before 120s limit

def _edge_set_to_edges(es):
    return [[u, v] for (u, v) in sorted(es)]

def _build_adj_mask(es):
    """Return adjacency as list of bitmasks."""
    adj = [0] * N
    for u, v in es:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free_mask(adj):
    """Check K4-free using adjacency masks. Fast: O(n^2 * degree)."""
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                # Check if any two vertices in common are adjacent
                mask = common
                while mask:
                    lb = mask & (-mask)
                    w1 = lb.bit_length() - 1
                    mask ^= lb
                    if adj[w1] & mask:
                        return False
    return True

def _largest_triangle_free_mask(adj):
    """
    Find largest triangle-free induced subgraph using branch-and-bound with bitmasks.
    Returns the size and the mask of the set.
    """
    best = [0, 0]  # [size, mask]
    
    def bnb(candidates, current_size, current_mask):
        """candidates: bitmask of vertices still to consider, current_mask: vertices chosen so far"""
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
                best[1] = current_mask
            return
        
        # Upper bound
        rem = candidates.bit_count()
        if current_size + rem <= best[0]:
            return
        
        # Pick a vertex to branch on
        # Heuristic: Pick vertex with highest degree in the induced subgraph of candidates?
        # Or just lowest index for determinism in search, but randomization helps.
        # Let's pick the vertex with the most neighbors in `candidates` to prune earlier?
        # Actually, standard B&B often picks first available. 
        # Let's stick to a simple heuristic: pick vertex with max degree in remaining.
        # But calculating degree for all is O(N). N=18 is small.
        
        # Simple pick: lowest bit
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        
        # Option 1: exclude v
        bnb(rest, current_size, current_mask)
        
        # Option 2: include v
        # Check: adding v to current_mask must not create a triangle
        N_v = adj[v] & current_mask
        # Check if N_v is independent
        mask = N_v
        independent = True
        while mask:
            lb2 = mask & (-mask)
            u = lb2.bit_length() - 1
            mask ^= lb2
            if adj[u] & mask:
                independent = False
                break
        
        if independent:
            # Restrict future candidates
            forbidden = 0
            m = N_v
            while m:
                lb3 = m & (-m)
                u = lb3.bit_length() - 1
                m ^= lb3
                forbidden |= (adj[u] & adj[v])
            new_rest = rest & ~forbidden
            bnb(new_rest, current_size + 1, current_mask | lb)
            
    bnb((1 << N) - 1, 0, 0)
    return best[0], best[1]

def _initial_graph(seed):
    """Start with a random K4-free graph."""
    random.seed(seed)
    es = set()
    # Try to build a dense random K4-free graph
    # Start empty, add random edges if K4-free
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_edges)
    adj = [0]*N
    for u, v in all_edges:
        # Check if adding (u,v) creates K4
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        # Check if common neighbors contain an edge
        valid = True
        mask = common
        while mask:
            lb = mask & (-mask)
            w1 = lb.bit_length() - 1
            mask ^= lb
            if adj[w1] & mask:
                valid = False
                break
        if valid:
            es.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return es

def _can_add_edge(es, adj, u, v):
    """Check if adding edge (u,v) keeps graph K4-free."""
    if (u, v) in es or (v, u) in es:
        return False
    common = adj[u] & adj[v]
    mask = common
    while mask:
        lb = mask & (-mask)
        w1 = lb.bit_length() - 1
        mask ^= lb
        if adj[w1] & mask:
            return False
    return True

def construct():
    start_time = time.time()
    
    best_score = N
    best_es = set()
    
    # Multiple restarts to escape local minima
    max_restarts = 20
    restart_time_budget = 5.0 # seconds per restart roughly
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        seed = 12345 + restart * 100
        es = _initial_graph(seed)
        adj = _build_adj_mask(es)
        
        current_score, current_mask = _largest_triangle_free_mask(adj)
        
        if current_score < best_score:
            best_score = current_score
            best_es = set(es)
            
        # Local Search Phase
        stagnation_count = 0
        last_improvement_time = time.time()
        
        while time.time() - start_time < DEADLINE - 5:
            # Check stagnation
            if time.time() - last_improvement_time > 3.0:
                break # Restart with new seed
            
            # 1. Find the largest triangle-free set S
            # We already have current_score and current_mask from the last evaluation
            # But if we made a move, we need to re-evaluate.
            # To save time, we only re-evaluate if we accept a move.
            
            # Targeted Move: Try to add an edge within S
            # S is current_mask
            improved = False
            vertices_in_S = []
            m = current_mask
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                vertices_in_S.append(v)
                m ^= lb
                
            # Try all pairs in S
            random.shuffle(vertices_in_S) # Randomize order to avoid bias
            for i in range(len(vertices_in_S)):
                for j in range(i+1, len(vertices_in_S)):
                    u = vertices_in_S[i]
                    v = vertices_in_S[j]
                    
                    # If u, v are already connected, skip
                    if adj[u] & (1 << v):
                        continue
                    
                    # Check if adding (u,v) is K4-free
                    if _can_add_edge(es, adj, u, v):
                        # Tentatively add
                        es.add((u, v))
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                        new_score, new_mask = _largest_triangle_free_mask(adj)
                        
                        if new_score < current_score:
                            current_score = new_score
                            current_mask = new_mask
                            improved = True
                            last_improvement_time = time.time()
                            if current_score < best_score:
                                best_score = current_score
                                best_es = set(es)
                            break # Restart inner loop with new S
                        else:
                            # Revert
                            es.discard((u, v))
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                
                if improved:
                    break
            
            if improved:
                continue
            
            # 2. If no targeted move worked, try random diversification
            # Try adding a random edge outside S or removing an edge
            if random.random() < 0.7:
                # Add random edge
                all_possible = [(i, j) for i in range(N) for j in range(i+1, N) if not (adj[i] & (1 << j))]
                if all_possible:
                    u, v = random.choice(all_possible)
                    if _can_add_edge(es, adj, u, v):
                        es.add((u, v))
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                        new_score, new_mask = _largest_triangle_free_mask(adj)
                        # Accept if better or equal (to allow plateau walking)
                        if new_score <= current_score:
                            current_score = new_score
                            current_mask = new_mask
                            if new_score < best_score:
                                best_score = new_score
                                best_es = set(es)
                                last_improvement_time = time.time()
                        else:
                            # Revert if worse, unless we allow SA-style acceptance
                            # For now, keep it strict to ensure progress, but this might get stuck.
                            # Let's allow some worsening with small probability
                            if random.random() < 0.1:
                                current_score = new_score
                                current_mask = new_mask
                            else:
                                es.discard((u, v))
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
            else:
                # Remove random edge
                if es:
                    u, v = random.choice(list(es))
                    es.discard((u, v))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    new_score, new_mask = _largest_triangle_free_mask(adj)
                    if new_score <= current_score:
                        current_score = new_score
                        current_mask = new_mask
                        if new_score < best_score:
                            best_score = new_score
                            best_es = set(es)
                            last_improvement_time = time.time()
                    else:
                        # Revert
                        es.add((u, v))
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

    return _edge_set_to_edges(best_es)
