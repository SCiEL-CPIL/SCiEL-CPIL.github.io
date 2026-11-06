import time
import random
from itertools import combinations

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj):
    n = len(adj)
    # Convert to bitmasks for speed
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    
    max_cliques = []
    
    def expand(R_mask, P_mask, X_mask):
        if P_mask == 0 and X_mask == 0:
            if R_mask != 0:
                # Extract vertices
                verts = []
                m = R_mask
                while m:
                    lsb = m & (-m)
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                if len(verts) >= 2:
                    max_cliques.append(frozenset(verts))
            return
        # Choose pivot to minimize branching
        # Try each vertex in P
        # Simple approach: iterate
        P_list = []
        m = P_mask
        while m:
            lsb = m & (-m)
            P_list.append(lsb.bit_length() - 1)
            m ^= lsb
        
        for v in P_list:
            v_bit = 1 << v
            if not (P_mask & v_bit):
                continue
            new_R = R_mask | v_bit
            new_P = P_mask & adj_mask[v]
            new_X = X_mask & adj_mask[v]
            expand(new_R, new_P, new_X)
            P_mask &= ~v_bit
            X_mask |= v_bit
    
    expand(0, (1 << n) - 1, 0)
    return max_cliques

def _greedy_transversal(cliques, n, rng):
    """Greedy set cover: repeatedly pick vertex in most uncovered cliques."""
    if not cliques:
        return set()
    
    # Count how many cliques each vertex is in
    cliques_list = list(cliques)
    # For each vertex, list of clique indices it covers
    vertex_to_cliques = {v: [] for v in range(n)}
    for i, c in enumerate(cliques_list):
        for v in c:
            vertex_to_cliques[v].append(i)
    
    uncovered = set(range(len(cliques_list)))
    cover = set()
    
    while uncovered:
        best_v = -1
        best_count = -1
        for v in range(n):
            if v in cover:
                continue
            count = 0
            for ci in vertex_to_cliques[v]:
                if ci in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_v = v
        if best_v == -1 or best_count == 0:
            break
        cover.add(best_v)
        for ci in vertex_to_cliques[best_v]:
            if ci in uncovered:
                uncovered.discard(ci)
    
    return cover

def _local_search(cover, cliques, n, rng, max_iters=5000):
    """Try to improve the transversal via removal and swap operations."""
    cover = set(cover)
    cliques_list = list(cliques)
    num_cliques = len(cliques_list)
    
    if num_cliques == 0:
        return cover
    
    # Precompute for each vertex which cliques it hits
    vertex_hits = {v: set() for v in range(n)}
    for i, c in enumerate(cliques_list):
        for v in c:
            vertex_hits[v].add(i)
    
    def is_valid(cov):
        for c in cliques_list:
            if not (c & cov):
                return False
        return True
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try removal: for each vertex in cover, check if removing it keeps validity
        for v in list(cover):
            new_cover = cover - {v}
            if is_valid(new_cover):
                cover = new_cover
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one vertex, add another
        for v in list(cover):
            # Find cliques that v is the only cover for
            only_v_cliques = set()
            for ci in vertex_hits[v]:
                c = cliques_list[ci]
                if c & (cover - {v}):
                    continue
                # v is the only one in cover hitting this clique
                only_v_cliques.add(ci)
            
            if not only_v_cliques:
                # v can be removed freely, but we already tried that
                continue
            
            # Try adding a new vertex u not in cover that covers all only_v_cliques
            for u in range(n):
                if u in cover:
                    continue
                if not (only_v_cliques <= vertex_hits[u]):
                    continue
                new_cover = (cover - {v}) | {u}
                if len(new_cover) < len(cover):
                    cover = new_cover
                    improved = True
                    break
                elif len(new_cover) == len(cover):
                    # Same size, but might enable further improvements
                    # Only accept with some probability to avoid cycles
                    if rng.random() < 0.1:
                        cover = new_cover
                        improved = True
                        break
            if improved:
                break
    
    return cover

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Fallback: return all vertices
            results.append(list(range(n)))
            continue
        
        adj = _generate_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj)
        
        if not cliques:
            results.append([])
            continue
        
        # Multiple greedy restarts with randomization
        best_cover = None
        best_size = n + 1
        rng = random.Random(seed + 1000)
        
        num_restarts = 10
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Greedy with slight randomization
            cliques_list = list(cliques)
            vertex_to_cliques = {v: [] for v in range(n)}
            for i, c in enumerate(cliques_list):
                for v in c:
                    vertex_to_cliques[v].append(i)
            
            uncovered = set(range(len(cliques_list)))
            cover = set()
            
            while uncovered:
                best_v = -1
                best_count = -1
                best_key = None
                candidates = []
                for v in range(n):
                    if v in cover:
                        continue
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                    if count > 0:
                        # Add random tie-breaking
                        key = (-count, rng.random(), v)
                        if not candidates or key < candidates[0][0]:
                            candidates = [(key, v)]
                        elif len(candidates) < 5 and key > candidates[0][0]:
                            candidates.append((key, v))
                            candidates.sort(key=lambda x: x[0])
                            if len(candidates) > 5:
                                candidates = candidates[:5]
                
                if not candidates:
                    break
                best_v = candidates[0][1]
                cover.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    if ci in uncovered:
                        uncovered.discard(ci)
            
            # Local search
            cover = _local_search(cover, cliques, n, rng, max_iters=2000)
            
            if len(cover) < best_size:
                best_size = len(cover)
                best_cover = set(cover)
            
            # Early exit if we found a small cover
            if best_size <= 3:
                break
        
        if best_cover is None:
            # Fallback
            best_cover = set(range(n))
        
        results.append(sorted(best_cover))
    
    return results
