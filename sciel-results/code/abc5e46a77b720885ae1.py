# EVOLVE-BLOCK-START
import time
from random import Random

def _random_graph_edges(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch_bitmask(adj_masks, n):
    """Return all maximal cliques as frozensets of vertices using bitmask representation."""
    cliques = []
    
    def bk(R_mask, P_mask, X_mask):
        if P_mask == 0 and X_mask == 0:
            if R_mask != 0:
                # Convert bitmask to frozenset
                R_set = frozenset(i for i in range(n) if (R_mask >> i) & 1)
                if len(R_set) >= 2:
                    cliques.append(R_set)
            return
        
        # Pivot selection: pick vertex in P∪X with max |P ∩ N(u)|
        union = P_mask | X_mask
        if union == 0:
            return
        
        # Find pivot with maximum |P ∩ N(u)|
        best_pivot = -1
        best_count = -1
        u = 0
        while (1 << u) <= union:
            if (union >> u) & 1:
                count = bin(P_mask & adj_masks[u]).count('1')
                if count > best_count:
                    best_count = count
                    best_pivot = u
            u += 1
        
        if best_pivot == -1:
            return
        
        candidates = P_mask & ~adj_masks[best_pivot]
        
        v = 0
        while (1 << v) <= candidates:
            if (candidates >> v) & 1:
                bk(R_mask | (1 << v), P_mask & adj_masks[v], X_mask & adj_masks[v])
                P_mask &= ~(1 << v)
                X_mask |= (1 << v)
            v += 1
    
    bk(0, (1 << n) - 1, 0)
    return cliques

def _find_min_hitting_set(vertices, cliques, timeout=40):
    """Find minimum hitting set for cliques using bitmask operations and local search."""
    if not cliques:
        return set()
    
    n = len(vertices)
    n_cliques = len(cliques)
    
    # Precompute for each clique, which vertices are in it (as bitmask)
    clique_masks = []
    for c in cliques:
        mask = 0
        for v in c:
            mask |= (1 << v)
        clique_masks.append(mask)
    
    # For each vertex, precompute which cliques it covers (as bitmask over cliques)
    # vertex_clique_mask[v] has bit ci set if vertex v is in clique ci
    vertex_clique_mask = [0] * n
    for ci, cm in enumerate(clique_masks):
        v = 0
        while (1 << v) <= cm:
            if (cm >> v) & 1:
                vertex_clique_mask[v] |= (1 << ci)
            v += 1
    
    def is_valid(chosen_mask):
        """Check if chosen vertices hit all cliques."""
        for ci, cm in enumerate(clique_masks):
            if (chosen_mask & cm) == 0:
                return False
        return True
    
    def count_covered(chosen_mask):
        """Count how many cliques are covered by chosen vertices."""
        covered_mask = 0
        for v in range(n):
            if (chosen_mask >> v) & 1:
                covered_mask |= vertex_clique_mask[v]
        return bin(covered_mask).count('1')
    
    def greedy_init():
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        chosen_mask = 0
        uncovered_mask = (1 << n_cliques) - 1  # all cliques uncovered
        
        while uncovered_mask != 0:
            best_v = -1
            best_new_coverage = 0
            
            for v in range(n):
                if (chosen_mask >> v) & 1:
                    continue
                # How many currently uncovered cliques does v cover?
                new_cov = bin(uncovered_mask & vertex_clique_mask[v]).count('1')
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_v = v
            
            if best_v == -1 or best_new_coverage == 0:
                break
            
            chosen_mask |= (1 << best_v)
            uncovered_mask &= ~vertex_clique_mask[best_v]
        
        return chosen_mask
    
    chosen_mask = greedy_init()
    
    # Fix any uncovered cliques
    for ci, cm in enumerate(clique_masks):
        if (chosen_mask & cm) == 0:
            # Pick vertex in this clique that covers most other uncovered cliques
            best_v = -1
            best_score = -1
            v = 0
            while (1 << v) <= cm:
                if (cm >> v) & 1:
                    # Count how many cliques v covers that aren't already covered
                    score = bin(vertex_clique_mask[v]).count('1')
                    if score > best_score:
                        best_score = score
                        best_v = v
                v += 1
            if best_v != -1:
                chosen_mask |= (1 << best_v)
    
    # Local search phase
    deadline = time.time() + timeout
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try single removals
        for v in range(n):
            if time.time() > deadline:
                break
            if (chosen_mask >> v) & 1:
                test_mask = chosen_mask & ~(1 << v)
                if is_valid(test_mask):
                    chosen_mask = test_mask
                    improved = True
        
        # Try single swaps: remove one, add one
        if not improved and time.time() < deadline:
            for v_remove in range(n):
                if time.time() > deadline:
                    break
                if (chosen_mask >> v_remove) & 1:
                    test_mask = chosen_mask & ~(1 << v_remove)
                    # Find uncovered cliques after removal
                    uncovered = 0
                    for ci, cm in enumerate(clique_masks):
                        if (test_mask & cm) == 0:
                            uncovered |= (1 << ci)
                    
                    if uncovered == 0:
                        chosen_mask = test_mask
                        improved = True
                        break
                    
                    # Try adding one vertex to cover all uncovered
                    best_add = -1
                    best_cov = 0
                    for v_add in range(n):
                        if (test_mask >> v_add) & 1:
                            continue
                        cov = bin(uncovered & vertex_clique_mask[v_add]).count('1')
                        if cov > best_cov:
                            best_cov = cov
                            best_add = v_add
                    
                    if best_add != -1 and best_cov > 0:
                        test2_mask = test_mask | (1 << best_add)
                        if is_valid(test2_mask) and bin(test2_mask).count('1') < bin(chosen_mask).count('1'):
                            chosen_mask = test2_mask
                            improved = True
                            break
        
        # Try 2-opt: remove two, add one or two
        if not improved and time.time() < deadline:
            for v1 in range(n):
                if time.time() > deadline:
                    break
                if (chosen_mask >> v1) & 1:
                    for v2 in range(v1 + 1, n):
                        if time.time() > deadline:
                            break
                        if (chosen_mask >> v2) & 1:
                            test_mask = chosen_mask & ~(1 << v1) & ~(1 << v2)
                            # Find uncovered cliques
                            uncovered = 0
                            for ci, cm in enumerate(clique_masks):
                                if (test_mask & cm) == 0:
                                    uncovered |= (1 << ci)
                            
                            if uncovered == 0:
                                chosen_mask = test_mask
                                improved = True
                                break
                            
                            # Try adding one vertex
                            best_add = -1
                            best_cov = 0
                            for v_add in range(n):
                                if (test_mask >> v_add) & 1:
                                    continue
                                cov = bin(uncovered & vertex_clique_mask[v_add]).count('1')
                                if cov > best_cov:
                                    best_cov = cov
                                    best_add = v_add
                            
                            if best_add != -1 and best_cov > 0:
                                test2_mask = test_mask | (1 << best_add)
                                if is_valid(test2_mask) and bin(test2_mask).count('1') < bin(chosen_mask).count('1'):
                                    chosen_mask = test2_mask
                                    improved = True
                                    break
                            
                            if improved:
                                break
        
        # Final removal pass after any improvement
        if improved:
            for v in range(n):
                if time.time() > deadline:
                    break
                if (chosen_mask >> v) & 1:
                    test_mask = chosen_mask & ~(1 << v)
                    if is_valid(test_mask):
                        chosen_mask = test_mask
    
    # Convert bitmask back to set
    result = set()
    for v in range(n):
        if (chosen_mask >> v) & 1:
            result.add(v)
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append(list(range(n)))
            continue
        
        edges = _random_graph_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        
        # Build adjacency as bitmasks
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        cliques = _bron_kerbosch_bitmask(adj_masks, n)
        
        vertices = list(range(n))
        hitting = _find_min_hitting_set(vertices, cliques, timeout=30)
        
        # Ensure validity
        if not all(hitting & c for c in cliques):
            # Fallback: add missing
            for c in cliques:
                if not hitting & c:
                    hitting.add(min(c))
        
        out.append(sorted(hitting))
    
    return out
# EVOLVE-BLOCK-END
