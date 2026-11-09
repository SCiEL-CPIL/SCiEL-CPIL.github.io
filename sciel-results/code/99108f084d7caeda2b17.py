# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import Counter

def _generate_edges(n, p, seed):
    """Generate edges for a random graph using the same RNG as the verifier."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    """Build adjacency list and bitmasks."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # Bitmasks for fast intersection
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    return adj, adj_mask

def _bron_kerbosch_pivot(R, P, X, adj, adj_mask, cliques):
    """Bron-Kerbosch with pivot. Only records cliques of size >= 2."""
    if not P and not X:
        if len(R) >= 2:
            cliques.append(list(R))
        return
    
    # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
    union_px = P | X
    if not union_px:
        if len(R) >= 2:
            cliques.append(list(R))
        return
    
    # Find pivot with max degree in P
    pivot = None
    max_count = -1
    for u in union_px:
        count = len(P & adj[u])
        if count > max_count:
            max_count = count
            pivot = u
        if count == len(P):  # Can't do better
            break
    
    # Recurse on vertices in P \ N(pivot)
    candidates = P - adj[pivot] if pivot is not None else P
    
    for v in list(candidates):
        new_R = R | {v}
        new_P = P & adj[v]
        new_X = X & adj[v]
        P.remove(v)
        X.add(v)
        _bron_kerbosch_pivot(new_R, new_P, new_X, adj, adj_mask, cliques)
        P.add(v)
        X.remove(v)

def _enumerate_maximal_cliques(n, edges):
    """Enumerate all maximal cliques of size >= 2."""
    adj, adj_mask = _build_adjacency(n, edges)
    cliques = []
    _bron_kerbosch_pivot(set(), set(range(n)), set(), adj, adj_mask, cliques)
    return cliques

def _greedy_hitting_set(cliques, n):
    """Greedy set cover: repeatedly pick the vertex covering the most uncovered cliques."""
    # Count how many cliques each vertex is in
    vertex_clique_count = Counter()
    for clique in cliques:
        for v in clique:
            vertex_clique_count[v] += 1
    
    # Track which cliques are covered
    uncovered = set(range(len(cliques)))
    selected = set()
    
    # For each vertex, track which cliques it covers
    vertex_cliques = {v: set() for v in vertex_clique_count}
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].add(i)
    
    while uncovered:
        # Find vertex covering most uncovered cliques
        best_v = None
        best_count = -1
        for v in vertex_clique_count:
            count = len(vertex_cliques[v] & uncovered)
            if count > best_count:
                best_count = count
                best_v = v
            elif count == best_count and best_v is not None:
                # Tie-break: prefer vertex with fewer total cliques (less "wasteful")
                if vertex_clique_count[v] < vertex_clique_count[best_v]:
                    best_v = v
        
        if best_v is None or best_count <= 0:
            break
        
        selected.add(best_v)
        uncovered -= vertex_cliques[best_v]
    
    return selected

def _is_valid_hitting_set(selected_mask, clique_masks):
    """Check if selected_mask intersects every clique_mask."""
    for cm in clique_masks:
        if (selected_mask & cm) == 0:
            return False
    return True

def _local_search_advanced(selected, cliques, n, time_limit=10.0):
    """
    Advanced local search using bitmask representations for speed.
    Includes 1-for-1 swaps and 2-for-1 swaps.
    """
    start_time = time.time()
    selected = set(selected)
    
    # Precompute clique masks
    clique_masks = [0] * len(cliques)
    for i, clique in enumerate(cliques):
        mask = 0
        for v in clique:
            mask |= (1 << v)
        clique_masks[i] = mask
    
    current_mask = 0
    for v in selected:
        current_mask |= (1 << v)
        
    current_size = len(selected)
    
    # Helper to get list of vertices in a mask
    def mask_to_list(mask):
        return [v for v in range(n) if (mask >> v) & 1]

    # Precompute which cliques each vertex touches (for redundancy check speedup)
    # vertex_to_clique_indices[v] = list of indices of cliques containing v
    vertex_to_clique_indices = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_to_clique_indices[v].append(i)

    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # 1. Try removing single vertices (Redundancy check)
        for v in list(selected):
            # Check if v is redundant
            # v is redundant if for all cliques c containing v, 
            # there exists another vertex u in selected \ {v} in c
            redundant = True
            for idx in vertex_to_clique_indices[v]:
                c_mask = clique_masks[idx]
                # Check if (c_mask & (current_mask ^ (1<<v))) is non-zero
                if (c_mask & (current_mask ^ (1 << v))) == 0:
                    redundant = False
                    break
            if redundant:
                selected.remove(v)
                current_mask ^= (1 << v)
                current_size -= 1
                improved = True
        
        # 2. Try 1-for-1 swaps
        if time.time() - start_time > time_limit:
            break
        for v in list(selected):
            # Find cliques that would be uncovered if v is removed
            # These are cliques where v is the ONLY selected vertex
            critical_cliques = []
            for idx in vertex_to_clique_indices[v]:
                c_mask = clique_masks[idx]
                if (c_mask & (current_mask ^ (1 << v))) == 0:
                    critical_cliques.append(idx)
            
            if not critical_cliques:
                continue # v is redundant, handled above
            
            # Find a vertex w not in selected that covers all critical_cliques
            for w in range(n):
                if w in selected:
                    continue
                # Check if w covers all critical cliques
                covers_all = True
                for idx in critical_cliques:
                    if (clique_masks[idx] & (1 << w)) == 0:
                        covers_all = False
                        break
                if covers_all:
                    # Perform swap: remove v, add w
                    # Since w covers all critical cliques of v, and v only covers those 
                    # (plus cliques where it wasn't unique), the new set is valid.
                    # Wait, we must ensure w doesn't break anything? 
                    # Removing v only breaks cliques where v was unique. 
                    # Adding w fixes those. 
                    # Does adding w break anything? No, adding vertices never breaks validity.
                    selected.remove(v)
                    selected.add(w)
                    current_mask ^= (1 << v)
                    current_mask |= (1 << w)
                    # Size remains same, so we don't mark 'improved' as True for size reduction,
                    # but we do it to escape local optima? 
                    # Actually, 1-for-1 swap doesn't reduce size. 
                    # The goal is to minimize size. 
                    # So 1-for-1 swaps are only useful if they enable future removals.
                    # We should probably limit these or use them in a stochastic way.
                    # For now, let's just do them if they lead to a state where a removal is possible.
                    # This is complex. Let's stick to size-reducing moves first.
                    # If we do 1-for-1, we might cycle. 
                    # Let's skip pure 1-for-1 swaps in this deterministic loop to avoid cycles,
                    # unless we implement a "tabu" or "random restart" strategy.
                    # Let's try 2-for-1 swaps which DO reduce size.
                    pass 

        # 3. Try 2-for-1 swaps (Remove v1, v2, Add w) -> Size -1
        if time.time() - start_time > time_limit:
            break
        selected_list = list(selected)
        # To save time, we can sample pairs or iterate all. n is small (40), so pairs are ~800.
        # For each pair, we need to find a w.
        # This is O(n^3) per iteration. 40^3 = 64000. Fast enough.
        
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                v1 = selected_list[i]
                v2 = selected_list[j]
                
                # Find cliques that are critically covered by v1 or v2
                # A clique is "critical" for {v1, v2} if it contains v1 or v2 
                # and does NOT contain any other vertex from selected \ {v1, v2}
                
                critical_cliques = set()
                mask_without_v1v2 = current_mask ^ (1 << v1) ^ (1 << v2)
                
                # Check cliques containing v1
                for idx in vertex_to_clique_indices[v1]:
                    c_mask = clique_masks[idx]
                    if (c_mask & mask_without_v1v2) == 0:
                        critical_cliques.add(idx)
                
                # Check cliques containing v2
                for idx in vertex_to_clique_indices[v2]:
                    c_mask = clique_masks[idx]
                    if (c_mask & mask_without_v1v2) == 0:
                        critical_cliques.add(idx)
                
                if not critical_cliques:
                    # Both v1 and v2 are redundant individually? 
                    # If so, they would have been removed in step 1.
                    # So this case shouldn't happen if step 1 converged.
                    continue
                
                # Find a vertex w not in selected that covers ALL critical_cliques
                for w in range(n):
                    if w in selected:
                        continue
                    
                    covers_all = True
                    for idx in critical_cliques:
                        if (clique_masks[idx] & (1 << w)) == 0:
                            covers_all = False
                            break
                    
                    if covers_all:
                        # Perform swap: remove v1, v2, add w
                        selected.remove(v1)
                        selected.remove(v2)
                        selected.add(w)
                        current_mask ^= (1 << v1)
                        current_mask ^= (1 << v2)
                        current_mask |= (1 << w)
                        current_size -= 1
                        improved = True
                        # Restart the pair search? Or just continue?
                        # Continuing might be fine, but the list selected_list is stale.
                        # Let's break out to restart the loop with the new state.
                        break
                if improved:
                    break
            if improved:
                break

    return selected

def construct():
    deadline = time.time() + 120  # Safety margin before 150s kill
    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Not enough time, return what we have
            break
        
        edges = _generate_edges(n, p, seed)
        cliques = _enumerate_maximal_cliques(n, edges)
        
        if not cliques:
            results.append([])
            continue
        
        # Greedy hitting set
        selected = _greedy_hitting_set(cliques, n)
        
        # Local search
        remaining_time = max(1.0, deadline - time.time() - 5)
        selected = _local_search_advanced(selected, cliques, n, time_limit=remaining_time)
        
        results.append(sorted(selected))
    
    # If we ran out of time, fill in with trivial solutions
    while len(results) < 3:
        n, p, seed = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))[len(results)]
        edges = _generate_edges(n, p, seed)
        touched = set()
        for u, v in edges:
            touched.add(u)
            touched.add(v)
        results.append(sorted(touched))
    
    return results
# EVOLVE-BLOCK-END
