import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 145.0

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []

    # Build bitmask adjacency for O(1) edge check
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    initial_edge_set = set(edges)
    best_partition = None
    best_count = float('inf')

    # We will try a few different greedy strategies with different random seeds for tie-breaking
    # to find a better partition.
    # Strategy:
    # 1. Maintain a set of active edges.
    # 2. Pick an edge (u, v).
    # 3. Expand to a maximal clique using a greedy heuristic.
    # 4. Remove all edges in the clique from the active set.
    # 5. Repeat.
    
    def _greedy_partition(active_edges, adj_mask, n, rng):
        # active_edges is a set of (u, v) with u < v
        partition = []
        # Convert to list for faster iteration if needed, but set is fine for removal.
        # We need to pick an edge. Picking a random one is okay.
        # To improve quality, we can pick an edge that is part of many triangles or has high degree.
        # Let's stick to a simple heuristic first: pick edge with max sum of degrees in active graph?
        # That requires updating degrees. 
        # Simpler: just pick a random edge from the set.
        
        while active_edges:
            if not _deadline_ok():
                break
                
            # Pick a random edge
            u, v = next(iter(active_edges))
            
            # Expand (u, v) to a maximal clique
            clique = [u, v]
            
            # Candidates: vertices connected to both u and v in the ORIGINAL graph
            # BUT they must also be connected to all existing clique members.
            # We can maintain a set of candidates.
            
            # Initial candidates: neighbors of u AND neighbors of v
            # Using bitmasks is fast.
            common = adj_mask[u] & adj_mask[v]
            common &= ~(1 << u) & ~(1 << v)
            
            # Convert mask to list of candidates
            candidates = []
            m = common
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                candidates.append(w)
                m ^= lsb
            
            # Greedy expansion:
            # While candidates exist:
            #   Pick candidate w that is connected to all other candidates? No, that's expensive.
            #   Pick candidate w that is connected to all current clique members (guaranteed by common)
            #   AND has max degree within the candidate set (in the original graph).
            #   Add w to clique.
            #   Remove from candidates all vertices not connected to w.
            
            while candidates:
                if not _deadline_ok():
                    break
                    
                # Find candidate with max degree in the subgraph induced by candidates
                # Note: We use original graph adjacency for this heuristic.
                best_w = None
                best_deg = -1
                
                # Optimization: if len(candidates) is small, just check all.
                for w in candidates:
                    # Count neighbors of w within candidates
                    # adj_mask[w] & mask_of_candidates
                    # We don't have a mask for candidates readily. 
                    # We can just count by checking adjacency to each other candidate.
                    # This is O(|C|^2) per step. 
                    # Better: maintain a bitmask for candidates?
                    # Let's just do the O(|C|^2) check since |C| is usually small in dense graphs 
                    # because the clique grows fast.
                    
                    deg = 0
                    for x in candidates:
                        if w == x: continue
                        if (adj_mask[w] >> x) & 1:
                            deg += 1
                            
                    if deg > best_deg:
                        best_deg = deg
                        best_w = w
                        
                if best_w is None:
                    break
                    
                clique.append(best_w)
                
                # Update candidates: remove best_w and any vertex not connected to best_w
                new_candidates = []
                for x in candidates:
                    if x == best_w:
                        continue
                    if (adj_mask[best_w] >> x) & 1:
                        new_candidates.append(x)
                candidates = new_candidates
                
            # We have a clique. Add to partition.
            partition.append(clique)
            
            # Remove all edges within this clique from active_edges
            for a, b in combinations(clique, 2):
                key = (min(a, b), max(a, b))
                active_edges.discard(key)
                
        return partition

    # Run multiple restarts with different random seeds for tie-breaking (though we used next(iter) which is deterministic for a set? 
    # Actually, set iteration order is arbitrary but consistent within a run. 
    # To get different results, we should shuffle the edge list or use a different selection strategy.
    # Let's modify the selection to use a random choice from a list of edges.
    
    def _greedy_partition_v2(active_edges_set, adj_mask, n, rng):
        active_edges = list(active_edges_set)
        partition = []
        
        while active_edges:
            if not _deadline_ok():
                break
                
            # Pick a random edge from the list
            idx = rng.randint(0, len(active_edges) - 1)
            u, v = active_edges[idx]
            
            # Check if this edge is still active (it might have been removed as part of a previous clique)
            # Since we are modifying the list, we should be careful.
            # Actually, it's safer to maintain a set and a list, or just use the set and pick a random item.
            # `next(iter(set))` is O(1). Let's stick to that but ensure we get variety.
            # The problem is `next(iter(set))` might always pick the same one if the set structure doesn't change much.
            # Let's use `rng.choice(list(active_edges_set))`? That is O(N) to create list.
            # For N=64, E~1000, it's fine.
            
            # Let's restart the loop logic with a slightly better selection:
            # Pick an edge that maximizes the number of common neighbors (triangles).
            # This requires calculating common neighbors for all edges? Too slow.
            # Let's just use random selection from the set.
            
            pass # This function is a stub, I will implement the logic in the main loop below.

    # Revised Greedy Partition
    def _run_greedy(active_edges_set, adj_mask, n, rng):
        partition = []
        # We need a way to pick edges. 
        # To ensure we don't get stuck in a local minimum caused by set ordering,
        # we can convert to a list, shuffle it, and pick from the front? 
        # Or just use rng.choice on a list of active edges.
        
        # For efficiency, let's keep a list of active edges and a set for O(1) removal check?
        # Actually, just using a set and `next(iter())` is fast. The "randomness" comes from the fact 
        # that the set changes. But to be safe, let's use a list.
        
        active_list = list(active_edges_set)
        
        while active_list:
            if not _deadline_ok():
                break
                
            # Pick a random edge
            idx = rng.randint(0, len(active_list) - 1)
            u, v = active_list[idx]
            
            # Expand to clique
            clique = [u, v]
            common = adj_mask[u] & adj_mask[v]
            common &= ~(1 << u) & ~(1 << v)
            
            candidates = []
            m = common
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                candidates.append(w)
                m ^= lsb
            
            while candidates:
                if not _deadline_ok():
                    break
                best_w = None
                best_deg = -1
                for w in candidates:
                    deg = 0
                    for x in candidates:
                        if w != x and (adj_mask[w] >> x) & 1:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_w = w
                        
                if best_w is None:
                    break
                    
                clique.append(best_w)
                
                new_candidates = []
                for x in candidates:
                    if x != best_w and (adj_mask[best_w] >> x) & 1:
                        new_candidates.append(x)
                candidates = new_candidates
                
            partition.append(clique)
            
            # Remove edges in clique from active_list
            # Create a set of edges to remove for O(1) lookup
            edges_to_remove = set()
            for a, b in combinations(clique, 2):
                edges_to_remove.add((min(a, b), max(a, b)))
                
            # Filter active_list
            # This is O(E) per clique. Total O(E * #cliques).
            # #cliques is roughly E / avg_clique_size.
            # So total is roughly E^2 / avg_size. For E=1000, E^2=10^6. This is fine.
            new_active_list = []
            for e in active_list:
                if e not in edges_to_remove:
                    new_active_list.append(e)
            active_list = new_active_list
            
        return partition

    # Try a few runs
    for run_seed in range(20):
        if not _deadline_ok():
            break
        rng = random.Random(seed * 1000 + run_seed)
        active_edges_set = set(initial_edge_set)
        partition = _run_greedy(active_edges_set, adj_mask, n, rng)
        
        # Validate
        valid = True
        seen_edges = set()
        for clique in partition:
            if len(clique) < 2:
                valid = False
                break
            for a, b in combinations(clique, 2):
                key = (min(a, b), max(a, b))
                if key in seen_edges:
                    valid = False
                    break
                seen_edges.add(key)
                if not ((adj_mask[a] >> b) & 1):
                    valid = False
                    break
            if not valid:
                break
        
        if valid and seen_edges == initial_edge_set:
            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition
        elif valid:
            # If we ran out of time, we might not have covered all edges.
            # We should only accept if all edges are covered.
            pass

    if best_partition is None:
        # Fallback: just return all edges as K2
        return [[u, v] for u, v in edges]

    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
