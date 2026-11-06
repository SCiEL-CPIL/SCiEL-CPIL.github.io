import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def check_non_2colorable(n, edges):
        """Check if hypergraph is NOT 2-colorable using bitmask enumeration.
        Returns True if non-2-colorable (no valid coloring exists)."""
        if not edges:
            return False
        # Convert edges to masks
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # By symmetry, fix vertex 0 to color 0 (bit 0 = 0)
        # So we iterate over colorings of vertices 1..n-1
        # Full coloring mask: bit v is 1 if vertex v has color 1
        # Vertex 0 fixed to color 0, so bit 0 = 0
        for c in range(1 << (n - 1)):
            valid = True
            for em in edge_masks:
                # Edge is monochromatic if all its vertices have same color
                # (c & em) == 0 means all vertices in edge have color 0
                # (c & em) == em means all vertices in edge have color 1
                if (c & em) == 0 or (c & em) == em:
                    valid = False
                    break
            if valid:
                return False  # Found valid coloring, IS 2-colorable
        return True  # No valid coloring, NOT 2-colorable

    def greedy_cover(n, max_edges, deadline):
        """Greedy maximum-coverage: select edges to forbid all 2-colorings.
        Returns list of edge tuples or None."""
        all_edges = list(combinations(range(n), 5))
        m_total = len(all_edges)
        if m_total < max_edges:
            max_edges = m_total

        # Precompute masks for all edges
        all_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            all_masks.append(mask)

        # Alive colorings: fix vertex 0 to color 0, so bits 1..n-1 vary
        # Total: 2^(n-1) colorings
        # Represent as a set of integers (the mask for vertices 1..n-1)
        # Actually, let's just use full masks with bit 0 = 0
        # alive = set of colorings not yet forbidden
        alive = set(range(1 << (n - 1)))  # bit 0 is always 0
        
        selected = []
        rng = random.Random(12345 + n)
        
        while alive:
            if time.time() > deadline - 3:
                return None
            if len(selected) >= max_edges:
                return None

            # Find edge that kills most alive colorings
            best_idx = -1
            best_kills = 0
            best_mask = 0

            # For efficiency, sample if too many edges
            if m_total > 200:
                sample_size = min(80, m_total)
                candidates = rng.sample(range(m_total), sample_size)
            else:
                candidates = range(m_total)

            selected_set = set(selected)

            for i in candidates:
                if i in selected_set:
                    continue
                em = all_masks[i]
                kills = 0
                for c in alive:
                    if (c & em) == 0 or (c & em) == em:
                        kills += 1
                if kills > best_kills:
                    best_kills = kills
                    best_idx = i
                    best_mask = em

            if best_idx == -1 or best_kills == 0:
                return None

            selected.append(best_idx)
            
            # Update alive: remove colorings killed by this edge
            em = best_mask
            new_alive = set()
            for c in alive:
                if not ((c & em) == 0 or (c & em) == em):
                    new_alive.add(c)
            alive = new_alive

        # Found a covering set
        return [all_edges[i] for i in selected]

    def minimize_edges(n, edges, deadline):
        """Try to remove edges while preserving non-2-colorability."""
        current = list(edges)
        rng = random.Random(999 + n)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
            # Shuffle to avoid always removing from the end
            order = list(range(len(current)))
            rng.shuffle(order)
            for i in order:
                if time.time() > deadline - 4:
                    break
                if len(current) <= 1:
                    break
                trial = current[:i] + current[i+1:]
                if check_non_2colorable(n, trial):
                    current = trial
                    improved = True
                    break  # Restart with new order
        
        return current

    best_edges = None
    best_m = float('inf')
    best_n = 0

    # Try different n values
    for n in [9, 10, 11, 12, 13, 14]:
        if time.time() > deadline - 15:
            break

        # Try greedy cover with various max_edges
        for max_e in [30, 40, 50, 60, 80, 100]:
            if time.time() > deadline - 10:
                break
            
            result = greedy_cover(n, max_e, deadline)
            if result is not None:
                m = len(result)
                if m < best_m:
                    best_m = m
                    best_edges = result[:]
                    best_n = n
                
                # Try to minimize
                if time.time() < deadline - 8:
                    minimized = minimize_edges(n, result, deadline)
                    if len(minimized) < best_m:
                        best_m = len(minimized)
                        best_edges = minimized
                        best_n = n
                
                # If we found a small one, try harder
                if best_m <= 35:
                    break

        # Also try random search for this n
        if time.time() < deadline - 10:
            all_edges_n = list(combinations(range(n), 5))
            rng = random.Random(n * 7 + 3)
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                target_m = min(51, len(all_edges_n))
                indices = rng.sample(range(len(all_edges_n)), target_m)
                subset = [all_edges_n[i] for i in indices]
                if check_non_2colorable(n, subset):
                    if target_m < best_m:
                        best_m = target_m
                        best_edges = subset[:]
                        best_n = n
                    # Minimize
                    if time.time() < deadline - 5:
                        minimized = minimize_edges(n, subset, deadline)
                        if len(minimized) < best_m:
                            best_m = len(minimized)
                            best_edges = minimized
                            best_n = n

    if best_edges is None:
        # Fallback: Fano-plane lift (9 vertices, 7 edges)
        fano_edges_3 = [
            (0, 1, 2), (0, 3, 4), (0, 5, 6),
            (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5),
        ]
        base_edges = [(e[0], e[1], e[2], 7, 8) for e in fano_edges_3]
        best_edges = base_edges
        best_n = 9
        best_m = len(base_edges)

    # Determine final n
    if best_edges:
        max_v = max(max(e) for e in best_edges) + 1
        n_final = max(5, max_v)
        edges_final = [list(e) for e in best_edges]
        return {'n': n_final, 'edges': edges_final}

    # Absolute fallback
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
