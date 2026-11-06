"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 16 vertices with exactly 65 edges that minimizes the
    maximum degree sum over all triangles.
    
    Uses spectral optimization (max eigenvalue of adjacency matrix) as a smooth
    surrogate to guide edge swaps, combined with bitmask-based exact scoring.
    """
    n = 16
    target_edges = 65
    
    # Deadline guard: return best found before 148s (2s safety margin)
    start_time = time.time()
    deadline = start_time + 148.0
    
    # Fast scoring using bitmasks
    def score_from_adj(adj_masks, deg):
        """Compute max triangle degree sum using bitmask adjacency."""
        max_sum = 0
        for u in range(n):
            mu = adj_masks[u]
            for v in range(u + 1, n):
                if mu & (1 << v):
                    common = mu & adj_masks[v]
                    # Iterate over set bits in common
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        cm ^= lsb
        return max_sum
    
    def build_adj(deg, edge_set):
        """Build bitmask adjacency and degree array from edge set."""
        adj_masks = [0] * n
        deg = [0] * n
        for u, v in edge_set:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_masks, deg
    
    def apply_swap(adj_masks, deg, e_remove, e_add):
        """Apply an edge swap in-place, returning new adj_masks and deg."""
        adj_masks = list(adj_masks)
        deg = list(deg)
        u, v = e_remove
        adj_masks[u] ^= (1 << v)
        adj_masks[v] ^= (1 << u)
        deg[u] -= 1
        deg[v] -= 1
        u, v = e_add
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        return adj_masks, deg
    
    def compute_score(edges):
        """Compute the maximum degree sum over all triangles."""
        es = set()
        for u, v in edges:
            es.add((min(u, v), max(u, v)))
        adj_masks, deg = build_adj(None, es)
        return score_from_adj(adj_masks, deg)
    
    # Start from K_{6,10} base: vertices 0-5 in part A (size 6), 6-15 in part B (size 10)
    # K_{6,10} has 60 edges. Need 5 more edges within part B.
    # Degrees in K_{6,10}: part A vertices have deg 10, part B vertices have deg 6.
    # Adding edges within B increases degrees of B vertices.
    # Strategy: add 5 edges in B to balance degrees and minimize triangle sums.
    
    def make_seed_edges():
        """K_{6,10} + 5 internal edges in the 10-vertex part."""
        edges = []
        # Cross edges: all pairs between {0..5} and {6..15}
        for u in range(6):
            for v in range(6, 16):
                edges.append([u, v])
        # 5 internal edges in part B (vertices 6-15)
        # Use a matching-like structure to keep degrees balanced
        # Add edges: (6,7), (8,9), (10,11), (12,13), (14,15)
        internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
        for u, v in internal_edges:
            edges.append([u, v])
        return edges
    
    best_edges = make_seed_edges()
    best_score = compute_score(best_edges)
    
    # Also try K_{8,8} + 1 edge as alternative seed
    def make_seed_edges_v2():
        """K_{8,8} + one internal edge."""
        edges = [[u, v] for u in range(8) for v in range(8, 16)]
        edges.append([0, 1])
        return edges
    
    s2 = compute_score(make_seed_edges_v2())
    if s2 < best_score:
        best_score = s2
        best_edges = make_seed_edges_v2()
    
    def edges_to_set(edges):
        es = set()
        for u, v in edges:
            es.add((min(u, v), max(u, v)))
        return es
    
    def edges_to_list(es):
        return [[u, v] for u, v in es]
    
    # Generate candidate edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local search with edge swaps using bitmask scoring
    current_es = edges_to_set(best_edges)
    current_adj, current_deg = build_adj(None, current_es)
    current_score = best_score
    
    random.seed(42)
    no_improve = 0
    max_no_improve = 3000  # Allow more iterations with faster scoring
    
    while time.time() < deadline - 8 and no_improve < max_no_improve:
        # Pick a random edge to remove
        if not current_es:
            break
        e_remove = random.choice(list(current_es))
        # Pick a random edge to add (not already present)
        candidates = [p for p in all_pairs if p not in current_es]
        if not candidates:
            break
        e_add = random.choice(candidates)
        
        new_es = set(current_es)
        new_es.remove(e_remove)
        new_es.add(e_add)
        
        new_adj, new_deg = apply_swap(current_adj, current_deg, e_remove, e_add)
        new_score = score_from_adj(new_adj, new_deg)
        
        if new_score < current_score:
            current_es = new_es
            current_adj, current_deg = new_adj, new_deg
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_edges = edges_to_list(current_es)
        else:
            no_improve += 1
    
    # Simulated annealing phase if still at or above 24
    if best_score >= 24 and time.time() < deadline - 8:
        random.seed(123)
        current_es = edges_to_set(best_edges)
        current_adj, current_deg = build_adj(None, current_es)
        current_score = best_score
        
        T = 8.0
        T_min = 0.05
        cooling = 0.9995
        
        for iteration in range(80000):
            if time.time() > deadline - 8:
                break
            if T < T_min:
                break
            
            # Pick a random edge swap
            if not current_es:
                break
            e_remove = random.choice(list(current_es))
            candidates = [p for p in all_pairs if p not in current_es]
            if not candidates:
                break
            e_add = random.choice(candidates)
            
            new_es = set(current_es)
            new_es.remove(e_remove)
            new_es.add(e_add)
            
            new_adj, new_deg = apply_swap(current_adj, current_deg, e_remove, e_add)
            new_score = score_from_adj(new_adj, new_deg)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_es = new_es
                current_adj, current_deg = new_adj, new_deg
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges_to_list(current_es)
            
            T *= cooling
    
    # Spectral-guided local search phase
    # Use max eigenvalue of adjacency matrix as a smooth surrogate
    # In quasi-regular graphs, spectral radius correlates with max degree sum
    if time.time() < deadline - 5:
        def spectral_radius_from_adj(adj_masks):
            """Compute max eigenvalue of adjacency matrix via power iteration."""
            A = np.zeros((n, n), dtype=np.float64)
            for u in range(n):
                mu = adj_masks[u]
                for v in range(u + 1, n):
                    if mu & (1 << v):
                        A[u, v] = 1.0
                        A[v, u] = 1.0
            # Power iteration
            x = np.ones(n, dtype=np.float64)
            x /= np.linalg.norm(x)
            for _ in range(150):
                x_new = A @ x
                norm = np.linalg.norm(x_new)
                if norm < 1e-15:
                    break
                x_new /= norm
                if np.abs(np.dot(x_new, x) - 1.0) < 1e-12:
                    x = x_new
                    break
                x = x_new
            return float(np.dot(x, A @ x))
        
        # Spectral-guided hill climbing
        current_es = edges_to_set(best_edges)
        current_adj, current_deg = build_adj(None, current_es)
        current_score = best_score
        current_spec = spectral_radius_from_adj(current_adj)
        
        random.seed(777)
        no_improve_spec = 0
        max_no_improve_spec = 4000
        
        while time.time() < deadline - 5 and no_improve_spec < max_no_improve_spec:
            e_remove = random.choice(list(current_es))
            candidates = [p for p in all_pairs if p not in current_es]
            if not candidates:
                break
            e_add = random.choice(candidates)
            
            new_es = set(current_es)
            new_es.remove(e_remove)
            new_es.add(e_add)
            
            new_adj, new_deg = apply_swap(current_adj, current_deg, e_remove, e_add)
            new_spec = spectral_radius_from_adj(new_adj)
            new_score = score_from_adj(new_adj, new_deg)
            
            # Accept if spectral radius improves OR exact score improves
            if new_spec < current_spec - 1e-6 or new_score < current_score:
                current_es = new_es
                current_adj, current_deg = new_adj, new_deg
                current_score = new_score
                current_spec = new_spec
                no_improve_spec = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges_to_list(current_es)
            else:
                no_improve_spec += 1
        
        # If spectral search found something better, restart local search from there
        if best_score >= 24 and time.time() < deadline - 5:
            # Quick local search from the spectrally-improved solution
            local_es = edges_to_set(best_edges)
            local_adj, local_deg = build_adj(None, local_es)
            local_score = best_score
            random.seed(4242)
            for _ in range(8000):
                if time.time() > deadline - 5:
                    break
                e_remove = random.choice(list(local_es))
                candidates = [p for p in all_pairs if p not in local_es]
                if not candidates:
                    break
                e_add = random.choice(candidates)
                new_es = set(local_es)
                new_es.remove(e_remove)
                new_es.add(e_add)
                new_adj, new_deg = apply_swap(local_adj, local_deg, e_remove, e_add)
                new_score = score_from_adj(new_adj, new_deg)
                if new_score < local_score:
                    local_es = new_es
                    local_adj, local_deg = new_adj, new_deg
                    local_score = new_score
                    if local_score < best_score:
                        best_score = local_score
                        best_edges = edges_to_list(local_es)
    
    # Final polish: try all single-edge swaps from best
    if time.time() < deadline - 2:
        best_es = edges_to_set(best_edges)
        best_adj, best_deg = build_adj(None, best_es)
        best_s = best_score
        
        # Try removing each edge and adding each non-edge
        # This is O(E * (V^2 - E)) which is manageable for n=16
        for e_remove in list(best_es):
            if time.time() > deadline - 2:
                break
            for e_add in all_pairs:
                if e_add in best_es:
                    continue
                new_es = set(best_es)
                new_es.remove(e_remove)
                new_es.add(e_add)
                new_adj, new_deg = apply_swap(best_adj, best_deg, e_remove, e_add)
                new_s = score_from_adj(new_adj, new_deg)
                if new_s < best_s:
                    best_s = new_s
                    best_es = new_es
                    best_adj, best_deg = new_adj, new_deg
                    best_edges = edges_to_list(best_es)
                    best_score = best_s
    
    # Second polish pass if time allows
    if time.time() < deadline - 1:
        best_es = edges_to_set(best_edges)
        best_adj, best_deg = build_adj(None, best_es)
        best_s = best_score
        
        for e_remove in list(best_es):
            if time.time() > deadline - 1:
                break
            for e_add in all_pairs:
                if e_add in best_es:
                    continue
                new_es = set(best_es)
                new_es.remove(e_remove)
                new_es.add(e_add)
                new_adj, new_deg = apply_swap(best_adj, best_deg, e_remove, e_add)
                new_s = score_from_adj(new_adj, new_deg)
                if new_s < best_s:
                    best_s = new_s
                    best_es = new_es
                    best_adj, best_deg = new_adj, new_deg
                    best_edges = edges_to_list(best_es)
                    best_score = best_s
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
