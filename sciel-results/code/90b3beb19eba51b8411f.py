"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
import random

def _build_bitmask(n, edges):
    """Build adjacency as list of bitmasks for fast triangle detection."""
    mask = [0] * n
    for u, v in edges:
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)
    return mask

def _compute_score_fast(n, mask, deg):
    """Compute max triangle degree sum using bitmask operations."""
    max_score = 0
    for i in range(n):
        mi = mask[i]
        m = mi & ~((1 << (i + 1)) - 1)
        while m:
            j_bit = m & (-m)
            j = j_bit.bit_length() - 1
            m ^= j_bit
            common = mask[i] & mask[j] & ~((1 << (j + 1)) - 1)
            while common:
                k_bit = common & (-common)
                k = k_bit.bit_length() - 1
                common ^= k_bit
                s = deg[i] + deg[j] + deg[k]
                if s > max_score:
                    max_score = s
    return max_score

def _get_non_edges(n, edge_set):
    return [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]

def _get_worst_triangles(n, mask, deg, score):
    """Get all triangles that achieve the current max score."""
    worst = []
    for i in range(n):
        mi = mask[i]
        m = mi & ~((1 << (i + 1)) - 1)
        while m:
            j_bit = m & (-m)
            j = j_bit.bit_length() - 1
            m ^= j_bit
            common = mask[i] & mask[j] & ~((1 << (j + 1)) - 1)
            while common:
                k_bit = common & (-common)
                k = k_bit.bit_length() - 1
                common ^= k_bit
                s = deg[i] + deg[j] + deg[k]
                if s == score:
                    worst.append((i, j, k))
    return worst

def _try_swap(n, mask, deg, edge_set, e_remove, e_add, best_score):
    """Try a single edge swap. Returns (new_score, improved)."""
    u1, v1 = e_remove
    u2, v2 = e_add
    
    mask[u1] &= ~(1 << v1)
    mask[v1] &= ~(1 << u1)
    mask[u2] |= (1 << v2)
    mask[v2] |= (1 << u2)
    deg[u1] -= 1
    deg[v1] -= 1
    deg[u2] += 1
    deg[v2] += 1
    
    new_score = _compute_score_fast(n, mask, deg)
    
    if new_score < best_score:
        return new_score, True
    else:
        # Revert
        mask[u1] |= (1 << v1)
        mask[v1] |= (1 << u1)
        mask[u2] &= ~(1 << v2)
        mask[v2] &= ~(1 << u2)
        deg[u1] += 1
        deg[v1] += 1
        deg[u2] -= 1
        deg[v2] -= 1
        return new_score, False

def _milp_branch_and_bound(n, target_score, max_time=60):
    """
    MILP-style branch and bound to find a graph with score <= target_score.
    
    Formulation:
    - Binary variables x_{uv} for each potential edge (u,v)
    - Constraints:
      1. Sum of all x_{uv} = 65 (exactly 65 edges)
      2. For every triangle (i,j,k): deg(i) + deg(j) + deg(k) <= target_score
         where deg(v) = sum of x_{uv} over all u != v
    
    We use a greedy relaxation + branch on edge variables.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    all_edges = list(combinations(range(n), 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges)  # 120 edges
    
    # Precompute for each edge (u,v), which triangles it participates in
    # and which vertices' degrees it affects
    edge_vertices = {}
    for i, (u, v) in enumerate(all_edges):
        edge_vertices[i] = (u, v)
    
    # For each vertex, list of edges incident to it
    vertex_edges = [[] for _ in range(n)]
    for i, (u, v) in enumerate(all_edges):
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)
    
    # All triangles
    triangles = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triangles.append((i, j, k))
    
    best_solution = None
    best_score = target_score + 1
    
    # Greedy construction: start with a random feasible solution, then improve
    rng = random.Random(7)
    
    def _greedy_construct(target_score):
        """Try to construct a graph with score <= target_score using greedy heuristic."""
        # Start with empty graph, add edges greedily
        edge_set = set()
        mask = [0] * n
        deg = [0] * n
        
        # Order edges by some heuristic: prioritize edges between high-degree vertices
        # to concentrate edges and reduce triangle degree sums
        # Actually, we want to AVOID creating high-degree-sum triangles
        # So we should prefer edges that don't create bad triangles
        
        # Strategy: build from a base structure
        # Try different base structures
        
        # Base 1: K_{a,b} + internal edges
        for a in range(4, 13):
            b = n - a
            if a > b:
                continue
            # Cross edges
            cross_edges = []
            for u in range(a):
                for v in range(a, n):
                    cross_edges.append((u, v))
            num_cross = a * b
            remaining = 65 - num_cross
            if remaining < 0:
                continue
            
            # Internal edges: choose 'remaining' edges from within the two parts
            part1_edges = list(combinations(range(a), 2))
            part2_edges = list(combinations(range(a, n), 2))
            all_internal = part1_edges + part2_edges
            
            if len(all_internal) < remaining:
                continue
            
            rng.shuffle(all_internal)
            internal_edges = all_internal[:remaining]
            
            edges = cross_edges + internal_edges
            edge_set_test = set(tuple(sorted(e)) for e in edges)
            mask_test = _build_bitmask(n, edges)
            deg_test = [0] * n
            for u, v in edges:
                deg_test[u] += 1
                deg_test[v] += 1
            
            score = _compute_score_fast(n, mask_test, deg_test)
            if score <= target_score:
                return edges, score
        
        # Base 2: Random perturbation of K_{8,8}+1
        for _ in range(50):
            edges = []
            for u in range(8):
                for v in range(8, 16):
                    edges.append((u, v))
            # Add 1 random edge within a part
            part1 = list(combinations(range(8), 2))
            part2 = list(combinations(range(8, 16), 2))
            extra = rng.choice(part1 + part2)
            edges.append(extra)
            
            edge_set_test = set(tuple(sorted(e)) for e in edges)
            mask_test = _build_bitmask(n, edges)
            deg_test = [0] * n
            for u, v in edges:
                deg_test[u] += 1
                deg_test[v] += 1
            
            score = _compute_score_fast(n, mask_test, deg_test)
            if score <= target_score:
                return edges, score
        
        return None, None
    
    # Try greedy construction first
    result, score = _greedy_construct(target_score)
    if result is not None:
        return result, score
    
    # If greedy fails, do local search from multiple random starts
    # with the constraint that we must achieve score <= target_score
    for attempt in range(20):
        if time.time() >= deadline:
            break
        
        # Random starting point
        edges = list(combinations(range(n), 2))
        rng.shuffle(edges)
        edges = edges[:65]
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        mask = _build_bitmask(n, edges)
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        current_score = _compute_score_fast(n, mask, deg)
        
        # Local search with target
        for iteration in range(500):
            if time.time() >= deadline:
                break
            if current_score <= target_score:
                return [list(e) for e in edge_set], current_score
            
            non_edges = _get_non_edges(n, edge_set)
            if not non_edges:
                break
            
            improved = False
            worst_tris = _get_worst_triangles(n, mask, deg, current_score)
            
            # Try to break worst triangles
            for (ti, tj, tk) in worst_tris:
                tri_edges = [(ti, tj), (ti, tk), (tj, tk)]
                for e_remove in tri_edges:
                    e_remove = tuple(sorted(e_remove))
                    if e_remove not in edge_set:
                        continue
                    for e_add in non_edges:
                        if e_add in edge_set:
                            continue
                        new_score, improved_try = _try_swap(n, mask, deg, edge_set, e_remove, e_add, current_score)
                        if improved_try:
                            current_score = new_score
                            improved = True
                            edge_set.discard(e_remove)
                            edge_set.add(tuple(sorted(e_add)))
                            non_edges = _get_non_edges(n, edge_set)
                            if current_score <= target_score:
                                return [list(e) for e in edge_set], current_score
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Random restart
                break
    
    return None, None

def _local_search(n, edges, max_time=100):
    """Constraint-driven local search: prioritize swaps that break worst triangles."""
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    rng = random.Random(123)
    
    while time.time() < deadline:
        non_edges = _get_non_edges(n, edge_set)
        if not non_edges:
            break
        
        improved = False
        worst_tris = _get_worst_triangles(n, mask, deg, current_score)
        
        for (ti, tj, tk) in worst_tris:
            if time.time() >= deadline:
                break
            tri_edges = [(ti, tj), (ti, tk), (tj, tk)]
            for e_remove in tri_edges:
                e_remove = tuple(sorted(e_remove))
                if e_remove not in edge_set:
                    continue
                shuffled_non_edges = list(non_edges)
                rng.shuffle(shuffled_non_edges)
                for e_add in shuffled_non_edges:
                    if e_add in edge_set:
                        continue
                    new_score, improved_try = _try_swap(n, mask, deg, edge_set, e_remove, e_add, best_score)
                    if improved_try:
                        best_score = new_score
                        best_edge_set = set(edge_set)
                        best_edge_set.discard(e_remove)
                        best_edge_set.add(tuple(sorted(e_add)))
                        current_score = new_score
                        improved = True
                        edge_set.discard(e_remove)
                        edge_set.add(tuple(sorted(e_add)))
                        non_edges = _get_non_edges(n, edge_set)
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # General local search
        shuffled_remove = list(edge_set)
        rng.shuffle(shuffled_remove)
        for e_remove in shuffled_remove:
            if time.time() >= deadline:
                break
            shuffled_add = list(non_edges)
            rng.shuffle(shuffled_add)
            for e_add in shuffled_add:
                if e_add in edge_set:
                    continue
                new_score, improved_try = _try_swap(n, mask, deg, edge_set, e_remove, e_add, best_score)
                if improved_try:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    non_edges = _get_non_edges(n, edge_set)
                    break
            if improved:
                break
        
        if not improved:
            # 2-swap moves
            two_swap_improved = False
            shuffled_remove1 = list(edge_set)
            rng.shuffle(shuffled_remove1)
            for e_r1 in shuffled_remove1[:20]:
                if time.time() >= deadline:
                    break
                u1, v1 = e_r1
                mask[u1] &= ~(1 << v1)
                mask[v1] &= ~(1 << u1)
                deg[u1] -= 1
                deg[v1] -= 1
                edge_set.discard(e_r1)
                
                shuffled_remove2 = list(edge_set)
                rng.shuffle(shuffled_remove2)
                for e_r2 in shuffled_remove2[:20]:
                    if time.time() >= deadline:
                        break
                    u2, v2 = e_r2
                    if e_r2 == e_r1:
                        continue
                    mask[u2] &= ~(1 << v2)
                    mask[v2] &= ~(1 << u2)
                    deg[u2] -= 1
                    deg[v2] -= 1
                    edge_set.discard(e_r2)
                    
                    non_edges_2 = _get_non_edges(n, edge_set)
                    shuffled_add = list(non_edges_2)
                    rng.shuffle(shuffled_add)
                    found = False
                    for e_a1 in shuffled_add[:30]:
                        if time.time() >= deadline:
                            break
                        a1, b1 = e_a1
                        mask[a1] |= (1 << b1)
                        mask[b1] |= (1 << a1)
                        deg[a1] += 1
                        deg[b1] += 1
                        edge_set.add(e_a1)
                        
                        non_edges_3 = _get_non_edges(n, edge_set)
                        for e_a2 in non_edges_3[:30]:
                            a2, b2 = e_a2
                            mask[a2] |= (1 << b2)
                            mask[b2] |= (1 << a2)
                            deg[a2] += 1
                            deg[b2] += 1
                            edge_set.add(e_a2)
                            
                            new_score = _compute_score_fast(n, mask, deg)
                            
                            if new_score < best_score:
                                best_score = new_score
                                best_edge_set = set(edge_set)
                                current_score = new_score
                                two_swap_improved = True
                                found = True
                            else:
                                mask[a2] &= ~(1 << b2)
                                mask[b2] &= ~(1 << a2)
                                deg[a2] -= 1
                                deg[b2] -= 1
                                edge_set.discard(e_a2)
                            
                            if found:
                                break
                        
                        if not found:
                            mask[a1] &= ~(1 << b1)
                            mask[b1] &= ~(1 << a1)
                            deg[a1] -= 1
                            deg[b1] -= 1
                            edge_set.discard(e_a1)
                        
                        if found:
                            break
                    
                    if not found:
                        mask[u2] |= (1 << v2)
                        mask[v2] |= (1 << u2)
                        deg[u2] += 1
                        deg[v2] += 1
                        edge_set.add(e_r2)
                    
                    if found:
                        break
                
                if not found:
                    mask[u1] |= (1 << v1)
                    mask[v1] |= (1 << u1)
                    deg[u1] += 1
                    deg[v1] += 1
                    edge_set.add(e_r1)
                
                if found:
                    improved = True
                    break
            
            if not two_swap_improved:
                break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def _polish(n, edges, max_time=30):
    """Final polish with targeted search."""
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        non_edges = _get_non_edges(n, edge_set)
        if not non_edges:
            break
        
        improved = False
        worst_tris = _get_worst_triangles(n, mask, deg, current_score)
        
        for (ti, tj, tk) in worst_tris:
            if time.time() >= deadline:
                break
            tri_edges = [(ti, tj), (ti, tk), (tj, tk)]
            for e_remove in tri_edges:
                e_remove = tuple(sorted(e_remove))
                if e_remove not in edge_set:
                    continue
                shuffled_non_edges = list(non_edges)
                rng.shuffle(shuffled_non_edges)
                for e_add in shuffled_non_edges:
                    if e_add in edge_set:
                        continue
                    new_score, improved_try = _try_swap(n, mask, deg, edge_set, e_remove, e_add, best_score)
                    if improved_try:
                        best_score = new_score
                        best_edge_set = set(edge_set)
                        best_edge_set.discard(e_remove)
                        best_edge_set.add(tuple(sorted(e_add)))
                        current_score = new_score
                        improved = True
                        edge_set.discard(e_remove)
                        edge_set.add(tuple(sorted(e_add)))
                        non_edges = _get_non_edges(n, edge_set)
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        shuffled_remove = list(edge_set)
        rng.shuffle(shuffled_remove)
        for e_remove in shuffled_remove:
            if time.time() >= deadline:
                break
            shuffled_add = list(non_edges)
            rng.shuffle(shuffled_add)
            for e_add in shuffled_add:
                if e_add in edge_set:
                    continue
                new_score, improved_try = _try_swap(n, mask, deg, edge_set, e_remove, e_add, best_score)
                if improved_try:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    non_edges = _get_non_edges(n, edge_set)
                    break
            if improved:
                break
        
        if not improved:
            break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def construct():
    n = 16
    target_edges = 65
    total_deadline = 145  # safety margin before 150s limit
    t0 = time.time()
    
    # Phase 1: Try MILP-style search for score <= 23
    milp_time = min(40, total_deadline - 30)
    milp_result, milp_score = _milp_branch_and_bound(n, 23, max_time=milp_time)
    
    if milp_result is not None:
        # Found a solution with score <= 23!
        elapsed = time.time() - t0
        polish_time = min(30, max(5, total_deadline - 10 - elapsed))
        if polish_time > 5:
            best_edges, best_score = _polish(n, milp_result, max_time=polish_time)
        else:
            best_edges, best_score = milp_result, milp_score
        return {'n': n, 'edges': best_edges}
    
    # Phase 2: Fall back to local search from good starting points
    initial_edges = []
    for u in range(6):
        for v in range(6, 16):
            initial_edges.append([u, v])
    internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in internal_edges:
        initial_edges.append([u, v])
    
    elapsed = time.time() - t0
    search_time = min(60, max(10, total_deadline - 20 - elapsed))
    best_edges, best_score = _local_search(n, initial_edges, max_time=search_time)
    
    # Polish pass
    elapsed = time.time() - t0
    polish_time = min(25, max(5, total_deadline - 10 - elapsed))
    if polish_time > 5:
        best_edges, best_score = _polish(n, best_edges, max_time=polish_time)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
