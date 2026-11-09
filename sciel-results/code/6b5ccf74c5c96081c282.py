"""Rewritten program: K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

N = 18
FULL_MASK = (1 << N) - 1

# Precompute all C(18,2) = 153 edges
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_INDEX = {}
for idx, (u, v) in enumerate(ALL_EDGES):
    EDGE_INDEX[(u, v)] = idx

# Precompute for each edge (u,v): the set of vertices w > v that are adjacent to both u and v
# This is used for fast K4-free checking
# For edge (u,v) with u<v: common neighbors above v
EDGE_COMMON_ABOVE = {}
for (u, v) in ALL_EDGES:
    # We'll fill this in dynamically based on the graph, but we can precompute the mask of vertices > v
    EDGE_COMMON_ABOVE[(u, v)] = 0  # Will be computed per-graph

def _is_k4_free(adj_bits):
    """Check if graph is K4-free using bitmasks."""
    for a in range(N):
        na = adj_bits[a]
        for b in range(a + 1, N):
            if not (na >> b) & 1:
                continue
            common_ab = na & adj_bits[b]
            for c in range(b + 1, N):
                if not (common_ab >> c) & 1:
                    continue
                common_abc = common_ab & adj_bits[c]
                mask_above_c = FULL_MASK ^ ((1 << (c + 1)) - 1)
                if common_abc & mask_above_c:
                    return False
    return True

def _largest_triangle_free(adj_bits):
    """Find largest triangle-free induced subgraph size using bitmask enumeration with precomputed triangle masks."""
    # Precompute: for each vertex i, the set of triangles containing i
    # A triangle is (i,j,k) with i<j<k and all three edges present
    # For a subset S to be triangle-free, it must not contain any triangle
    
    # Build list of all triangles in the graph
    triangles = []
    for i in range(N):
        ai = adj_bits[i]
        for j in range(i + 1, N):
            if not (ai >> j) & 1:
                continue
            common = ai & adj_bits[j]
            for k in range(j + 1, N):
                if (common >> k) & 1:
                    triangles.append((i, j, k))
    
    # Now enumerate subsets by size descending
    # For each subset, check if it contains any triangle
    # A subset contains a triangle iff for some triangle (i,j,k), all three are in the subset
    
    # Precompute triangle masks
    tri_masks = [(1 << i) | (1 << j) | (1 << k) for (i, j, k) in triangles]
    
    # Check sizes from N down to 1
    for size in range(N, 0, -1):
        # Enumerate all subsets of given size
        for subset in combinations(range(N), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            # Check if mask contains any triangle
            has_triangle = False
            for tm in tri_masks:
                if (mask & tm) == tm:
                    has_triangle = True
                    break
            if not has_triangle:
                return size
    return 0

def _bits_to_edges(adj_bits):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(42)
    
    # === Phase 1: Construct dense K4-free graph ===
    # Start with complete graph and remove edges to make it K4-free
    # Or use a known construction: Turan graph T(18,3) is K4-free
    
    # Turan graph T(n, r): partition n vertices into r parts as equal as possible,
    # connect all vertices in different parts. This is K_{r+1}-free.
    # T(18, 3) has 3 parts of size 6 each, and is K4-free.
    
    def turan_graph(n, r):
        adj = [0] * n
        part_size = n // r
        remainder = n % r
        # Part i contains vertices [i*part_size + min(i, remainder), ...)
        # Simpler: assign vertex v to part v % r
        for i in range(n):
            for j in range(i + 1, n):
                if i % r != j % r:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    best_adj = turan_graph(N, 3)
    best_score = _largest_triangle_free(best_adj)
    
    if time.time() > deadline:
        return _bits_to_edges(best_adj)
    
    # === Phase 2: Simulated annealing ===
    def evaluate(adj_bits):
        return _largest_triangle_free(adj_bits)
    
    current_adj = best_adj[:]
    current_score = best_score
    
    T = 3.0  # Initial temperature
    T_min = 0.01
    cooling = 0.997
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try adding a random non-edge
            non_edges = [(u, v) for (u, v) in ALL_EDGES if not (current_adj[u] >> v) & 1]
            if not non_edges:
                continue
            u, v = rng.choice(non_edges)
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if not _is_k4_free(new_adj):
                continue
            new_score = evaluate(new_adj)
            delta = new_score - current_score  # We want to minimize
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        elif move_type < 0.85:
            # Try removing a random edge
            edges_present = [(u, v) for (u, v) in ALL_EDGES if (current_adj[u] >> v) & 1]
            if not edges_present:
                continue
            u, v = rng.choice(edges_present)
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_score = evaluate(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        else:
            # Try a 2-opt swap: remove one edge, add another
            edges_present = [(u, v) for (u, v) in ALL_EDGES if (current_adj[u] >> v) & 1]
            non_edges = [(u, v) for (u, v) in ALL_EDGES if not (current_adj[u] >> v) & 1]
            if not edges_present or not non_edges:
                continue
            u1, v1 = rng.choice(edges_present)
            u2, v2 = rng.choice(non_edges)
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
            new_adj = current_adj[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            if not _is_k4_free(new_adj):
                continue
            new_score = evaluate(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        
        # Cool down
        T = max(T_min, T * cooling)
        
        if best_score <= 9:
            break
    
    # === Phase 3: Local search refinement on best found ===
    if time.time() < deadline - 10:
        for iteration in range(2000):
            if time.time() > deadline - 5:
                break
            
            adj = best_adj[:]
            improved = False
            # Try all single-edge additions
            for (u, v) in ALL_EDGES:
                if (adj[u] >> v) & 1:
                    continue
                new_adj = adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                if not _is_k4_free(new_adj):
                    continue
                new_score = evaluate(new_adj)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = new_adj[:]
                    improved = True
                    break
            
            # Try all single-edge removals
            if not improved:
                for (u, v) in ALL_EDGES:
                    if not (adj[u] >> v) & 1:
                        continue
                    new_adj = adj[:]
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                    new_score = evaluate(new_adj)
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = new_adj[:]
                        improved = True
                        break
            
            if not improved and iteration > 50:
                break
    
    return _bits_to_edges(best_adj)
# EVOLVE-BLOCK-END
