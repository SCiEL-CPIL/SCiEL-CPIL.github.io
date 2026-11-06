import time
import random
from itertools import combinations

def _build_adj(edges, n):
    """Build adjacency bitmasks from edge list."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free(adj, n):
    """Check if graph (given by adjacency bitmasks) is K4-free."""
    for i in range(n):
        ai = adj[i]
        # Iterate over neighbors j > i
        j = (ai >> (i + 1)) & ((1 << (n - i - 1)) - 1)
        while j:
            lsb = j & -j
            jj = lsb.bit_length() - 1
            j -= lsb
            # Common neighbors of i and j, restricted to > j
            common = (ai & adj[jj]) >> (jj + 1)
            common &= (1 << (n - jj - 1)) - 1
            # Check if common contains an edge
            # Iterate through common neighbors k
            k = common
            while k:
                lsb_k = k & -k
                kk = lsb_k.bit_length() - 1
                k -= lsb_k
                # Check if any neighbor of k is also in common
                if adj[kk] & common:
                    return False
    return True

def _max_triangle_free_induced(adj, n):
    """
    Find the size of the largest triangle-free induced subgraph.
    Uses recursive branch and bound with bitsets.
    """
    # Precompute triangles as bitmasks
    triangles = []
    for i in range(n):
        ai = adj[i]
        # Neighbors j > i
        j = (ai >> (i + 1)) & ((1 << (n - i - 1)) - 1)
        while j:
            lsb = j & -j
            jj = lsb.bit_length() - 1
            j -= lsb
            # Common neighbors of i and j, restricted to > j
            common = (ai & adj[jj]) >> (jj + 1)
            common &= (1 << (n - jj - 1)) - 1
            k = common
            while k:
                lsb_k = k & -k
                kk = lsb_k.bit_length() - 1
                k -= lsb_k
                # Triangle (i, jj, kk)
                triangles.append((1 << i) | (1 << jj) | (1 << kk))
                
    if not triangles:
        return n

    # We want to find the max size of a set S such that no triangle is fully contained in S.
    # This is equivalent to: Find max S such that for all t in triangles, (S & t) != t.
    
    # Branch and Bound:
    # State: (candidates_mask, selected_count)
    # We try to add vertices from candidates_mask to the solution.
    # If adding a vertex v creates a triangle with two already selected vertices, we can't add it?
    # No, the condition is that the FINAL set is triangle-free.
    # So if we have a set S, it is valid if it contains no triangle.
    # We can build S incrementally.
    # If S is triangle-free, any subset is triangle-free.
    # We want to maximize |S|.
    
    # Algorithm:
    # 1. Start with all vertices as candidates.
    # 2. Pick a vertex v from candidates.
    # 3. Branch 1: Include v.
    #    - If including v creates a triangle with existing selected vertices, this branch is invalid?
    #    - Actually, it's easier to think of it as: We are selecting a subset S.
    #    - If we select S, and S contains a triangle, it's invalid.
    #    - So, if we have a partial solution P, and we consider adding v, we must ensure P ∪ {v} is triangle-free.
    #    - This means v cannot form a triangle with any two vertices in P.
    #    - So, if v has two neighbors in P that are connected to each other, we cannot add v.
    #    - If we can't add v, we must exclude it.
    #    - This logic is slightly flawed for maximization because excluding a vertex might allow us to include others that form triangles with it? No, if P is triangle free, and we add v, the only new triangles are those involving v. If v forms a triangle with P, we can't add v.
    #    - Wait, if we can't add v, we must exclude v. Is that optimal?
    #    - Not necessarily. Maybe we should have excluded one of the vertices in P that forms the triangle with v, to allow v and potentially others.
    #    - So simple greedy "add if possible" is not sufficient for finding the MAX. We need to branch.
    
    # Correct Branch and Bound:
    # We want to find a maximum independent set in the hypergraph of triangles? No.
    # We want a maximum subset of vertices that does NOT contain any hyperedge (triangle).
    # This is the "Maximum Subset with Forbidden Subsets" problem.
    
    # Let's use a standard B&B for Max Independent Set in a graph, but adapted?
    # No, the constraint is on triples.
    
    # Alternative: Iterate over all subsets? 2^18 is too slow for pure Python if done naively, but with bitsets and pruning it might be okay.
    # However, a recursive B&B is better.
    
    # Let's try a different perspective:
    # The complement of a triangle-free set is a vertex cover of the triangles?
    # No. A set S is triangle-free if it hits no triangle completely.
    # So the complement C = V \ S must hit every triangle.
    # i.e., C is a Vertex Cover of the 3-uniform hypergraph of triangles.
    # We want to MINIMIZE |C| to MAXIMIZE |S|.
    # This is the Minimum Vertex Cover problem for 3-uniform hypergraphs, which is NP-hard.
    # But N=18 is small.
    
    # Let's solve Min Vertex Cover on the hypergraph of triangles.
    # Branch and Bound for Min Vertex Cover:
    # State: (remaining_hyperedges, current_cover_size, candidate_vertices)
    # If no hyperedges remain, return current_cover_size.
    # Pick a hyperedge e = {u, v, w}.
    # We must pick at least one of u, v, w.
    # Branch 1: Pick u. Remove u from candidates. Remove all hyperedges containing u.
    # Branch 2: Pick v. ...
    # Branch 3: Pick w. ...
    # Pruning: If current_cover_size + lower_bound >= best_found, prune.
    # Lower bound: Max number of disjoint hyperedges remaining?
    
    # This seems robust.
    
    best_cover_size = n  # Worst case: remove all vertices
    
    # Preprocess: remove hyperedges that are subsets of others? (Not possible for size 3)
    # Remove vertices that are in no hyperedges? They can always be in S (never in C).
    # Actually, if a vertex is in no triangle, it can be in S. So it doesn't need to be in C.
    
    # Let's implement the B&B for Min Vertex Cover.
    
    # Convert triangles to a list of bitmasks
    # For efficiency, we can represent the set of remaining triangles as a list of indices or bitmasks?
    # Since N=18, we can represent the set of triangles as a bitmask of triangles? No, too many.
    # We can just pass the list of active triangles.
    
    # To speed up, we can precompute for each vertex, the list of triangle indices it belongs to.
    tri_indices = [[] for _ in range(n)]
    for idx, t in enumerate(triangles):
        for v in range(n):
            if t & (1 << v):
                tri_indices[v].append(idx)
                
    # State: (active_triangles_mask, current_size, candidate_mask)
    # active_triangles_mask is a bitmask over the list of triangles?
    # Number of triangles can be up to 816. 2^816 is too big.
    # So we can't use a bitmask for triangles.
    
    # We must use a list or a set of active triangles.
    # Or, we can use a recursive approach that picks a triangle and branches.
    
    def solve_min_vc(active_tris, current_size, candidate_mask):
        nonlocal best_cover_size
        
        if not active_tris:
            if current_size < best_cover_size:
                best_cover_size = current_size
            return
            
        # Pruning
        if current_size >= best_cover_size:
            return
            
        # Lower bound: 
        # Find max number of disjoint triangles in active_tris.
        # This is a matching in the hypergraph.
        # Greedy matching is a valid lower bound.
        lb = 0
        remaining = list(active_tris)
        while remaining:
            t = remaining.pop()
            lb += 1
            # Remove all triangles sharing a vertex with t
            # This is expensive.
            # Simpler LB: len(active_tris) / 3 ? No.
            # Let's skip complex LB for now, N=18 is small enough with good branching.
            
        # Pick a triangle to branch on.
        # Heuristic: Pick a triangle with vertices having high degree in the hypergraph?
        # Or just the first one.
        t = active_tris[0]
        u = -1
        v = -1
        w = -1
        mask = t
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            if u == -1: u = idx
            elif v == -1: v = idx
            else: w = idx
            mask -= lsb
            
        # Branch 1: Include u in Cover
        new_active = []
        for tri in active_tris:
            if not (tri & (1 << u)):
                new_active.append(tri)
        # Update candidate mask? We don't really need candidate_mask for this formulation, 
        # as we are just picking vertices to cover edges.
        # But we should ensure we don't pick vertices that are already picked?
        # The recursion naturally handles this because we remove edges covered by u.
        # We don't need to track picked vertices explicitly if we just remove covered edges.
        # However, we might pick the same vertex again? No, because if we pick u, we remove all edges with u.
        # So u won't appear in any remaining edge.
        
        solve_min_vc(new_active, current_size + 1, candidate_mask)
        
        # Branch 2: Include v in Cover
        new_active = []
        for tri in active_tris:
            if not (tri & (1 << v)):
                new_active.append(tri)
        solve_min_vc(new_active, current_size + 1, candidate_mask)
        
        # Branch 3: Include w in Cover
        new_active = []
        for tri in active_tris:
            if not (tri & (1 << w)):
                new_active.append(tri)
        solve_min_vc(new_active, current_size + 1, candidate_mask)

    solve_min_vc(triangles, 0, (1 << n) - 1)
    
    return n - best_cover_size

def _get_paley_edges(q):
    """Get edges for Paley graph of order q."""
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    edges = []
    for u in range(q):
        for v in range(u + 1, q):
            diff = (v - u) % q
            if diff in residues:
                edges.append([u, v])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 18
    
    best_edges = None
    best_score = float('inf')
    
    def evaluate_and_update(edges, name):
        nonlocal best_edges, best_score
        adj = _build_adj(edges, n)
        if not _is_k4_free(adj, n):
            return
        score = _max_triangle_free_induced(adj, n)
        if score < best_score:
            best_score = score
            best_edges = edges[:]
            
    # Candidate 1: Paley graph P(17) + isolated vertex
    edges_17 = _get_paley_edges(17)
    evaluate_and_update(edges_17, "Paley17")
    
    # Candidate 2: Two disjoint Paley graphs P(9)
    edges_9_1 = _get_paley_edges(9)
    edges_9_2 = [[u + 9, v + 9] for u, v in edges_9_1]
    edges_18 = edges_9_1 + edges_9_2
    evaluate_and_update(edges_18, "2xPaley9")
    
    # Candidate 3: K6,6,6 (Baseline)
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    edges_k666 = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    evaluate_and_update(edges_k666, "K666")
    
    # Local Search
    def local_search(start_edges, max_iters=200):
        current_edges = start_edges[:]
        current_adj = _build_adj(current_edges, n)
        if not _is_k4_free(current_adj, n):
            return None, float('inf')
        
        current_score = _max_triangle_free_induced(current_adj, n)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Generate list of possible moves: Add edge, Remove edge
            # To keep it simple and effective, we'll try flipping each edge/non-edge.
            # But that's O(N^2) * Cost(Score). Cost(Score) is high.
            # Better: Try adding edges first (usually improves score).
            # Then try removing edges if stuck?
            
            # Let's try a random order of non-edges to add
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (current_adj[u] & (1 << v)):
                        non_edges.append((u, v))
                        
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                
                # Check K4 constraint for adding (u, v)
                common = current_adj[u] & current_adj[v]
                # Remove u and v from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                creates_k4 = False
                # Check if any edge exists in the subgraph induced by common
                # Iterate through common
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    c -= lsb
                    if current_adj[w] & common:
                        creates_k4 = True
                        break
                        
                if not creates_k4:
                    # Try adding
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append([u, v])
                    
                    new_score = _max_triangle_free_induced(current_adj, n)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Restart loop to update non_edges list
                        break
                    else:
                        # Revert
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                        current_edges.pop()
                        
            # If no improvement from adding, try removing?
            # Removing an edge might allow adding other edges later?
            # Or removing an edge might increase score (bad).
            # So we generally only want to add edges.
            # Unless we are in a local minimum where we need to remove an edge to add a "better" edge?
            # Given the objective (minimize max triangle-free set), adding edges is monotonic (non-increasing score) if K4-free.
            # So we should just add edges until we can't.
            
        return current_edges, current_score

    # Run local search from best candidate
    if best_edges is not None:
        ls_edges, ls_score = local_search(best_edges)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges

    # Multi-start with random K4-free graphs
    # Generate random graphs and check K4-free
    # This is hard to generate directly.
    # Instead, start from Paley and perturb?
    
    # Let's try starting from a random graph and removing edges to make it K4-free?
    # Or start from empty and add edges?
    
    # Let's try a few random restarts from Paley(17) with some edges flipped (removed)
    for restart in range(5):
        if time.time() > deadline:
            break
        start_edges = best_edges[:]
        # Remove a few random edges
        for _ in range(5):
            if not start_edges: break
            idx = random.randint(0, len(start_edges) - 1)
            start_edges.pop(idx)
            
        ls_edges, ls_score = local_search(start_edges, max_iters=100)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges

    if best_edges is None:
        return edges_k666
        
    return best_edges
