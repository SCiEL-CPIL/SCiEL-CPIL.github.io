import time
import random
import math
import itertools

N = 18
FULL = (1 << N) - 1
DEADLINE_MARGIN = 5  # safety margin before 150s

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    rng = random.Random(42)

    def is_k4_free(adj_mask):
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    c_list = []
                    while m:
                        c = (m & -m).bit_length() - 1
                        c_list.append(c)
                        m &= m - 1
                    for i in range(len(c_list)):
                        for j in range(i + 1, len(c_list)):
                            if (adj_mask[c_list[i]] >> c_list[j]) & 1:
                                return False
        return True

    def find_all_triangles(adj_mask):
        tris = []
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        if c > b:
                            tris.append((a, b, c))
                        m &= m - 1
        return tris

    def max_tri_free(adj_mask):
        """Exact size of largest triangle-free induced subgraph via branch and bound."""
        tris = find_all_triangles(adj_mask)
        if not tris:
            return N
        best = [0]
        tri_list = list(tris)
        
        def bnb(avail):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return
            tri = None
            for (a, b, c) in tri_list:
                if (avail >> a) & 1 and (avail >> b) & 1 and (avail >> c) & 1:
                    tri = (a, b, c)
                    break
            if tri is None:
                best[0] = cnt
                return
            a, b, c = tri
            bnb(avail & ~(1 << a))
            bnb(avail & ~(1 << b))
            bnb(avail & ~(1 << c))
        
        bnb(FULL)
        return best[0]

    def make_k666():
        adj_mask = [0] * N
        for i in range(N):
            for j in range(N):
                if i != j and i // 6 != j // 6:
                    adj_mask[i] |= (1 << j)
        return adj_mask

    def build_adj_from_cliques(A, B, C):
        """Build adjacency from three 6-vertex cliques with inter-clique edge patterns.
        A, B, C are 36-bit masks where bit (i,j) for i,j in 0..5 indicates edge between
        clique vertex i and clique vertex j."""
        adj = [0] * N
        # Intra-clique: all edges within each clique
        for clique in range(3):
            base = clique * 6
            for i in range(6):
                for j in range(6):
                    if i != j:
                        adj[base + i] |= (1 << (base + j))
        # Inter-clique A-B
        for i in range(6):
            for j in range(6):
                if (A >> (i * 6 + j)) & 1:
                    adj[i] |= (1 << (6 + j))
                    adj[6 + j] |= (1 << i)
        # Inter-clique B-C
        for i in range(6):
            for j in range(6):
                if (B >> (i * 6 + j)) & 1:
                    adj[6 + i] |= (1 << (12 + j))
                    adj[12 + j] |= (1 << (6 + i))
        # Inter-clique A-C
        for i in range(6):
            for j in range(6):
                if (C >> (i * 6 + j)) & 1:
                    adj[i] |= (1 << (12 + j))
                    adj[12 + j] |= (1 << i)
        return adj

    def check_k4_free_cliques(A, B, C):
        """Check if the 3-clique structure with inter-clique patterns A, B, C is K4-free."""
        # A K4 can be:
        # 1. 4 vertices all in same clique - impossible since clique has 6 vertices but K4 needs 4 mutually adjacent
        #    Actually wait, within a clique of 6, any 4 form a K4! So we need cliques of size at most 3?
        #    No wait - the problem says K4-free, meaning no 4 vertices are ALL mutually adjacent.
        #    If we have a clique of 6, then any 4 of those 6 form a K4. So we CANNOT have cliques of size >= 4.
        #    
        #    Let me reconsider. We need K4-free, so no 4 mutually adjacent vertices.
        #    A clique of size 3 is fine (triangle), but a clique of size 4 is a K4.
        #    So we need each "clique" to be at most size 3, or we need to be more careful.
        #
        #    Actually, let me reconsider the structural approach. Instead of 3 cliques of 6,
        #    let's think differently.
        #
        #    For a K4-free graph, we can use the structure of 3 parts where each part is an
        #    independent set (no edges within), and edges only between parts. This is a
        #    3-partite graph, which is automatically K4-free (any 4 vertices must have at
        #    least 2 in the same part, hence not adjacent).
        #
        #    But 3-partite graphs on 18 vertices with parts of 6 each... the largest
        #    triangle-free induced subgraph could be large.
        #
        #    Alternative: Use 3 cliques of size 3 (9 vertices) plus 9 more vertices arranged
        #    to minimize triangle-free sets. This is getting complex.
        #
        #    Let me try a different structural approach: start from known good constructions.
        #    The Paley(17) + isolated vertex approach scored well. Let me combine that with
        #    the clique-partition idea.
        
        # Actually, let me just verify K4-freedom directly for any given adj_mask
        adj = build_adj_from_cliques(A, B, C)
        return is_k4_free(adj)

    def search_clique_patterns(deadline):
        """Search over inter-clique edge patterns for 3 cliques of 6 vertices.
        
        Key insight: For K4-freedom, we need to ensure no 4 vertices are mutually adjacent.
        Within a clique of 6, any 4 form a K4. So we CANNOT use cliques of size >= 4.
        
        Revised approach: Use 6 cliques of size 3 (triangles). Each triangle is a K3.
        A K4 would need 4 mutually adjacent vertices. If we have 6 triangles, a K4 could
        consist of vertices from different triangles that are all connected to each other.
        
        Actually, let me use a completely different structural approach:
        Start from Paley(17) + isolated vertex as the base, then use local search.
        """
        # This function will be replaced - see the main search below
        return None

    def make_paley17_plus_iso():
        """Create Paley(17) graph on vertices 0..16 plus isolated vertex 17.
        Paley(17): vertices 0..16, edge (i,j) iff (j-i) mod 17 is a quadratic residue mod 17.
        QRs mod 17: {1, 2, 4, 8, 9, 13, 15, 16} (i.e., {1^2, 2^2, 3^2, ..., 8^2} mod 17)"""
        qr = set()
        for k in range(1, 9):
            qr.add((k * k) % 17)
        adj = [0] * N
        for i in range(17):
            for j in range(17):
                if i != j and ((j - i) % 17) in qr:
                    adj[i] |= (1 << j)
        # Vertex 17 is isolated
        return adj

    def random_k4_free_graph():
        """Generate a random K4-free graph by adding edges one at a time."""
        adj_mask = [0] * N
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.5:
                    creates_k4 = False
                    for w in range(N):
                        if w == u or w == v:
                            continue
                        if not ((adj_mask[u] >> w) & 1) or not ((adj_mask[v] >> w) & 1):
                            continue
                        for x in range(N):
                            if x in (u, v, w):
                                continue
                            if ((adj_mask[u] >> x) & 1 and (adj_mask[v] >> x) & 1
                                    and (adj_mask[w] >> x) & 1
                                    and (adj_mask[x] >> w) & 1):
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                    if not creates_k4:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
        return adj_mask

    def flip(adj_mask, u, v):
        if (adj_mask[u] >> v) & 1:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
        else:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

    def local_search(adj_mask, current_score, deadline):
        """Structured local search: only accept moves that improve or maintain score."""
        local_best = current_score
        best_local_mask = adj_mask[:]
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            # Try all possible edge flips
            for u in range(N):
                for v in range(u + 1, N):
                    if time.time() >= deadline:
                        return local_best, best_local_mask
                    
                    flip(adj_mask, u, v)
                    
                    if not is_k4_free(adj_mask):
                        flip(adj_mask, u, v)
                        continue
                    
                    new_score = max_tri_free(adj_mask)
                    
                    if new_score < local_best:
                        local_best = new_score
                        best_local_mask = adj_mask[:]
                        improved = True
                        no_improve_count = 0
                        # Continue from this improved state
                        break
                    else:
                        flip(adj_mask, u, v)  # revert
                else:
                    continue
                break
            
            if not improved:
                no_improve_count += 1
                # Try a random restart perturbation
                if no_improve_count % 5 == 0:
                    for _ in range(3):
                        u, v = rng.sample(range(N), 2)
                        flip(adj_mask, u, v)
                        if not is_k4_free(adj_mask):
                            flip(adj_mask, u, v)
        
        return local_best, best_local_mask

    def search_3partite(deadline):
        """Search over 3-partite graphs with parts of 6 vertices each.
        A 3-partite graph is automatically K4-free. We search for the edge pattern
        between parts that minimizes the largest triangle-free induced subgraph."""
        best_score = N + 1
        best_adj = None
        
        # Represent inter-part edges as three 36-bit masks
        # Part A: vertices 0-5, Part B: vertices 6-11, Part C: vertices 12-17
        
        # Start with a random 3-partite graph
        for attempt in range(20):
            if time.time() >= deadline:
                break
            
            AB = 0
            BC = 0
            AC = 0
            for i in range(6):
                for j in range(6):
                    if rng.random() < 0.5:
                        AB |= (1 << (i * 6 + j))
                    if rng.random() < 0.5:
                        BC |= (1 << (i * 6 + j))
                    if rng.random() < 0.5:
                        AC |= (1 << (i * 6 + j))
            
            adj = build_adj_from_cliques(AB, BC, AC)
            # Note: build_adj_from_cliques treats them as cliques, but for 3-partite
            # we want NO intra-part edges. Let me fix this.
            
            # Build 3-partite adj properly
            adj = [0] * N
            for i in range(6):
                for j in range(6):
                    if (AB >> (i * 6 + j)) & 1:
                        adj[i] |= (1 << (6 + j))
                        adj[6 + j] |= (1 << i)
            for i in range(6):
                for j in range(6):
                    if (BC >> (i * 6 + j)) & 1:
                        adj[6 + i] |= (1 << (12 + j))
                        adj[12 + j] |= (1 << (6 + i))
            for i in range(6):
                for j in range(6):
                    if (AC >> (i * 6 + j)) & 1:
                        adj[i] |= (1 << (12 + j))
                        adj[12 + j] |= (1 << i)
            
            score = max_tri_free(adj)
            if score < best_score:
                best_score = score
                best_adj = adj[:]
            
            # Local search on the 3-partite structure
            if time.time() < deadline - 2:
                local_best, local_mask = local_search(adj, score, min(deadline, time.time() + 5))
                if local_best < best_score:
                    best_score = local_best
                    best_adj = local_mask[:]
        
        return best_score, best_adj

    best_score = N + 1
    best_mask = None

    # Strategy 1: Paley(17) + isolated vertex
    paley_adj = make_paley17_plus_iso()
    if is_k4_free(paley_adj):
        paley_score = max_tri_free(paley_adj)
        if paley_score < best_score:
            best_score = paley_score
            best_mask = paley_adj[:]
        
        # Local search from Paley
        if time.time() < deadline - 10:
            local_best, local_mask = local_search(paley_adj, paley_score, min(deadline, time.time() + 30))
            if local_best < best_score:
                best_score = local_best
                best_mask = local_mask[:]

    # Strategy 2: 3-partite search
    if time.time() < deadline - 15:
        part_score, part_adj = search_3partite(min(deadline, time.time() + 40))
        if part_score < best_score:
            best_score = part_score
            best_mask = part_adj[:]

    # Strategy 3: Random K4-free graphs with local search
    for r in range(5):
        if time.time() >= deadline - 5:
            break
        adj = random_k4_free_graph()
        if not is_k4_free(adj):
            continue
        cur = max_tri_free(adj)
        if cur < best_score:
            best_score = cur
            best_mask = adj[:]
        
        if time.time() < deadline - 5:
            local_best, local_mask = local_search(adj, cur, min(deadline, time.time() + 10))
            if local_best < best_score:
                best_score = local_best
                best_mask = local_mask[:]

    # Final polish
    if best_mask is not None and time.time() < deadline - 1:
        final_score = max_tri_free(best_mask)
        final_best, final_mask = local_search(best_mask, final_score, deadline)
        if final_best < best_score:
            best_score = final_best
            best_mask = final_mask[:]

    if best_mask is None:
        best_mask = make_paley17_plus_iso()

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
