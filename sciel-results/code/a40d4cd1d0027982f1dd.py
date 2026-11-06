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
                verts = []
                m = R_mask
                while m:
                    lsb = m & (-m)
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                if len(verts) >= 2:
                    max_cliques.append(frozenset(verts))
            return
        if P_mask == 0:
            return
        PX = P_mask | X_mask
        best_v = -1
        best_count = -1
        m = PX
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            cnt = (P_mask & adj_mask[v]).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_v = v
            m ^= lsb
        
        if best_v >= 0:
            candidates = P_mask & ~adj_mask[best_v]
        else:
            candidates = P_mask
        
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            v_bit = 1 << v
            new_R = R_mask | v_bit
            new_P = P_mask & adj_mask[v]
            new_X = X_mask & adj_mask[v]
            expand(new_R, new_P, new_X)
            P_mask &= ~v_bit
            X_mask |= v_bit
            m &= ~(1 << v)
    
    expand(0, (1 << n) - 1, 0)
    return max_cliques
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
        if P_mask == 0:
            return
        # Choose pivot: vertex in P ∪ X with max neighbors in P
        # Branch only on vertices in P not adjacent to pivot
        PX = P_mask | X_mask
        best_v = -1
        best_count = -1
        m = PX
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            cnt = (P_mask & adj_mask[v]).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_v = v
            m ^= lsb
        
        # Candidates: vertices in P not adjacent to best_v
        if best_v >= 0:
            candidates = P_mask & ~adj_mask[best_v]
        else:
            candidates = P_mask
        
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            v_bit = 1 << v
            new_R = R_mask | v_bit
            new_P = P_mask & adj_mask[v]
            new_X = X_mask & adj_mask[v]
            expand(new_R, new_P, new_X)
            P_mask &= ~v_bit
            X_mask |= v_bit
            m &= ~(1 << v)
    
    expand(0, (1 << n) - 1, 0)
    return max_cliques

def _greedy_transversal(cliques, n, vertex_clique_mask, rng):
    """Greedy set cover: repeatedly pick vertex in most uncovered cliques."""
    if not cliques:
        return set()
    
    num_cliques = len(cliques)
    uncovered = (1 << num_cliques) - 1
    cover = set()
    
    while uncovered:
        best_v = -1
        best_count = -1
        best_rnd = 2.0
        for v in range(n):
            if v in cover:
                continue
            count = (vertex_clique_mask[v] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_rnd = rng.random()
                best_v = v
            elif count == best_count and count > 0:
                r = rng.random()
                if r < best_rnd:
                    best_rnd = r
                    best_v = v
        if best_v == -1 or best_count == 0:
            break
        cover.add(best_v)
        uncovered &= ~vertex_clique_mask[best_v]
    
    return cover

def _local_search(cover, cliques, n, vertex_clique_mask, rng, max_iters=5000):
    """Try to improve the transversal via removal, 1-1 swap, and 2-1 moves using bitmasks."""
    cover = set(cover)
    num_cliques = len(cliques)
    
    if num_cliques == 0:
        return cover
    
    # For each clique, which vertices are in it (as set)
    clique_verts = [set(c) for c in cliques]
    
    # Track which cliques are covered by the current cover
    # Use a counter per clique: how many cover vertices hit it
    clique_count = [0] * num_cliques
    for v in cover:
        mask = vertex_clique_mask[v]
        m = mask
        while m:
            lsb = m & (-m)
            ci = lsb.bit_length() - 1
            clique_count[ci] += 1
            m ^= lsb
    
    def is_valid(cov):
        for ci in range(num_cliques):
            if clique_count[ci] == 0:
                return False
        return True
    
    def try_remove(v):
        """Try removing v. Returns True if valid."""
        mask = vertex_clique_mask[v]
        m = mask
        while m:
            lsb = m & (-m)
            ci = lsb.bit_length() - 1
            if clique_count[ci] == 1:
                # Can't remove, would uncover clique ci
                # Restore counts
                m2 = mask
                while m2:
                    lsb2 = m2 & (-m2)
                    ci2 = lsb2.bit_length() - 1
                    if ci2 == ci:
                        break
                    clique_count[ci2] += 1
                    m2 ^= lsb2
                return False
            clique_count[ci] -= 1
            m ^= lsb
        # All good, actually remove
        cover.discard(v)
        return True
    
    def undo_remove(v):
        """Undo a removal of v."""
        mask = vertex_clique_mask[v]
        m = mask
        while m:
            lsb = m & (-m)
            ci = lsb.bit_length() - 1
            clique_count[ci] += 1
            m ^= lsb
        cover.add(v)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try removal: for each vertex in cover, check if removing it keeps validity
        for v in sorted(cover):
            mask = vertex_clique_mask[v]
            m = mask
            can_remove = True
            while m:
                lsb = m & (-m)
                ci = lsb.bit_length() - 1
                if clique_count[ci] == 1:
                    can_remove = False
                    break
                m ^= lsb
            
            if can_remove:
                # Remove v
                m2 = mask
                while m2:
                    lsb2 = m2 & (-m2)
                    ci2 = lsb2.bit_length() - 1
                    clique_count[ci2] -= 1
                    m2 ^= lsb2
                cover.discard(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-1 swap: remove v, add u (same size, may enable future removals)
        for v in sorted(cover):
            # Find cliques where v is the only cover
            mask_v = vertex_clique_mask[v]
            only_v_cliques = 0
            m = mask_v
            while m:
                lsb = m & (-m)
                ci = lsb.bit_length() - 1
                if clique_count[ci] == 1:
                    only_v_cliques |= (1 << ci)
                m ^= lsb
            
            if only_v_cliques == 0:
                continue
            
            for u in range(n):
                if u in cover:
                    continue
                if (vertex_clique_mask[u] & only_v_cliques) == only_v_cliques:
                    # Do the swap
                    # Remove v
                    m2 = mask_v
                    while m2:
                        lsb2 = m2 & (-m2)
                        ci2 = lsb2.bit_length() - 1
                        clique_count[ci2] -= 1
                        m2 ^= lsb2
                    cover.discard(v)
                    # Add u
                    m3 = vertex_clique_mask[u]
                    while m3:
                        lsb3 = m3 & (-m3)
                        ci3 = lsb3.bit_length() - 1
                        clique_count[ci3] += 1
                        m3 ^= lsb3
                    cover.add(u)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-1 move: remove two vertices, add one
        cover_list = sorted(cover)
        for i in range(len(cover_list)):
            for j in range(i + 1, len(cover_list)):
                v1, v2 = cover_list[i], cover_list[j]
                mask1 = vertex_clique_mask[v1]
                mask2 = vertex_clique_mask[v2]
                
                # Check if removing both is valid (all cliques still covered)
                needed = 0
                m = mask1 | mask2
                while m:
                    lsb = m & (-m)
                    ci = lsb.bit_length() - 1
                    # Count how many of v1,v2 cover this clique
                    cnt = 0
                    if mask1 & (1 << ci):
                        cnt += 1
                    if mask2 & (1 << ci):
                        cnt += 1
                    if clique_count[ci] - cnt == 0:
                        needed |= (1 << ci)
                    m ^= lsb
                
                if needed == 0:
                    # Just remove both
                    m1 = mask1
                    while m1:
                        lsb1 = m1 & (-m1)
                        ci1 = lsb1.bit_length() - 1
                        clique_count[ci1] -= 1
                        m1 ^= lsb1
                    m2 = mask2
                    while m2:
                        lsb2 = m2 & (-m2)
                        ci2 = lsb2.bit_length() - 1
                        clique_count[ci2] -= 1
                        m2 ^= lsb2
                    cover.discard(v1)
                    cover.discard(v2)
                    improved = True
                    break
                
                # Try adding one vertex to cover needed
                for u in range(n):
                    if u in cover:
                        continue
                    if (vertex_clique_mask[u] & needed) == needed:
                        # Remove v1, v2 and add u
                        m1 = mask1
                        while m1:
                            lsb1 = m1 & (-m1)
                            ci1 = lsb1.bit_length() - 1
                            clique_count[ci1] -= 1
                            m1 ^= lsb1
                        m2 = mask2
                        while m2:
                            lsb2 = m2 & (-m2)
                            ci2 = lsb2.bit_length() - 1
                            clique_count[ci2] -= 1
                            m2 ^= lsb2
                        cover.discard(v1)
                        cover.discard(v2)
                        m3 = vertex_clique_mask[u]
                        while m3:
                            lsb3 = m3 & (-m3)
                            ci3 = lsb3.bit_length() - 1
                            clique_count[ci3] += 1
                            m3 ^= lsb3
                        cover.add(u)
                        improved = True
                        break
                if improved:
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
        
        # Precompute vertex_clique_mask once
        num_cliques = len(cliques)
        vertex_clique_mask = [0] * n
        for i, c in enumerate(cliques):
            bit = 1 << i
            for v in c:
                vertex_clique_mask[v] |= bit
        
        # Multiple greedy restarts with randomization
        best_cover = None
        best_size = n + 1
        rng = random.Random(seed + 1000)
        
        # Scale restarts and local search iterations with problem size
        if n <= 30:
            num_restarts = 30
            ls_iters = 5000
        elif n <= 35:
            num_restarts = 35
            ls_iters = 4500
        else:
            num_restarts = 40
            ls_iters = 4000
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Fast greedy with bitmask and random tie-breaking
            cover = _greedy_transversal(cliques, n, vertex_clique_mask, rng)
            
            # Local search
            cover = _local_search(cover, cliques, n, vertex_clique_mask, rng, max_iters=ls_iters)
            
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
