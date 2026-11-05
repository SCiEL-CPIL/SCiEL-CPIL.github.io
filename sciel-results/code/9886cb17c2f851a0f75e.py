import time
import random
import numpy as np

# Deadline guard - evaluator kills at 150s, use 130s limit
_DEADLINE = time.time() + 130.0

def _has_c5(n, nmask):
    """
    Check if graph (given by neighbor bitmasks) contains a C5.
    For each edge (u,v), check if there exist a,b,c with
    u-a-b-c-v forming a 5-cycle.
    """
    for u in range(n):
        for v in range(u+1, n):
            if not (nmask[u] >> v) & 1:
                continue
            # a in N(u) \ {v}
            a_mask = nmask[u] & ~(1 << v)
            while a_mask:
                a_bit = a_mask & (-a_mask)
                a = a_bit.bit_length() - 1
                a_mask ^= a_bit
                # b in N(a) \ {u, v}
                b_mask = nmask[a] & ~(1 << u) & ~(1 << v)
                while b_mask:
                    b_bit = b_mask & (-b_mask)
                    b = b_bit.bit_length() - 1
                    b_mask ^= b_bit
                    # c in N(b) \ {a, u, v} and c in N(v)
                    c_mask = nmask[b] & ~(1 << a) & ~(1 << u) & ~(1 << v)
                    if c_mask & nmask[v]:
                        return True
    return False

def _has_c5_fast(n, nmask):
    """
    Faster C5 check using adjacency matrix with numpy.
    C5 exists iff there's a cycle of length 5.
    Use the fact that C5 exists iff there exist u,v,w,x,y with
    edges (u,v),(v,w),(w,x),(x,y),(y,u) all present and all distinct.
    
    Alternative: For each pair (u,v) that is an edge, check if there's a path
    of length 3 from u to v avoiding the edge (u,v) directly, but through
    intermediate vertices not equal to u or v.
    """
    # For each edge (u,v), check if there exists a path u-a-b-c-v
    # where a,b,c are distinct and not u,v
    # This is equivalent to: in the graph with edge (u,v) removed,
    # is there a path of length 3 from u to v?
    # Path of length 3 from u to v: u-a-b-c-v
    # a in N(u)\{v}, b in N(a)\{u,v}, c in N(b)\{a,u,v} ∩ N(v)
    
    for u in range(n):
        for v in range(u+1, n):
            if not (nmask[u] >> v) & 1:
                continue
            a_mask = nmask[u] & ~(1 << v)
            while a_mask:
                a_bit = a_mask & (-a_mask)
                a = a_bit.bit_length() - 1
                a_mask ^= a_bit
                b_mask = nmask[a] & ~(1 << u) & ~(1 << v)
                while b_mask:
                    b_bit = b_mask & (-b_mask)
                    b = b_bit.bit_length() - 1
                    b_mask ^= b_bit
                    c_mask = nmask[b] & ~(1 << a) & ~(1 << u) & ~(1 << v)
                    if c_mask & nmask[v]:
                        return True
    return False

def _max_independent(n, nmask, target=9):
    """
    Check if there's an independent set of size >= target.
    Independent set in G = clique in complement.
    Use branch-and-bound with bitmasks.
    """
    # Complement neighbor masks
    comp_mask = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_mask[i] = full_mask & ~(1 << i) & ~nmask[i]
    
    found = [False]
    
    def search(R_size, P_mask, X_mask):
        if found[0]:
            return
        if R_size >= target:
            found[0] = True
            return
        if P_mask == 0:
            return
        
        # Upper bound
        if R_size + bin(P_mask).count('1') < target:
            return
        
        # Pivot: pick vertex in P∪X with max complement-degree in P
        U = P_mask | X_mask
        if U == 0:
            return
        best_pivot = -1
        best_deg = -1
        um = U
        while um:
            ub = um & (-um)
            uv = ub.bit_length() - 1
            um ^= ub
            deg = bin(comp_mask[uv] & P_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_pivot = uv
        
        # Candidates: vertices in P NOT adjacent to pivot in comp
        if best_pivot >= 0:
            candidates = P_mask & ~comp_mask[best_pivot]
        else:
            candidates = P_mask
        
        cm = candidates
        while cm:
            cb = cm & (-cm)
            cv = cb.bit_length() - 1
            cm ^= cb
            
            if found[0]:
                return
            
            new_P = P_mask & comp_mask[cv]
            new_X = X_mask & comp_mask[cv]
            
            search(R_size + 1, new_P, new_X)
            
            P_mask = P_mask & ~cb
            X_mask = X_mask | cb
    
    search(0, full_mask, 0)
    return found[0]

def _validate(n, edges):
    """Validate: no C5 and alpha <= 8."""
    nmask = [0] * n
    for u, v in edges:
        nmask[u] |= (1 << v)
        nmask[v] |= (1 << u)
    if _has_c5(n, nmask):
        return False
    if _max_independent(n, nmask, 9):
        return False
    return True

def _make_k4_disjoint(num_blocks):
    """Create graph of num_blocks disjoint K4s."""
    n = 4 * num_blocks
    edges = []
    for b in range(num_blocks):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                edges.append([base+i, base+j])
    return n, edges

def _build_nmask(n, edges):
    """Build neighbor bitmask list from edge list."""
    nmask = [0] * n
    for u, v in edges:
        nmask[u] |= (1 << v)
        nmask[v] |= (1 << u)
    return nmask

def _edges_from_nmask(n, nmask):
    """Extract edge list from neighbor bitmasks."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (nmask[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    
    # Start with the known valid 32-vertex graph: 8 disjoint K4s
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                best_edges.append([base+i, base+j])
    
    # Try to extend to 33, 34, 35 vertices
    for target_n in [33, 34, 35]:
        if time.time() > _DEADLINE - 20:
            break
        
        n = target_n
        # Build adjacency as bitmasks
        nmask = [0] * n
        # Initialize with 8 K4s
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    nmask[base+i] |= (1 << (base+j))
                    nmask[base+j] |= (1 << (base+i))
        
        found = False
        max_iters = 300000
        
        # Strategy: add edges to cover extra vertices
        # Each extra vertex must not be able to join an independent set of size 8
        # In the 8-K4 graph, an independent set of size 8 picks one vertex from each K4
        # An extra vertex v is "dangerous" if it's non-adjacent to at least one vertex in each K4
        # To make v safe, for each K4, v must be adjacent to at least 3 vertices in that K4
        # (so that any independent set picking one from each K4 that includes v's non-neighbors
        #  would need v to be non-adjacent to the picked vertex, but if v is adjacent to 3 of 4,
        #  only 1 vertex in the K4 can be picked with v in the independent set)
        
        # Actually, let's think more carefully.
        # In the 8-K4 graph, max independent set is 8 (one from each K4).
        # Adding a new vertex v: if v has no edges to existing vertices, then
        # we can pick v plus one from each K4 = independent set of size 9. Bad.
        # If v is adjacent to some vertices, then v can only be in an independent set
        # with vertices it's NOT adjacent to.
        # For v to NOT be in any independent set of size 9:
        #   For every choice of one vertex from each K4 (8 vertices total),
        #   v must be adjacent to at least one of them.
        # This means: for each K4, v must be adjacent to at least 1 vertex? No...
        # Actually: v is in an independent set of size 9 iff there exists a choice
        # of one vertex from each K4 such that v is non-adjacent to all 8 chosen.
        # So v is safe iff: for EVERY choice of one vertex from each K4,
        # v is adjacent to at least one chosen vertex.
        # Equivalently: the set of vertices v is non-adjacent to does NOT contain
        # an independent set of size 8 in the 8-K4 graph.
        # In the 8-K4 graph, an independent set of size 8 is exactly one vertex from each K4.
        # So v is safe iff: the non-neighbors of v do NOT contain one vertex from each K4.
        # I.e., there exists at least one K4 in which v is adjacent to ALL 4 vertices.
        
        # Wait, that's not quite right either. Let me reconsider.
        # Non-neighbors of v form a set S. v is in an independent set of size 9 iff
        # S contains an independent set of size 8 in the original 8-K4 graph.
        # An independent set of size 8 in 8 disjoint K4s is one vertex from each K4.
        # So S contains such a set iff for every K4, S intersects that K4 (i.e., v is
        # non-adjacent to at least one vertex in every K4).
        # Therefore v is safe iff: there exists a K4 in which v is adjacent to ALL 4 vertices.
        
        # Hmm, but that's a strong condition. Let me verify with a simpler case.
        # Actually wait - the independent set doesn't have to be exactly one from each K4.
        # In 8 disjoint K4s, the maximum independent set is 8, and any maximum independent
        # set picks exactly one from each K4. But could there be an independent set of size 8
        # that doesn't pick from all K4s? No, because each K4 contributes at most 1 to an
        # independent set, so size 8 requires one from each.
        
        # So the condition is: v is safe iff there exists at least one K4 block where
        # v is adjacent to all 4 vertices in that block.
        
        # But wait, we're adding multiple extra vertices. They might interact with each other too.
        # Let's handle them one at a time and check the full condition.
        
        # Better approach: just use randomized search with validation.
        
        for it in range(max_iters):
            if time.time() > _DEADLINE - 15:
                break
            
            # Try to add an edge that helps cover extra vertices
            # Priority: connect extra vertices to K4 vertices
            extra_verts = list(range(32, n))
            if extra_verts and random.random() < 0.7:
                u = random.choice(extra_verts)
                # Pick a K4 to connect to - prefer K4s where u has few connections
                k4_idx = random.randint(0, 7)
                base = 4 * k4_idx
                # Pick a vertex in this K4 that u is not yet connected to
                candidates = [base+i for i in range(4) if not (nmask[u] >> (base+i)) & 1]
                if not candidates:
                    continue
                v = random.choice(candidates)
            else:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
            
            if u == v or (nmask[u] >> v) & 1:
                continue
            
            # Try adding edge (u,v)
            nmask[u] |= (1 << v)
            nmask[v] |= (1 << u)
            
            # Check C5 first (cheaper)
            if _has_c5(n, nmask):
                nmask[u] &= ~(1 << v)
                nmask[v] &= ~(1 << u)
                continue
            
            # Check alpha
            if not _max_independent(n, nmask, 9):
                found = True
                best_n = n
                best_edges = _edges_from_nmask(n, nmask)
                break
            
            # Alpha still too large, revert
            nmask[u] &= ~(1 << v)
            nmask[v] &= ~(1 << u)
        
        if found:
            continue
        else:
            # Try a different strategy: start from scratch with a different construction
            # Try 7 K4s + some other structure
            # Or try to find a valid graph via random restarts
            break
    
    # If we couldn't extend, try alternative constructions
    if best_n < 33:
        # Try random construction for 33 vertices
        for attempt in range(50):
            if time.time() > _DEADLINE - 10:
                break
            n = 33
            nmask = [0] * n
            # Start with 8 K4s
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i+1, 4):
                        nmask[base+i] |= (1 << (base+j))
                        nmask[base+j] |= (1 << (base+i))
            
            # Now add edges for vertex 32 strategically
            # Vertex 32 must be adjacent to all 4 vertices in at least one K4
            # Pick a random K4 to fully connect
            k4_idx = random.randint(0, 7)
            base = 4 * k4_idx
            for i in range(4):
                nmask[32] |= (1 << (base+i))
                nmask[base+i] |= (1 << 32)
            
            # Check if this is valid
            if _has_c5(n, nmask):
                continue
            if not _max_independent(n, nmask, 9):
                best_n = 33
                best_edges = _edges_from_nmask(n, nmask)
                break
    
    # Final validation
    if best_n >= 1:
        if not _validate(best_n, best_edges):
            # Fallback to known valid 32-vertex graph
            best_n = 32
            best_edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i+1, 4):
                        best_edges.append([base+i, base+j])
    
    return {'n': best_n, 'edges': best_edges}
