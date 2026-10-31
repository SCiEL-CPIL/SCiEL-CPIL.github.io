import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Build clique structure for faster neighborhood queries
    # Each maximal clique is defined by (3 positions, 3 signs)
    # A vector belongs to a clique if it contains those 3 positions with those signs
    
    # Precompute: for each vector, its support and signs
    supports = []
    signs_list = []
    for v in v4_candidates:
        sup = [j for j in range(D) if v[j] != 0]
        sgn = tuple(v[j] for j in sup)
        supports.append(sup)
        signs_list.append(sgn)
    
    # Map from (3_pos, 3_signs) to clique id
    pos3_sign3_to_clique = {}
    clique_members = []  # list of lists of vector indices
    for pos3 in itertools.combinations(range(D), 3):
        for sign3 in itertools.product([1, -1], repeat=3):
            key = (pos3, sign3)
            pos3_sign3_to_clique[key] = len(clique_members)
            clique_members.append([])
    
    # For each vector, determine which cliques it belongs to
    vec_to_cliques = [[] for _ in range(N4)]
    for i in range(N4):
        sup = supports[i]
        for pos3 in itertools.combinations(sup, 3):
            sign3 = tuple(v4_candidates[i][p] for p in pos3)
            key = (pos3, sign3)
            if key in pos3_sign3_to_clique:
                cid = pos3_sign3_to_clique[key]
                clique_members[cid].append(i)
                vec_to_cliques[i].append(cid)
    
    # Each vector has exactly 4 cliques it belongs to
    # Each clique has 16 members
    # Two vectors conflict iff they share a clique (since sharing 3 coords with same sign = in same clique)
    # Wait, that's not quite right. Let me verify.
    # If two vectors share 3 positions with same signs, they're in the same clique.
    # But they could also share 3 positions with different signs... no, the clique is defined by specific signs.
    # Actually, two vectors conflict iff their dot product > 2.
    # For weight-4 vectors, dot = sum over intersection of supports of (s1[i]*s2[i]).
    # If intersection size is m, and t of those have same sign, dot = t - (m-t) = 2t - m.
    # Conflict if 2t - m > 2.
    # m=4: 2t-4>2 => t>3 => t=4 (identical, not in our set since unique)
    # m=3: 2t-3>2 => t>2.5 => t=3 (all 3 shared have same sign)
    # m=2: 2t-2>2 => t>2 (impossible)
    # So conflict iff they share exactly 3 positions with all same signs.
    # This means: two vectors conflict iff they belong to at least one common clique.
    # And since each vector is in exactly 4 cliques, and each clique has 16 members,
    # the neighborhood of a vector v is: all other vectors in any of v's 4 cliques.
    # |N(v)| = 4 * (16-1) - (overlaps). Overlaps: two vectors in two common cliques would be counted twice.
    # Can two distinct vectors be in two common cliques? They'd need to share two different 3-subsets of their support with same signs.
    # If they share 3 positions with same signs (clique A) and also share another 3 positions with same signs (clique B),
    # then they share at least 4 positions... but weight is 4, so they'd be identical. So no overlaps for distinct vectors.
    # Thus |N(v)| = 4*15 = 60.
    
    # Verify: degree should be 60 for all vertices
    degrees = conflicts.sum(axis=1)
    # Should all be 60
    
    # Now implement efficient maximum independent set using the clique structure
    # Key insight: we can use the clique decomposition for fast neighborhood queries
    
    def get_neighbors(v):
        """Get all neighbors of vertex v using clique structure."""
        neighbors = set()
        for cid in vec_to_cliques[v]:
            for u in clique_members[cid]:
                if u != v:
                    neighbors.add(u)
        return neighbors
    
    # Strategy: Use a sophisticated greedy + local search that exploits the structure
    # The graph is 60-regular on 5280 vertices
    
    best_selected = None
    best_count = 0
    
    def greedy_min_degree(conflicts, N, rng, deadline, use_structure=True):
        """Greedy IS with minimum degree, using structure for speed if available."""
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        
        while np.any(alive):
            if time.time() > deadline - 15:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        
        return selected
    
    def local_search_fast(selected, conflicts, N, vec_to_cliques, clique_members, deadline):
        """Fast local search using clique structure for neighborhood queries."""
        selected = selected.copy()
        improved = True
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break
            improved = False
            non_selected = np.where(~selected)[0]
            if len(non_selected) == 0:
                break
            
            # Try to add any non-selected vertex with no conflicts with selected
            # Use structure: v can be added iff none of its clique-mates are selected
            for v in non_selected:
                if time.time() > deadline - 15:
                    break
                # Check if any of v's clique-mates are selected
                can_add = True
                for cid in vec_to_cliques[v]:
                    for u in clique_members[cid]:
                        if u != v and selected[u]:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    selected[v] = True
                    improved = True
                    break
            
            if not improved:
                # Try 1-swap: remove one selected, add multiple non-selected
                selected_indices = np.where(selected)[0]
                if len(selected_indices) == 0:
                    break
                
                # For each selected vertex u, find how many non-selected vertices would become addable
                best_gain = 0
                best_u = None
                for u in selected_indices:
                    if time.time() > deadline - 15:
                        break
                    # If we remove u, which non-selected vertices become addable?
                    # v becomes addable if its only selected neighbor was u
                    # i.e., all of v's clique-mates that are selected are just u
                    addable_count = 0
                    non_sel = np.where(~selected)[0]
                    for v in non_sel:
                        if time.time() > deadline - 15:
                            break
                        # Count how many selected neighbors v has
                        sel_neighbors = 0
                        for cid in vec_to_cliques[v]:
                            for w in clique_members[cid]:
                                if w != v and selected[w]:
                                    sel_neighbors += 1
                        if sel_neighbors == 1:
                            # Check if that neighbor is u
                            is_u_only = False
                            for cid in vec_to_cliques[v]:
                                if u in clique_members[cid]:
                                    is_u_only = True
                                    break
                            if is_u_only:
                                addable_count += 1
                    gain = addable_count - 1
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                
                if best_u is not None and best_gain > 0:
                    u = best_u
                    selected[u] = False
                    # Add all non-selected vertices that now have no selected neighbors
                    non_sel = np.where(~selected)[0]
                    for v in non_sel:
                        if time.time() > deadline - 15:
                            break
                        can_add = True
                        for cid in vec_to_cliques[v]:
                            for w in clique_members[cid]:
                                if w != v and selected[w]:
                                    can_add = False
                                    break
                            if not can_add:
                                break
                        if can_add:
                            selected[v] = True
                    improved = True
        
        return selected
    
    # Run multiple strategies with different random seeds
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        rng_r = np.random.default_rng(1000 + restart)
        sel = greedy_min_degree(conflicts, N4, rng_r, deadline)
        sel_ls = local_search_fast(sel, conflicts, N4, vec_to_cliques, clique_members, deadline)
        cnt = sel_ls.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = sel_ls.copy()
    
    # Also try spectral initialization
    if time.time() < deadline - 30:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        degree = conflicts.sum(axis=1)
        L = np.diag(degree) - conflicts.astype(np.float64)
        L_sparse = csr_matrix(L)
        
        try:
            eigvals, eigvecs = eigsh(L_sparse, k=3, sigma=0, which='LM')
            idx = np.argsort(eigvals)
            eigvecs = eigvecs[:, idx]
            fiedler = eigvecs[:, 1]
        except Exception:
            rng = np.random.default_rng(42)
            fiedler = rng.standard_normal(N4)
        
        # Use Fiedler vector to order vertices
        order = np.argsort(fiedler)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
            alive[to_remove] = False
        sel_ls = local_search_fast(selected, conflicts, N4, vec_to_cliques, clique_members, deadline)
        cnt = sel_ls.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = sel_ls.copy()
    
    # More random restarts if time permits
    restart_num = 10
    while time.time() < deadline - 15:
        rng_r = np.random.default_rng(5000 + restart_num)
        order = rng_r.permutation(N4)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
            alive[to_remove] = False
        sel_ls = local_search_fast(selected, conflicts, N4, vec_to_cliques, clique_members, deadline)
        cnt = sel_ls.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = sel_ls.copy()
        restart_num += 1
    
    if best_selected is None:
        best_selected = np.zeros(N4, dtype=bool)
    
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors (they're all compatible with V4 and each other)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
