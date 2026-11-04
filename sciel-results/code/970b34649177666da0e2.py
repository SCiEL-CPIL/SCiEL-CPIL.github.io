"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 115.0  # hard wall-clock safety margin before the 120 s limit

def construct():
    t0 = time.perf_counter()

    def elapsed():
        return time.perf_counter() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    #    There are exactly C(11,2)*2^2 = 220 such vectors, and they satisfy
    #    min pairwise distance (sqrt(2) for shared coordinate) == max norm (sqrt(2)).
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    best = base_arr
    best_k = base_arr.shape[0]

    def update_best(arr):
        nonlocal best, best_k
        if arr is None:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 2) Structural approach: Model as independent set on supports of size 4.
    #    We restrict to vectors with exactly 4 non-zero entries, each +/-1.
    #    Norm^2 = 4. Max norm is 2.
    #    Condition: dist(u, v) >= 2  =>  ||u-v||^2 >= 4.
    #    ||u-v||^2 = 4 + 4 - 2<u,v> = 8 - 2<u,v>.
    #    So we need 8 - 2<u,v> >= 4  =>  <u,v> <= 2.
    #    
    #    Let S_u and S_v be the supports (sets of indices).
    #    <u,v> = sum_{i in S_u ∩ S_v} u_i v_i.
    #    Since u_i, v_i in {1, -1}, each term is +/-1.
    #    Let k = |S_u ∩ S_v|. Then <u,v> is an integer with same parity as k,
    #    and |<u,v>| <= k.
    #    
    #    If k=0, <u,v>=0 <= 2. OK.
    #    If k=1, <u,v> in {-1, 1}. <= 2. OK.
    #    If k=2, <u,v> in {-2, 0, 2}. Max is 2. OK.
    #    If k=3, <u,v> in {-3, -1, 1, 3}. Max is 3. FAIL if <u,v>=3.
    #    If k=4, <u,v> in {-4, -2, 0, 2, 4}. Max is 4. FAIL if <u,v>=4 or 2?
    #       Wait, if k=4, S_u = S_v. Then u and v differ only in signs.
    #       <u,v> = sum u_i v_i. If u=v, <u,v>=4. FAIL.
    #       If u=-v, <u,v>=-4. OK.
    #       If 2 signs match, 2 mismatch, <u,v> = 2 - 2 = 0. OK.
    #       If 3 match, 1 mismatch, <u,v> = 3 - 1 = 2. OK.
    #       If 1 match, 3 mismatch, <u,v> = 1 - 3 = -2. OK.
    #       So if S_u = S_v, we can include at most one sign pattern?
    #       Actually, if we include u, we cannot include v if <u,v> > 2.
    #       <u,v> = 4 only if u=v. <u,v>=2 is allowed.
    #       So for a fixed support, we can include multiple vectors as long as
    #       pairwise <u,v> <= 2.
    #       Max <u,v> for distinct vectors with same support is 2 (3 signs match).
    #       So we can include ALL 16 sign patterns for a given support?
    #       Let's check: u=(1,1,1,1), v=(1,1,1,-1). <u,v> = 3-1 = 2. OK.
    #       u=(1,1,1,1), v=(1,1,-1,-1). <u,v> = 2-2 = 0. OK.
    #       u=(1,1,1,1), v=(-1,-1,-1,-1). <u,v> = -4. OK.
    #       Yes, for a single support, all 16 vectors are mutually compatible.
    #    
    #    Now consider two different supports S1, S2.
    #    If |S1 ∩ S2| <= 2, then for ANY sign patterns, <u,v> <= |S1 ∩ S2| <= 2. OK.
    #    If |S1 ∩ S2| = 3, then <u,v> can be 3. This happens if the 3 shared
    #    coordinates have matching signs.
    #    So, if we select a set of supports, we must ensure that for any pair
    #    of supports with intersection size 3, we do NOT select sign patterns
    #    that match on the 3 shared coordinates.
    #    
    #    However, the inspiration suggests a simpler model:
    #    "Conflict if intersection size is 3".
    #    This implies that if we treat the *supports* as nodes, and edges as
    #    conflicts, then finding an independent set of supports gives us a set
    #    of supports where NO pair intersects in 3.
    #    If no pair intersects in 3 (or 4), then for any selected supports,
    #    their intersection is <= 2.
    #    Then ANY sign patterns are compatible!
    #    So the problem reduces to: Find the Maximum Independent Set (MIS) of the
    #    graph where nodes are 4-subsets of {0..10} and edges connect subsets
    #    with intersection size 3 (which implies intersection size 3, since
    #    intersection size 4 means same subset, and we don't have self-loops,
    #    and intersection size > 4 is impossible for distinct 4-sets).
    #    
    #    Wait, can two distinct 4-sets intersect in 4? No.
    #    So the only "bad" intersection is size 3.
    #    If we find a set of supports with pairwise intersection <= 2,
    #    then we can take ALL 16 sign patterns for each support.
    #    Total vectors = 16 * |MIS|.
    #    
    #    Let's estimate the size of MIS.
    #    Total nodes = C(11,4) = 330.
    #    Degree of a node: How many 4-sets intersect a given 4-set in exactly 3?
    #    Pick 3 from the 4 (4 ways), pick 1 from the remaining 7 (7 ways).
    #    So degree = 4 * 7 = 28.
    #    This is a relatively sparse graph.
    #    A greedy MIS might get a decent fraction.
    #    If we get ~100 supports, we get 1600 vectors.
    #    Baseline is 220. 1600 is much better.
    # ------------------------------------------------------------------

    # Generate all supports
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)

    # Build conflict graph
    # Conflict if intersection size is 3
    adj = [set() for _ in range(N_S)]
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)

    # Convert adj to list of lists for faster access
    adj_list = [list(s) for s in adj]
    
    # Precompute adjacency as boolean matrix for fast vectorized operations
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in adj_list[i]:
            adj_bool[i][j] = True
            adj_bool[j][i] = True

    # ------------------------------------------------------------------
    # 3) Find Maximum Independent Set
    #    Use a combination of greedy initialization and local search.
    # ------------------------------------------------------------------

    def greedy_min_degree():
        """Greedy min-degree heuristic for independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining and not is_deadline():
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for nb in adj_list[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            to_remove = {min_node} | {nb for nb in adj_list[min_node] if nb in remaining}
            remaining -= to_remove
        
        return s

    def local_search_fast(init_set, max_iter=1000):
        """Fast local search with vectorized blocked set computation."""
        current_set = set(init_set)
        best_local = set(current_set)
        best_local_size = len(current_set)
        
        # Pre-compute in_set vector for faster updates? 
        # For now, recompute or update incrementally.
        # Given N_S=330, recomputing is fast enough.
        
        for it in range(max_iter):
            if is_deadline():
                break
            
            # Compute in_set and blocked using vectorized ops
            in_set = np.zeros(N_S, dtype=bool)
            for v in current_set:
                in_set[v] = True
            
            # blocked = OR of all rows of adj_bool for vertices in current_set
            if current_set:
                blocked = np.any(adj_bool[in_set], axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            # Candidates: not in set and not blocked
            candidates_mask = ~(in_set | blocked)
            cand_indices = np.where(candidates_mask)[0]
            
            if len(cand_indices) > 0:
                # Add a random candidate
                v_add = int(cand_indices[np.random.randint(len(cand_indices))])
                current_set.add(v_add)
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local = set(current_set)
                continue
            
            # Stuck: try 1-swap
            if current_set:
                v_remove = list(current_set)[np.random.randint(len(current_set))]
                current_set.remove(v_remove)
                
                # Recompute
                in_set2 = np.zeros(N_S, dtype=bool)
                for v in current_set:
                    in_set2[v] = True
                
                if current_set:
                    blocked2 = np.any(adj_bool[in_set2], axis=0)
                else:
                    blocked2 = np.zeros(N_S, dtype=bool)
                
                candidates2_mask = ~(in_set2 | blocked2)
                cand_indices2 = np.where(candidates2_mask)[0]
                
                if len(cand_indices2) > 0:
                    v_add = int(cand_indices2[np.random.randint(len(cand_indices2))])
                    current_set.add(v_add)
                    if len(current_set) > best_local_size:
                        best_local_size = len(current_set)
                        best_local = set(current_set)
                else:
                    # Revert
                    current_set.add(v_remove)
        
        return best_local

    # Run multiple restarts with different random seeds
    np.random.seed(42)
    
    best_size = 0
    best_set = set()
    
    # Time budget allocation
    # Greedy is fast. Local search is heavier.
    # We will run greedy once, then improve it with local search.
    # Then restart with random perturbations.
    
    initial_set = greedy_min_degree()
    current_best_set = local_search_fast(initial_set, max_iter=500)
    best_size = len(current_best_set)
    best_set = set(current_best_set)

    # Further improvements with random restarts if time permits
    num_restarts = 50
    for r in range(num_restarts):
        if is_deadline():
            break
        
        # Perturb the current best set
        perturbed = set(best_set)
        # Remove a few random elements
        if len(perturbed) > 10:
            to_remove = list(perturbed)
            np.random.shuffle(to_remove)
            for _ in range(3):
                if to_remove:
                    perturbed.remove(to_remove.pop())
        
        improved = local_search_fast(perturbed, max_iter=200)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = set(improved)

    # ------------------------------------------------------------------
    # 4) Construct vectors from the best independent set of supports
    # ------------------------------------------------------------------
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if vectors:
        final_vectors = np.array(vectors, dtype=np.int64)
        update_best(final_vectors)
    else:
        # Fallback to baseline if something went wrong
        update_best(base_arr)

    return best
