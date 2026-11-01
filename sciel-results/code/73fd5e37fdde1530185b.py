"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    # Generate all weight-4 vectors with entries in {-1, 0, 1}
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)

    vectors_v4 = np.array(v4_list, dtype=np.int64)
    N4 = len(vectors_v4)

    # Generate weight-1 vectors with entries in {-2, 0, 2}
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    N1 = len(vectors_v1)

    if N4 == 0:
        return vectors_v1

    # Compute pairwise dot products among v4 vectors
    dots = vectors_v4 @ vectors_v4.T
    # Conflict if dot > 2 (since min pairwise distance >= max norm means
    # for unit-normalized vectors, dot product should be <= 1 - 1/4 = 3/4? 
    # Actually, let's think carefully.
    #
    # We need: for all i != j, ||v_i - v_j|| >= max_k ||v_k||
    # If all vectors have norm 2 (squared norm 4), then max norm = 2.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 4 + 4 - 2*v_i.v_j = 8 - 2*v_i.v_j
    # We need ||v_i - v_j|| >= 2, so ||v_i - v_j||^2 >= 4
    # 8 - 2*v_i.v_j >= 4 => 2*v_i.v_j <= 4 => v_i.v_j <= 2
    # So conflict if dot > 2.
    
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Also need to check compatibility between v4 and v1 vectors
    # v4 has norm^2 = 4, v1 has norm^2 = 4
    # ||v4 - v1||^2 = 8 - 2*v4.v1 >= 4 => v4.v1 <= 2
    # v4.v1: v4 has four ±1 entries, v1 has one ±2 entry
    # If v1 is at position p with value 2s, and v4 has value t at position p,
    # then v4.v1 = 2s*t where t is 0 or ±1
    # If t = 0: dot = 0, OK
    # If t = 1, s = 1: dot = 2, OK (<=2)
    # If t = 1, s = -1: dot = -2, OK
    # If t = -1, s = 1: dot = -2, OK
    # If t = -1, s = -1: dot = 2, OK
    # So all v4 and v1 vectors are compatible! No conflicts between v4 and v1.

    # Also v1-v1: two different v1 vectors at positions p and q with values 2s_p and 2s_q
    # dot = 4*s_p*s_q if p=q, but they're at different positions so dot = 0
    # Wait, v1 vectors are at different positions, so dot = 0. OK.
    # Same position different sign: e.g., (2,0,...) and (-2,0,...): dot = -4, OK.
    # So v1 vectors are all mutually compatible.

    # So we just need to find a maximum independent set in the v4 conflict graph.
    # Then add all v1 vectors.

    # Strategy: Use LP relaxation + greedy + local search

    # Precompute adjacency lists
    adj = [np.where(conflicts[i])[0] for i in range(N4)]
    adj_sizes = np.array([len(a) for a in adj], dtype=np.int32)

    rng = np.random.default_rng(12345)

    def greedy_init():
        """Greedy independent set using minimum degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = adj_sizes.copy()
        
        while np.any(alive) and time.time() < deadline - 10:
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            to_remove = np.unique(np.concatenate(([chosen], adj[chosen])))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                for u in to_remove:
                    if alive[u] or u == chosen:
                        pass
                # Update degrees
                mask = np.isin(rem, to_remove)
                # Actually, just recompute or subtract
                for u in to_remove:
                    if u in rem:
                        pass
                # Simpler: for each remaining node, subtract number of removed neighbors
                for u in rem:
                    deg = np.sum(conflicts[u] & (alive | in_set))  # this is expensive
                    # Better approach:
                    pass
                
                # Let's just recompute degrees for remaining nodes
                # This is O(N^2) per iteration which might be slow
                # Alternative: maintain degrees incrementally
                pass
        
        return in_set

    def greedy_init_fast():
        """Faster greedy with incremental degree updates."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = adj_sizes.copy()
        
        while np.any(alive) and time.time() < deadline - 10:
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = adj[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            
            # Update degrees for remaining alive nodes
            for u in to_remove:
                if alive[u]:
                    alive[u] = False
                    # Decrease degree of alive neighbors of u
                    for w in adj[u]:
                        if alive[w]:
                            degrees[w] -= 1
            
            # Remove chosen from alive (already done above)
            if alive[chosen]:
                alive[chosen] = False
        
        return in_set

    def local_search(in_set, max_time=10.0):
        """Local search: try to add nodes, remove conflicting ones if beneficial."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # Precompute conflict counts for each node
        conflict_count = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[adj[i]] += 1
        
        ls_deadline = time.time() + max_time
        T = 2.0
        cooling = 0.999
        min_T = 0.1
        
        while time.time() < ls_deadline:
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            
            # Find node with minimum conflict count
            cc = conflict_count[not_in]
            min_cc = np.min(cc)
            cands = not_in[cc == min_cc]
            v = cands[rng.integers(len(cands))]
            
            k = conflict_count[v]
            delta = 1 - k  # net change in set size
            
            if delta >= 0 or rng.random() < np.exp(delta / T):
                # Accept: add v, remove its conflicting neighbors in the set
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        conflict_count[adj[u]] -= 1
                
                in_set[v] = True
                current_size += delta
                conflict_count[adj[v]] += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_in_set, best_size

    def lp_relaxation():
        """Solve LP relaxation of maximum independent set."""
        # Maximize sum(x_i) subject to x_i + x_j <= 1 for each edge (i,j), 0 <= x_i <= 1
        # c = -1 for maximization
        c = -np.ones(N4)
        
        # Build A_ub: for each edge (i,j), x_i + x_j <= 1
        edges = []
        for i in range(N4):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        if len(edges) == 0:
            return np.ones(N4)
        
        A_ub = np.zeros((len(edges), N4))
        for k, (i, j) in enumerate(edges):
            A_ub[k, i] = 1
            A_ub[k, j] = 1
        b_ub = np.ones(len(edges))
        
        bounds = [(0, 1)] * N4
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                return result.x
            else:
                return None
        except Exception:
            return None

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    # First, get LP relaxation solution
    lp_sol = None
    if time.time() < deadline - 30:
        lp_sol = lp_relaxation()
    
    # Multiple restarts with different initializations
    n_restarts = 0
    max_restarts = 30

    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        
        if n_restarts == 1 and lp_sol is not None:
            # Round LP solution: sort by LP value, greedily add
            in_set = np.zeros(N4, dtype=bool)
            order = np.argsort(-lp_sol)
            for i in order:
                if lp_sol[i] < 0.01:
                    break
                if not np.any(conflicts[i] & in_set):
                    in_set[i] = True
        else:
            in_set = greedy_init_fast()
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # Local search
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
        
        ls_time = min(remaining * 0.5, 12.0)
        if ls_time < 1.5:
            ls_time = 1.5
        
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
        
        # Second pass: polish
        remaining = deadline - 3 - time.time()
        if remaining > 4 and ls_size > current_size:
            polish_time = min(remaining * 0.5, 8.0)
            if polish_time > 2:
                p_result, p_size = local_search(ls_result, max_time=polish_time)
                if p_size > best_size:
                    best_size = p_size
                    best_in_set = p_result.copy()

    # Final: combine best v4 independent set with all v1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    if len(result_v4) > 0:
        final_vectors = np.vstack([result_v4, vectors_v1])
    else:
        final_vectors = vectors_v1

    return final_vectors

# EVOLVE-BLOCK-END
