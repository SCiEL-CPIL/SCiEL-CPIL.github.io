"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 55.0  # 60s wall-clock, leave 5s margin
    
    N = 256
    max_sum = 2 * N
    perfect_squares = set()
    for k in range(1, int(math.isqrt(max_sum)) + 2):
        sq = k * k
        if sq <= max_sum:
            perfect_squares.add(sq)
    
    # Self-loop check: 2a is a perfect square
    self_loop_set = set()
    for m in range(1, 12):
        a = 2 * m * m
        if 1 <= a <= N:
            self_loop_set.add(a)
    
    remaining = [a for a in range(1, N + 1) if a not in self_loop_set]
    n_rem = len(remaining)
    vert_to_idx = {v: i for i, v in enumerate(remaining)}
    
    # Build adjacency as bitmasks
    adj_mask = [0] * n_rem
    for i in range(n_rem):
        a = remaining[i]
        mask = 0
        for b in remaining:
            if a != b and (a + b) in perfect_squares:
                j = vert_to_idx[b]
                mask |= (1 << j)
        adj_mask[i] = mask
        
    # Precompute degrees
    degrees = [bin(m).count('1') for m in adj_mask]
    
    def greedy_min_degree(rs=None):
        if rs is None:
            rs = random.Random(0)
        available = (1 << n_rem) - 1
        result_indices = []
        
        while available:
            min_deg = n_rem + 1
            candidates = []
            avail_list = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                avail_list.append(i)
                temp ^= lsb
            
            for i in avail_list:
                deg = bin(adj_mask[i] & available).count('1')
                if deg < min_deg:
                    min_deg = deg
                    candidates = [i]
                elif deg == min_deg:
                    candidates.append(i)
            
            v = rs.choice(candidates)
            result_indices.append(v)
            available &= ~(1 << v)
            available &= ~adj_mask[v]
        
        return result_indices

    def local_improvement(indices, rs, max_iters=500):
        """Fast local improvement using bitmasks"""
        if not indices:
            return indices
        
        selected_mask = 0
        for i in indices:
            selected_mask |= (1 << i)
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            non_selected = ((1 << n_rem) - 1) & ~selected_mask
            non_sel_list = []
            temp = non_selected
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                non_sel_list.append(i)
                temp ^= lsb
            
            rs.shuffle(non_sel_list)
            
            for v in non_sel_list:
                conflicts = adj_mask[v] & selected_mask
                if conflicts == 0:
                    selected_mask |= (1 << v)
                    improved = True
        
        result = []
        temp = selected_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb
        return result

    def simulated_annealing(start_mask, rs, max_iters=2000, T0=1.0, cooling=0.99):
        """Simulated Annealing to escape local optima"""
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        T = T0
        
        all_verts = list(range(n_rem))
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
                
            v = rs.choice(all_verts)
            bit = 1 << v
            
            if current_mask & bit:
                new_mask = current_mask & ~bit
                new_size = current_size - 1
                delta = -1
            else:
                conflicts = adj_mask[v] & current_mask
                if conflicts != 0:
                    new_mask = (current_mask & ~conflicts) | bit
                    new_size = current_size - bin(conflicts).count('1') + 1
                    delta = new_size - current_size
                else:
                    new_mask = current_mask | bit
                    new_size = current_size + 1
                    delta = 1
            
            if delta >= 0:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                if rs.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_size = new_size
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        result = []
        temp = best_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb
        return result

    def iterated_sa(rs, max_rounds=50):
        """Run SA from multiple starts"""
        best_indices = []
        best_size = 0
        
        for r in range(max_rounds):
            if time.time() > DEADLINE:
                break
            
            if r == 0:
                start = greedy_min_degree(rs)
            else:
                if best_indices:
                    k = rs.randint(2, 10)
                    to_remove = set(rs.sample(best_indices, min(k, len(best_indices))))
                    start = [x for x in best_indices if x not in to_remove]
                else:
                    start = greedy_min_degree(rs)
            
            start_mask = 0
            for i in start:
                start_mask |= (1 << i)
                
            improved_indices = local_improvement(start, rs, max_iters=200)
            improved_mask = 0
            for i in improved_indices:
                improved_mask |= (1 << i)
                
            sa_indices = simulated_annealing(improved_mask, rs, max_iters=1000)
            
            if len(sa_indices) > best_size:
                best_size = len(sa_indices)
                best_indices = sa_indices
                
        return best_indices

    # Branch-and-bound with LP relaxation bound
    def branch_and_bound(initial_mask, rs, max_nodes=5000):
        """Branch-and-bound for maximum independent set with LP bound."""
        try:
            from scipy.optimize import linprog
        except ImportError:
            return local_improvement([i for i in range(n_rem) if initial_mask & (1 << i)], rs, max_iters=100)
        
        best_size = bin(initial_mask).count('1')
        best_mask = initial_mask
        nodes_visited = 0
        
        # Build LP constraint matrix for the subproblem
        # Variables x_i in [0,1], maximize sum x_i
        # Constraints: x_i + x_j <= 1 for each edge (i,j)
        
        # Get list of vertices in the subproblem
        sub_verts = []
        temp = initial_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sub_verts.append(i)
            temp ^= lsb
        
        n_sub = len(sub_verts)
        if n_sub == 0:
            return [i for i in range(n_rem) if best_mask & (1 << i)]
        
        # Build edge list within subproblem
        edges = []
        for idx_i in range(n_sub):
            vi = sub_verts[idx_i]
            # Find neighbors of vi in subproblem
            neigh_mask = adj_mask[vi] & initial_mask
            temp_n = neigh_mask
            while temp_n:
                lsb = temp_n & (-temp_n)
                vj = lsb.bit_length() - 1
                if vj in sub_verts:
                    idx_j = sub_verts.index(vj)
                    if idx_i < idx_j:
                        edges.append((idx_i, idx_j))
                temp_n ^= lsb
        
        # Build A_ub and b_ub for LP
        m_edges = len(edges)
        if m_edges > 0:
            A_ub = np.zeros((m_edges, n_sub))
            for e_idx, (i, j) in enumerate(edges):
                A_ub[e_idx, i] = 1.0
                A_ub[e_idx, j] = 1.0
            b_ub = np.ones(m_edges)
        else:
            A_ub = None
            b_ub = None
        
        c = -np.ones(n_sub)  # Minimize -sum(x_i)
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1)] * n_sub
        
        # Solve LP relaxation
        lp_bound = n_sub  # Fallback: trivial bound
        if A_ub is not None:
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    lp_bound = int(math.ceil(-res.fun - 1e-9))
            except Exception:
                lp_bound = n_sub
        else:
            lp_bound = n_sub
        
        # If LP bound <= best, we're done
        if lp_bound <= best_size:
            return [i for i in range(n_rem) if best_mask & (1 << i)]
        
        # Branch on the vertex with highest degree in the subproblem
        # Compute degrees within subproblem
        sub_degrees = np.zeros(n_sub, dtype=int)
        for i, j in edges:
            sub_degrees[i] += 1
            sub_degrees[j] += 1
        
        # Recursive B&B with time limit
        def bb_solve(vert_mask, current_size, best_size_ref):
            nonlocal nodes_visited
            if time.time() > DEADLINE or nodes_visited >= max_nodes:
                return current_size, best_size_ref[0]
            
            nodes_visited += 1
            
            if vert_mask == 0:
                return current_size, best_size_ref[0]
            
            # Get vertices in this subproblem
            vlist = []
            temp = vert_mask
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                vlist.append(i)
                temp ^= lsb
            
            n_v = len(vlist)
            if n_v == 0:
                return current_size, best_size_ref[0]
            
            # Compute a quick upper bound: current_size + n_v
            if current_size + n_v <= best_size_ref[0]:
                return current_size, best_size_ref[0]
            
            # Try LP bound for tighter pruning
            if n_v <= 30:  # Only do LP for small subproblems
                sub_edges = []
                vset = set(vlist)
                for idx_i in range(n_v):
                    vi = vlist[idx_i]
                    neigh_mask = adj_mask[vi] & vert_mask
                    temp_n = neigh_mask
                    while temp_n:
                        lsb = temp_n & (-temp_n)
                        vj = lsb.bit_length() - 1
                        if vj in vset:
                            idx_j = vlist.index(vj)
                            if idx_i < idx_j:
                                sub_edges.append((idx_i, idx_j))
                        temp_n ^= lsb
                
                if len(sub_edges) > 0:
                    A_ub2 = np.zeros((len(sub_edges), n_v))
                    for e_idx, (i, j) in enumerate(sub_edges):
                        A_ub2[e_idx, i] = 1.0
                        A_ub2[e_idx, j] = 1.0
                    b_ub2 = np.ones(len(sub_edges))
                    c2 = -np.ones(n_v)
                    bounds2 = [(0, 1)] * n_v
                    try:
                        res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs')
                        if res2.success:
                            lp_b = int(math.ceil(-res2.fun - 1e-9))
                            if current_size + lp_b <= best_size_ref[0]:
                                return current_size, best_size_ref[0]
                    except Exception:
                        pass
            
            # Choose branch vertex: highest degree in subproblem
            v_degrees = {}
            for i in vlist:
                deg = bin(adj_mask[i] & vert_mask).count('1')
                v_degrees[i] = deg
            
            # Branch on vertex with max degree
            max_deg_v = max(v_degrees, key=lambda x: v_degrees[x])
            
            # Branch 1: Include max_deg_v
            new_mask_incl = vert_mask & ~(adj_mask[max_deg_v] | (1 << max_deg_v))
            # Recurse: we've decided to include max_deg_v
            size_incl, bs_incl = bb_solve(new_mask_incl, current_size + 1, best_size_ref)
            best_size_ref[0] = max(best_size_ref[0], size_incl)
            
            # Branch 2: Exclude max_deg_v
            new_mask_excl = vert_mask & ~(1 << max_deg_v)
            size_excl, bs_excl = bb_solve(new_mask_excl, current_size, best_size_ref)
            best_size_ref[0] = max(best_size_ref[0], size_excl)
            
            return best_size_ref[0], best_size_ref[0]
        
        # Run B&B
        bs_ref = [best_size]
        final_size, _ = bb_solve(initial_mask, 0, bs_ref)
        
        # Convert back to indices
        result = []
        temp = best_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb
        return result

    best_indices = []
    
    # Phase 1: Multiple SA starts to get a good initial solution
    for seed in range(30):
        if time.time() > DEADLINE:
            break
        rs = random.Random(seed + 100)
        indices = iterated_sa(rs, max_rounds=15)
        if len(indices) > len(best_indices):
            best_indices = indices

    # Phase 2: Branch-and-bound from best solution
    if time.time() < DEADLINE - 5:
        rs_bb = random.Random(42)
        best_mask = 0
        for i in best_indices:
            best_mask |= (1 << i)
        bb_result = branch_and_bound(best_mask, rs_bb, max_nodes=3000)
        if len(bb_result) > len(best_indices):
            best_indices = bb_result

    # Phase 3: Final local improvement
    final_rs = random.Random(99999)
    best_indices = local_improvement(best_indices, final_rs, max_iters=1000)
    
    # Phase 4: One more SA pass if time remains
    if time.time() < DEADLINE - 2:
        rs_final = random.Random(777)
        sa_result = simulated_annealing(
            sum(1 << i for i in best_indices), rs_final, max_iters=2000
        )
        if len(sa_result) > len(best_indices):
            best_indices = sa_result
            best_indices = local_improvement(best_indices, final_rs, max_iters=500)
    
    result = sorted([remaining[i] for i in best_indices])
    
    # Validate
    valid = True
    for i in range(len(result)):
        a = result[i]
        if (2 * a) in perfect_squares:
            valid = False
            break
        for j in range(i + 1, len(result)):
            b = result[j]
            if (a + b) in perfect_squares:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return [1]
    
    return result

# EVOLVE-BLOCK-END
