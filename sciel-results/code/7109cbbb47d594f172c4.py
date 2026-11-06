import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def popcount(x):
    return bin(x).count('1')

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_valid_addition(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # 1. Common neighbors of u and v must be < 3
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False
        
        # 2. For any w in N(v) \ {u}, v becomes a common neighbor of u and w.
        #    So |N(u) ∩ N(w)| must currently be < 2.
        neighbors_v = adj[v] & ~(1 << u)
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = (lsb.bit_length() - 1)
            common_uw = adj[u] & adj[w]
            if popcount(common_uw) >= 2:
                return False
            neighbors_v ^= lsb
            
        # 3. Similarly for w in N(u) \ {v}
        neighbors_u = adj[u] & ~(1 << v)
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = (lsb.bit_length() - 1)
            common_vw = adj[v] & adj[w]
            if popcount(common_vw) >= 2:
                return False
            neighbors_u ^= lsb
            
        return True

    def get_missing_edges(adj):
        """Return list of missing edges."""
        missing = []
        for i in range(n):
            mask = ~adj[i] & ((1 << n) - 1) & ~((1 << i) - 1)
            mask &= ~((1 << (i + 1)) - 1)  # Only j > i
            while mask:
                lsb = mask & -mask
                j = (lsb.bit_length() - 1)
                missing.append((i, j))
                mask ^= lsb
        return missing

    def try_add_all(adj):
        """Try to add all possible missing edges until no more can be added."""
        improved = True
        while improved:
            if time.time() > deadline:
                return False
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline:
                    return False
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return True

    def edge_count(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2

    def extract_edges(adj):
        current_edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = (lsb.bit_length() - 1)
                if i < j:
                    current_edges.append([i, j])
                mask ^= lsb
        return current_edges

    best_edges = []
    best_count = 0
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Heuristic Search
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Initialize adjacency bitmasks
        adj = [0] * n
        
        # Greedy construction with random edge order
        edges = all_pairs[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Hill climbing: add all possible edges
        try_add_all(adj)
        
        # Swap-based local search: try removing one edge and adding two others
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            current_count = edge_count(adj)
            
            # Get current edges
            current_edges = extract_edges(adj)
            random.shuffle(current_edges)
            
            for rem_u, rem_v in current_edges:
                if time.time() > deadline:
                    break
                
                # Temporarily remove edge (rem_u, rem_v)
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                
                # Try to add all possible edges
                added = []
                missing = get_missing_edges(adj)
                random.shuffle(missing)
                for u, v in missing:
                    if time.time() > deadline:
                        break
                    if is_valid_addition(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added.append((u, v))
                
                new_count = edge_count(adj)
                
                if new_count > current_count:
                    # Keep the change
                    improved = True
                else:
                    # Revert
                    for u, v in added:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
        
        # Final hill climbing
        try_add_all(adj)
        
        # Update best
        count = edge_count(adj)
        if count > best_count:
            best_count = count
            best_edges = extract_edges(adj)
            
    # Phase 2: MILP (if time permits)
    if time.time() < deadline - 20 and best_count < 130:
        try:
            # Map edges to indices
            edge_idx = {}
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    edge_idx[(i, j)] = idx
                    idx += 1
            num_edges = idx
            
            # Map auxiliary variables z_{uvw}
            # z is for each pair (u,v) and w != u,v
            # We only need z for w where u<w or v<w? No, z_{uvw} is symmetric in u,v?
            # The constraint is sum_{w} z_{uvw} <= 2.
            # z_{uvw} <= x_{uw}, z_{uvw} <= x_{vw}, z_{uvw} >= x_{uw} + x_{vw} - 1.
            # We can index z by (u, v, w) with u < v and w != u, v.
            z_idx = {}
            z_count = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(n):
                        if k != i and k != j:
                            z_idx[(i, j, k)] = num_edges + z_count
                            z_count += 1
            total_vars = num_edges + z_count
            
            # Objective: maximize sum x
            c = np.zeros(total_vars)
            for i in range(num_edges):
                c[i] = -1  # Minimize -sum x
            
            # Constraints
            # 1. z_{uvw} <= x_{uw}
            # 2. z_{uvw} <= x_{vw}
            # 3. z_{uvw} >= x_{uw} + x_{vw} - 1  => x_{uw} + x_{vw} - z_{uvw} <= 1
            # 4. sum_{w} z_{uvw} <= 2
            
            # Use sparse matrix construction
            rows = []
            cols = []
            data = []
            
            # Constraint 1: z_{uvw} - x_{uw} <= 0
            # Constraint 2: z_{uvw} - x_{vw} <= 0
            # Constraint 3: x_{uw} + x_{vw} - z_{uvw} <= 1
            # Constraint 4: sum_{w} z_{uvw} <= 2
            
            # We will build a single A matrix for all constraints.
            # Number of constraints:
            # 3 * z_count + num_pairs (for sum)
            # num_pairs = n*(n-1)/2
            
            num_pairs = n * (n - 1) // 2
            total_constraints = 3 * z_count + num_pairs
            
            A = lil_matrix((total_constraints, total_vars), dtype=int)
            b_ub = np.zeros(total_constraints)
            b_lb = -np.inf * total_constraints
            
            # Fill constraints
            # 1 & 2 & 3 for each z
            z_iter = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(n):
                        if k != i and k != j:
                            # z_idx
                            zi = num_edges + z_iter
                            
                            # x_{ik}
                            if i < k:
                                xi = edge_idx[(i, k)]
                            else:
                                xi = edge_idx[(k, i)]
                            # x_{jk}
                            if j < k:
                                xj = edge_idx[(j, k)]
                            else:
                                xj = edge_idx[(k, j)]
                            
                            # C1: z - x_ik <= 0
                            A[z_iter, zi] = 1
                            A[z_iter, xi] = -1
                            b_ub[z_iter] = 0
                            
                            # C2: z - x_jk <= 0
                            A[z_iter + z_count, zi] = 1
                            A[z_iter + z_count, xj] = -1
                            b_ub[z_iter + z_count] = 0
                            
                            # C3: x_ik + x_jk - z <= 1
                            A[z_iter + 2*z_count, xi] = 1
                            A[z_iter + 2*z_count, xj] = 1
                            A[z_iter + 2*z_count, zi] = -1
                            b_ub[z_iter + 2*z_count] = 1
                            
                            z_iter += 1
                            
            # Constraint 4: sum_{w} z_{uvw} <= 2
            # We need to map (i,j) to the range of z indices
            # This is tricky with the flat index. Let's store the ranges.
            # Actually, we can just iterate again or store offsets.
            # Let's store offsets for (i,j)
            offsets = {}
            z_iter2 = 0
            for i in range(n):
                for j in range(i+1, n):
                    offsets[(i, j)] = z_iter2
                    count_w = 0
                    for k in range(n):
                        if k != i and k != j:
                            count_w += 1
                    z_iter2 += count_w
            
            pair_iter = 0
            for i in range(n):
                for j in range(i+1, n):
                    start_idx = offsets[(i, j)]
                    # We need to know how many w's there are for this pair
                    # It's always n-2
                    # So indices start_idx to start_idx + n - 3
                    for w_offset in range(n - 2):
                        zi = num_edges + start_idx + w_offset
                        row = 3 * z_count + pair_iter
                        A[row, zi] = 1
                    b_ub[3 * z_count + pair_iter] = 2
                    pair_iter += 1
            
            A = csr_matrix(A)
            
            # Bounds: 0 <= x <= 1, 0 <= z <= 1
            bounds = Bounds(lb=0, ub=1)
            
            cons = LinearConstraint(A, b_lb, b_ub)
            
            res = milp(c, constraints=cons, integrality=np.ones(total_vars), bounds=bounds, options={"time_limit": 60})
            
            if res.status == 0:
                # Extract solution
                x_sol = res.x[:num_edges]
                new_edges = []
                new_count = 0
                for (u, v), idx in edge_idx.items():
                    if x_sol[idx] > 0.5:
                        new_edges.append([u, v])
                        new_count += 1
                
                if new_count > best_count:
                    best_count = new_count
                    best_edges = new_edges
                    
        except Exception as e:
            pass # Ignore errors, fall back to heuristic
            
    return best_edges
