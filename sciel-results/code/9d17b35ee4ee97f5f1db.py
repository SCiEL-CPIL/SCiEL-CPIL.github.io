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
    deadline = start_time + 115  # Safety margin before 150s kill
    
    def is_valid_addition(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # 1. Common neighbors of u and v must be < 3
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False
        
        # 2. For any w in N(v) \ {u}, v becomes a common neighbor of u and w.
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
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    missing.append((i, j))
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
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    current_edges.append([i, j])
        return current_edges

    def verify_k23_free(adj):
        """Verify the graph is K2,3-free."""
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if popcount(common) >= 3:
                    return False
        return True

    best_edges = []
    best_count = 0
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Heuristic Search with multiple strategies
    for restart in range(800):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        
        # Strategy 1: Random greedy
        if restart % 3 == 0:
            edges = all_pairs[:]
            random.shuffle(edges)
            for u, v in edges:
                if time.time() > deadline:
                    break
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Strategy 2: Greedy with minimum common neighbors first
        elif restart % 3 == 1:
            # Sort edges by potential: prefer edges where u and v have few common neighbors
            edge_potential = []
            for u, v in all_pairs:
                common = popcount(adj[u] & adj[v])
                edge_potential.append((common, u, v))
            random.shuffle(edge_potential)
            edge_potential.sort(key=lambda x: x[0])
            for _, u, v in edge_potential:
                if time.time() > deadline:
                    break
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Strategy 3: Start from a random subset, then fill
        else:
            # Randomly include ~50% of edges first
            for u, v in all_pairs:
                if random.random() < 0.5 and is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Hill climbing: add all possible edges
        try_add_all(adj)
        
        # Swap-based local search: try removing one edge and adding others
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            current_count = edge_count(adj)
            
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
        if count > best_count and verify_k23_free(adj):
            best_count = count
            best_edges = extract_edges(adj)
    
    # Phase 2: MILP with warm start from best heuristic solution
    if time.time() < deadline - 15 and best_count < 130:
        try:
            # Map edges to indices
            edge_idx = {}
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    edge_idx[(i, j)] = idx
                    idx += 1
            num_edges = idx
            
            # z_{uvw} for each pair (u,v) with u<v and w != u,v
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
            
            # Build constraints
            num_pairs = n * (n - 1) // 2
            total_constraints = 3 * z_count + num_pairs
            
            A = lil_matrix((total_constraints, total_vars), dtype=int)
            b_ub = np.zeros(total_constraints)
            b_lb = -np.inf * total_constraints
            
            # Constraints for each z_{uvw}
            z_iter = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(n):
                        if k != i and k != j:
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
                    for w_offset in range(n - 2):
                        zi = num_edges + start_idx + w_offset
                        row = 3 * z_count + pair_iter
                        A[row, zi] = 1
                    b_ub[3 * z_count + pair_iter] = 2
                    pair_iter += 1
            
            A = csr_matrix(A)
            
            # Bounds: 0 <= x <= 1, 0 <= z <= 1
            bounds = Bounds(lb=0, ub=1)
            
            # Warm start: set initial values from best heuristic
            x0 = np.zeros(total_vars)
            for (u, v), idx in edge_idx.items():
                if best_edges and any(sorted(e) == sorted([u, v]) for e in best_edges):
                    x0[idx] = 1.0
            
            cons = LinearConstraint(A, b_lb, b_ub)
            
            res = milp(c, constraints=cons, integrality=np.ones(total_vars), 
                      bounds=bounds, x0=x0,
                      options={"time_limit": 55, "mip_rel_gap": 0.01})
            
            if res.status in [0, 1]:  # Optimal or gap limit
                x_sol = res.x[:num_edges]
                new_edges = []
                new_count = 0
                for (u, v), idx in edge_idx.items():
                    if x_sol[idx] > 0.5:
                        new_edges.append([u, v])
                        new_count += 1
                
                # Verify
                if new_count > best_count:
                    adj_check = [0] * n
                    for u, v in new_edges:
                        adj_check[u] |= (1 << v)
                        adj_check[v] |= (1 << u)
                    if verify_k23_free(adj_check):
                        best_count = new_count
                        best_edges = new_edges
                        
        except Exception as e:
            pass
    
    # Final verification and return
    if best_edges:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        if not verify_k23_free(adj_final):
            # Fallback: return empty or re-verify
            best_edges = []
    
    return best_edges
