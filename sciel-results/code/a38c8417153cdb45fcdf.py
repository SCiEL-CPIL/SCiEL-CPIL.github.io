import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    def check_time():
        return time.time() < deadline

    def compute_cost_vectorized(A):
        """
        Calculate total K5 count + independent 5-set count using vectorized numpy.
        A is an NxN adjacency matrix (0/1, symmetric, zero diagonal).
        Returns total count of (K5 cliques + independent 5-sets).
        """
        N = A.shape[0]
        if N < 5:
            return 0
        
        A2 = A @ A
        deg = A.sum(axis=1)
        
        # Count K5s: for each edge (u,v), C(common_neighbors, 3)
        # common_neighbors = A2[u,v] for edges
        # We need sum over all edges of C(A2[u,v], 3)
        # C(x,3) = x*(x-1)*(x-2)/6 for x >= 3, else 0
        upper_tri = np.triu(A, 1)
        common = A2 * upper_tri  # only edges
        # C(common, 3)
        k5_contrib = np.where(common >= 3, common * (common - 1) * (common - 2) // 6, 0)
        k5_total = int(np.sum(k5_contrib))
        
        # Count Independent 5-sets: for each non-edge (u,v), C(common_non_neighbors, 3)
        # common_non_neighbors = N - deg[u] - deg[v] + A2[u,v]
        # (vertices that are non-neighbors of both u and v)
        non_edge_mask = 1 - A  # 1 where no edge
        non_edge_upper = np.triu(non_edge_mask, 1)
        
        # For each non-edge (u,v), compute d_comp
        deg_sum = deg[:, None] + deg[None, :]  # deg[u] + deg[v] for all pairs
        d_comp = N - deg_sum + A2  # common non-neighbors
        d_comp = d_comp * non_edge_upper  # only non-edges
        ind5_contrib = np.where(d_comp >= 3, d_comp * (d_comp - 1) * (d_comp - 2) // 6, 0)
        ind5_total = int(np.sum(ind5_contrib))
        
        return k5_total + ind5_total

    def compute_delta_fast(A, A2, deg, u, v):
        """
        Compute the change in total cost when flipping edge (u,v).
        Returns delta = new_cost - old_cost.
        Uses precomputed A2 and deg.
        """
        N = A.shape[0]
        is_edge = A[u, v] == 1
        
        if is_edge:
            # Removing edge (u,v)
            # Old K5 contribution from this edge: C(A2[u,v], 3)
            old_common = A2[u, v]
            old_k5 = old_common * (old_common - 1) * (old_common - 2) // 6 if old_common >= 3 else 0
            
            # Old independent contribution from this non-edge (now becoming edge):
            # Wait, this is an edge, so it wasn't contributing to independent sets.
            # After removing, it becomes a non-edge and may contribute to independent sets.
            old_d_comp = N - deg[u] - deg[v] + A2[u, v]
            # This is the d_comp for the non-edge (u,v) after removal
            new_ind = old_d_comp * (old_d_comp - 1) * (old_d_comp - 2) // 6 if old_d_comp >= 3 else 0
            
            # But we also need to check how other edges/non-edges are affected
            # When we remove edge (u,v), the common neighbor counts for edges involving u or v change
            # Specifically, for any w:
            #   If (u,w) is an edge: A2'[u,w] = A2[u,w] - A[v,w] (lost v as common neighbor if v-w edge)
            #   If (v,w) is an edge: A2'[v,w] = A2[v,w] - A[u,w]
            #   If (u,w) is non-edge: d_comp changes
            #   If (v,w) is non-edge: d_comp changes
            
            # This is complex. For correctness, let's just recompute the full cost.
            # But that's what we're trying to avoid. Let me think...
            
            # Actually, for the delta, we need to track all affected terms.
            # Let's just recompute. With N=43, vectorized compute_cost is fast enough.
            pass
        
        # For simplicity and correctness, just recompute
        # But we can optimize by only recomputing affected parts
        # For now, use full recompute
        return None  # Placeholder

    def solve_N(N, time_limit, init_A=None, seed_edges=None):
        """
        Solve for N vertices using simulated annealing.
        """
        if not check_time():
            return None
        
        if init_A is not None:
            A = init_A.copy()
        else:
            # Initialize with random graph
            A = np.random.randint(0, 2, (N, N))
            np.fill_diagonal(A, 0)
            A = np.triu(A, 1)
            A = A + A.T
        
        current_cost = compute_cost_vectorized(A)
        if current_cost == 0:
            return A
        
        T = 1.0
        cooling = 0.995
        min_T = 0.005
        best_A = A.copy()
        best_cost = current_cost
        no_improve = 0
        max_no_improve = 500
        
        steps = 0
        while check_time() and time.time() < start_time + time_limit:
            # Pick random edge
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            
            # Flip
            old_val = A[u, v]
            A[u, v] = 1 - old_val
            A[v, u] = A[u, v]
            
            new_cost = compute_cost_vectorized(A)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = A.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                if current_cost == 0:
                    return A
            else:
                # Revert
                A[u, v] = old_val
                A[v, u] = old_val
                no_improve += 1
            
            T *= cooling
            steps += 1
            
            if T < min_T or no_improve > max_no_improve:
                # Restart
                T = 1.0
                no_improve = 0
                if random.random() < 0.3:
                    A = best_A.copy()
                    current_cost = best_cost
        
        return best_A if best_cost == 0 else None

    def extend_paley_to_N(N):
        """
        Extend the Paley graph on 37 vertices to N vertices by adding random edges.
        """
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        
        A = np.zeros((N, N), dtype=int)
        
        # Add Paley edges for first q vertices
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # Add random edges for new vertices
        for i in range(q, N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    A[i, j] = 1
                    A[j, i] = 1
        
        return A

    # Try N from 43 down to 38
    # Allocate time more aggressively for larger N
    time_alloc = {
        43: 150,
        42: 80,
        41: 40,
        40: 20,
        39: 10,
        38: 5
    }
    
    # Try to find the largest N
    for N in [43, 42, 41, 40, 39, 38]:
        if not check_time():
            break
        
        limit = time_alloc.get(N, 5)
        remaining = deadline - time.time()
        if limit > remaining:
            limit = max(remaining - 5, 1)
        
        if limit < 2:
            break
        
        # Multiple restarts
        found = None
        num_restarts = max(1, int(limit / 10))  # ~10s per restart
        
        for r in range(num_restarts):
            if not check_time():
                break
            
            restart_limit = min(10, limit - (time.time() - start_time))
            if restart_limit < 1:
                break
            
            if r == 0:
                # First restart: extend Paley
                A_init = extend_paley_to_N(N)
            else:
                # Subsequent restarts: random
                A_init = None
            
            A = solve_N(N, restart_limit, init_A=A_init)
            if A is not None:
                found = A
                break
        
        if found is not None:
            A = found
            edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if A[i, j] == 1:
                        edges.append((i, j))
            return N, edges

    # Fallback to Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return n, edges
