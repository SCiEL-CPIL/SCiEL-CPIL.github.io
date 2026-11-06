import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    def check_time():
        return time.time() < deadline

    def compute_cost(A):
        """
        Calculate the total count of K5s and independent 5-sets (each counted 10 times).
        Returns the raw count (before dividing by 10).
        """
        N = A.shape[0]
        A2 = A @ A
        deg = A.sum(axis=1)
        cost = 0
        
        # Count K5s: for each edge (u,v), C(common_neighbors, 3)
        edges = np.where(A > 0)
        for u, v in zip(edges[0], edges[1]):
            if u < v:
                common = A2[u, v]
                if common >= 3:
                    cost += common * (common - 1) * (common - 2) // 6
        
        # Count Independent 5-sets: for each non-edge (u,v), C(common_non_neighbors, 3)
        for u in range(N):
            for v in range(u + 1, N):
                if A[u, v] == 0:
                    d_comp = N - deg[u] - deg[v] + A2[u, v]
                    if d_comp >= 3:
                        cost += d_comp * (d_comp - 1) * (d_comp - 2) // 6
        
        return cost

    def compute_delta_cost(A, u, v, A2, deg):
        """
        Compute the change in cost (raw, before /10) when flipping edge (u,v).
        A is the current adjacency matrix BEFORE the flip.
        A2 = A @ A, deg = A.sum(axis=1)
        Returns delta_cost = new_raw_cost - old_raw_cost
        """
        N = A.shape[0]
        is_edge = A[u, v] == 1
        delta = 0
        
        if is_edge:
            # We are removing edge (u,v)
            # K5 contribution: this edge contributed C(A2[u,v], 3) to K5 count
            old_common = A2[u, v]
            if old_common >= 3:
                delta -= old_common * (old_common - 1) * (old_common - 2) // 6
            
            # Independent 5-set contribution: this non-edge (now becoming edge) 
            # previously contributed C(d_comp, 3) where d_comp = N - deg[u] - deg[v] + A2[u,v]
            old_d_comp = N - deg[u] - deg[v] + A2[u, v]
            if old_d_comp >= 3:
                delta -= old_d_comp * (old_d_comp - 1) * (old_d_comp - 2) // 6
            
            # After removing edge (u,v):
            # New A2'[u,v] = A2[u,v] - deg[u] - deg[v] + 2*A[u,v] = A2[u,v] - deg[u] - deg[v] + 2
            # Wait, let's think more carefully.
            # A2'[u,v] = sum_k A'[u,k] A'[v,k]
            # A'[u,k] = A[u,k] for k != v, A'[u,v] = 0
            # A'[v,k] = A[v,k] for k != u, A'[v,u] = 0
            # So A2'[u,v] = sum_{k != u,v} A[u,k] A[v,k] = A2[u,v] - A[u,u]A[v,u] - A[u,v]A[v,v] = A2[u,v]
            # Actually A[u,u]=0 and A[v,v]=0, so A2'[u,v] = A2[u,v] - A[u,v]*A[v,v] - A[u,u]*A[v,u] = A2[u,v]
            # Hmm, that's not right either. Let me reconsider.
            # A2[u,v] = sum_k A[u,k] A[v,k] = A[u,v]*A[v,v] + A[u,u]*A[v,u] + sum_{k!=u,v} A[u,k]A[v,k]
            # Since A[v,v]=0 and A[u,u]=0, A2[u,v] = sum_{k!=u,v} A[u,k]A[v,k]
            # When we flip A[u,v] from 1 to 0:
            # A'[u,k] = A[u,k] for k != v, A'[u,v] = 0
            # A'[v,k] = A[v,k] for k != u, A'[v,u] = 0
            # A2'[u,v] = sum_k A'[u,k] A'[v,k] = sum_{k != u,v} A[u,k] A[v,k] + A'[u,v]*A'[v,v] + A'[u,u]*A'[v,u]
            # = sum_{k != u,v} A[u,k] A[v,k] + 0 + 0 = A2[u,v]
            # So A2'[u,v] = A2[u,v]. Interesting.
            
            # But deg changes: deg'[u] = deg[u] - 1, deg'[v] = deg[v] - 1
            
            # For K5 count, we need to check all edges (i,j) where the common neighbor count changes.
            # The common neighbor count A2'[i,j] changes only if i or j is u or v, AND the edge (u,v) was part of a triangle.
            # Specifically, for any w, A2'[w,x] changes if w=u or x=u or w=v or x=v, and the path w-u-x or w-v-x was affected.
            
            # This is getting complex. Let me just recompute the full cost for simplicity in the initial version,
            # but optimize later. For now, let's use a simpler approach:
            # Just recompute the affected terms.
            
            pass  # Placeholder - we'll use full recomputation for correctness first
        
        else:
            # We are adding edge (u,v)
            pass  # Placeholder
        
        # For correctness, let's just recompute. We'll optimize in a later iteration.
        # But that defeats the purpose. Let me implement the delta properly.
        
        # Actually, let me just recompute the full cost. With N=43, it's fast enough.
        # The bottleneck is the Python loop, not the numpy operations.
        # Let me try a different optimization: precompute and cache.
        
        return None  # Not used yet

    def solve_N(N, time_limit, init_A=None):
        if not check_time():
            return None
        
        if init_A is not None:
            A = init_A.copy()
        else:
            # Initialize with a random graph, but try to make it "balanced"
            # Start with a random graph with edge probability ~0.5
            A = np.random.randint(0, 2, (N, N))
            np.fill_diagonal(A, 0)
            A = np.triu(A, 1)
            A = A + A.T
        
        current_cost = compute_cost(A)
        if current_cost == 0:
            return A
        
        T = 1.0
        cooling = 0.995
        min_T = 0.005
        best_A = A.copy()
        best_cost = current_cost
        
        while check_time() and time.time() < start_time + time_limit:
            # Pick random edge
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            
            # Flip
            A[u, v] = 1 - A[u, v]
            A[v, u] = A[u, v]
            
            new_cost = compute_cost(A)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = A.copy()
                if current_cost == 0:
                    return A
            else:
                # Revert
                A[u, v] = 1 - A[u, v]
                A[v, u] = A[u, v]
            
            T *= cooling
            if T < min_T:
                # Restart with higher T
                T = 1.0
                # Optionally reset to best
                if random.random() < 0.3:
                    A = best_A.copy()
                    current_cost = best_cost
        
        # Final polish: try to reduce cost further with local search
        if best_cost > 0:
            A = best_A.copy()
            current_cost = best_cost
            T = 0.01
            for _ in range(1000):
                if not check_time():
                    break
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                A[u, v] = 1 - A[u, v]
                A[v, u] = A[u, v]
                new_cost = compute_cost(A)
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost == 0:
                        return A
                else:
                    A[u, v] = 1 - A[u, v]
                    A[v, u] = A[u, v]
        
        return best_A if best_cost == 0 else None

    # Try N from 43 down to 38
    # Allocate time
    time_alloc = {
        43: 120,
        42: 90,
        41: 50,
        40: 30,
        39: 15,
        38: 5
    }
    
    for N in [43, 42, 41, 40, 39, 38]:
        if not check_time():
            break
        
        limit = time_alloc.get(N, 5)
        remaining = deadline - time.time()
        if limit > remaining:
            limit = max(remaining - 5, 1)  # Keep 5s margin
        
        A = solve_N(N, limit)
        if A is not None:
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
