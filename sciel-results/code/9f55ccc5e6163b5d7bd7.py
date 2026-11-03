"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def calc_cost(p):
        b_perm = b[:, p][:, p]
        return int(np.sum(a * b_perm))
    
    # Precompute for fast delta calculation
    # We will use a vectorized approach for the inner loop where possible,
    # but for n=150, a tight Python loop with precomputed arrays is often faster
    # than complex vectorization due to overhead.
    
    # Symmetry check
    is_symmetric_a = np.array_equal(a, a.T)
    is_symmetric_b = np.array_equal(b, b.T)
    
    # Helper to calculate delta for swapping i and j
    # Returns the change in cost
    def get_delta(i, j, p, current_cost, B_p):
        u = p[i]
        v = p[j]
        
        # The change in cost when swapping locations of facilities i and j
        # Let p' be the new permutation.
        # Cost(p) = sum_{k,l} a[k,l] b[p[k], p[l]]
        # Only terms involving i or j change.
        # Terms:
        # 1. a[i,i] b[u,u] -> a[i,i] b[v,v]
        # 2. a[j,j] b[v,v] -> a[j,j] b[u,u]
        # 3. a[i,j] b[u,v] -> a[i,j] b[v,u]
        # 4. a[j,i] b[v,u] -> a[j,i] b[u,v]
        # 5. For k not in {i,j}:
        #    a[i,k] b[u, p[k]] -> a[i,k] b[v, p[k]]
        #    a[k,i] b[p[k], u] -> a[k,i] b[p[k], v]
        #    a[j,k] b[v, p[k]] -> a[j,k] b[u, p[k]]
        #    a[k,j] b[p[k], v] -> a[k,j] b[p[k], u]
        
        # We can group the "k not in {i,j}" terms.
        # Delta = sum_{k not in {i,j}} [ a[i,k](b[v,p[k]] - b[u,p[k]]) + a[k,i](b[p[k],v] - b[p[k],u]) 
        #                                + a[j,k](b[u,p[k]] - b[v,p[k]]) + a[k,j](b[p[k],u] - b[p[k],v]) ]
        #      + direct terms (i,i), (j,j), (i,j), (j,i)
        
        # Let's vectorize the sum over k.
        # Note: B_p[:, k] is b[:, p[k]].
        # b[v, p[k]] is B_p[v, k].
        # b[u, p[k]] is B_p[u, k].
        # b[p[k], v] is B_p[k, v].
        # b[p[k], u] is B_p[k, u].
        
        # Vectorized calculation for k != i, j
        # We create a mask or just compute for all k and subtract i,j contributions if they overlap?
        # Easier: compute sum over all k, then subtract k=i and k=j terms and add direct terms.
        
        # Sum over all k:
        # Term A: a[i,k] * (B_p[v,k] - B_p[u,k])
        # Term B: a[k,i] * (B_p[k,v] - B_p[k,u])
        # Term C: a[j,k] * (B_p[u,k] - B_p[v,k]) = -a[j,k] * (B_p[v,k] - B_p[u,k])
        # Term D: a[k,j] * (B_p[k,u] - B_p[k,v]) = -a[k,j] * (B_p[k,v] - B_p[k,u])
        
        # Let D_b_row = B_p[v, :] - B_p[u, :]  (shape n)
        # Let D_b_col = B_p[:, v] - B_p[:, u]  (shape n)
        
        # Sum_A = dot(a[i, :], D_b_row)
        # Sum_B = dot(a[:, i], D_b_col)
        # Sum_C = -dot(a[j, :], D_b_row)
        # Sum_D = -dot(a[:, j], D_b_col)
        
        # Total Sum_all = Sum_A + Sum_B + Sum_C + Sum_D
        
        # However, this sum includes k=i and k=j.
        # We need to remove the contributions for k=i and k=j from the "k not in {i,j}" part,
        # because those specific swaps are handled by the "direct terms" or are part of the 4 corner terms.
        # Actually, the formula above for "k not in {i,j}" is the standard derivation.
        
        # Let's compute the "bulk" sum for all k first.
        D_b_row = B_p[v] - B_p[u]
        D_b_col = B_p[:, v] - B_p[:, u]
        
        sum_all = np.dot(a[i], D_b_row) + np.dot(a[:, i], D_b_col) - np.dot(a[j], D_b_row) - np.dot(a[:, j], D_b_col)
        
        # Now subtract the terms where k=i and k=j from this bulk sum.
        # For k=i:
        # A_i = a[i,i] * (B_p[v,i] - B_p[u,i]) = a[i,i] * (b[v, p[i]] - b[u, p[i]]) = a[i,i] * (b[v, u] - b[u, u])
        # B_i = a[i,i] * (B_p[i,v] - B_p[i,u]) = a[i,i] * (b[p[i], v] - b[p[i], u]) = a[i,i] * (b[u, v] - b[u, u])
        # C_i = -a[j,i] * (B_p[v,i] - B_p[u,i]) = -a[j,i] * (b[v, u] - b[u, u])
        # D_i = -a[i,j] * (B_p[i,v] - B_p[i,u]) = -a[i,j] * (b[u, v] - b[u, u])
        
        # For k=j:
        # A_j = a[i,j] * (B_p[v,j] - B_p[u,j]) = a[i,j] * (b[v, v] - b[u, v])
        # B_j = a[j,i] * (B_p[j,v] - B_p[j,u]) = a[j,i] * (b[v, v] - b[v, u])
        # C_j = -a[j,j] * (B_p[v,j] - B_p[u,j]) = -a[j,j] * (b[v, v] - b[u, v])
        # D_j = -a[j,j] * (B_p[j,v] - B_p[j,u]) = -a[j,j] * (b[v, v] - b[v, u])
        
        # The direct terms (the 4 corners) are:
        # New: a[i,i]b[v,v] + a[j,j]b[u,u] + a[i,j]b[v,u] + a[j,i]b[u,v]
        # Old: a[i,i]b[u,u] + a[j,j]b[v,v] + a[i,j]b[u,v] + a[j,i]b[v,u]
        # Direct_Delta = a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u])
        
        # Total Delta = (sum_all - terms_k_i - terms_k_j) + Direct_Delta
        
        # Let's compute terms_k_i and terms_k_j explicitly.
        b_vu = b[v, u]
        b_uu = b[u, u]
        b_uv = b[u, v]
        b_vv = b[v, v]
        b_vv_u_v = b[v, v] - b[u, v] # Wait, for C_j/D_j we need b[v,v]-b[u,v] and b[v,v]-b[v,u]
        
        # Let's just use the values directly to avoid confusion.
        # k=i terms:
        t_ki = (a[i, i] * (b[v, u] - b[u, u]) + 
                a[i, i] * (b[u, v] - b[u, u]) - 
                a[j, i] * (b[v, u] - b[u, u]) - 
                a[i, j] * (b[u, v] - b[u, u]))
        
        # k=j terms:
        t_kj = (a[i, j] * (b[v, v] - b[u, v]) + 
                a[j, i] * (b[v, v] - b[v, u]) - 
                a[j, j] * (b[v, v] - b[u, v]) - 
                a[j, j] * (b[v, v] - b[v, u]))
        
        # Direct terms:
        direct = (a[i, i] * (b[v, v] - b[u, u]) + 
                  a[j, j] * (b[u, u] - b[v, v]) + 
                  a[i, j] * (b[v, u] - b[u, v]) + 
                  a[j, i] * (b[u, v] - b[v, u]))
        
        delta = sum_all - t_ki - t_kj + direct
        
        return int(delta)

    # Initialization Strategies
    
    def init_rank():
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p

    def init_spectral():
        # Spectral initialization
        # Map QAP to Max-Cut style problem.
        # We want to minimize sum a[i,j] b[p[i], p[j]].
        # This is equivalent to maximizing -sum a[i,j] b[p[i], p[j]].
        # Let C be the cost matrix.
        # If we assume b is distance-like, we want high flow a[i,j] to be close (low b).
        
        # A common spectral heuristic for QAP:
        # 1. Compute the Laplacian or similar matrix of B.
        # 2. Use eigenvectors to embed facilities and locations.
        
        # Simpler spectral approach:
        # Use the fact that QAP can be approximated by linear assignment if we relax.
        # Or use the "spectral bisection" idea.
        
        # Let's try:
        # 1. Compute row sums of A and B.
        # 2. Create a matrix M where M[i,j] = a[i,j] * b_row_sums[j]? No.
        
        # Alternative: Use the top eigenvector of the "interaction" matrix.
        # Let's try a simple spectral embedding:
        # L_b = D_b - B (Laplacian of B, assuming B is graph adjacency/distance)
        # If B is distance, we want to minimize sum w_ij d_ij.
        
        # Let's use the method from "Spectral Methods for QAP":
        # 1. Compute the matrix A.
        # 2. Compute the matrix B.
        # 3. Find the eigenvector corresponding to the smallest eigenvalue of the Laplacian of B?
        #    Or largest eigenvalue of B if B is similarity?
        #    Here B is distance, so we want to minimize.
        
        # Let's try:
        # 1. Compute row sums of B to get "centrality".
        # 2. Sort facilities by "flow centrality" (row sum of A).
        # 3. Sort locations by "distance centrality" (row sum of B).
        # This is basically rank matching again.
        
        # Let's try a different spectral approach:
        # Use the first few eigenvectors of B to create a low-dimensional embedding of locations.
        # Then project facilities into the same space using A?
        
        # Actually, a very effective simple spectral init for QAP is:
        # 1. Compute the Laplacian L_b of B (if B is distance, L_b = D_b - B might not be positive definite).
        #    Let's just use the eigen decomposition of B itself.
        #    If B is distance, small distances are good.
        
        # Let's try:
        # 1. Compute eigenvectors of B.
        # 2. The location of facility i should be close to the "center of mass" of its flows.
        
        # Let's stick to a robust Rank Matching + Random Perturbation for reliability,
        # and maybe add a Spectral one if it's fast enough.
        
        # For now, let's implement a "Gravity" heuristic which is similar to rank but more nuanced.
        # Or just use Rank Matching and Random.
        
        # Let's implement a simple spectral init based on the idea of mapping
        # facilities to locations that minimize the "energy".
        
        # 1. Compute the "flow potential" for each facility: f_i = sum_j a[i,j]
        # 2. Compute the "distance potential" for each location: d_k = sum_l b[k,l]
        # 3. Sort and match. (This is Rank Matching)
        
        # Let's try a different one:
        # Use the top eigenvector of the matrix M = A * B_row_sums?
        
        # Given time constraints, let's just use Rank Matching and Random.
        # The local search is the main driver.
        
        # Let's return a random permutation for this placeholder, to be replaced by better logic if needed.
        return np.random.permutation(n).astype(np.int64)

    def init_random():
        return np.random.permutation(n).astype(np.int64)

    # Local Search: Steepest Descent with Tabu
    def local_search(p, current_cost, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0
        
        best_p = p.copy()
        best_cost = current_cost
        B_p = b[:, p]
        
        tabu_tenure = 15
        tabu_list = {}
        iteration = 0
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < max_time:
            best_delta = np.inf
            best_i = -1
            best_j = -1
            is_aspiration = False
            
            # Iterate over all pairs
            # To speed up, we can skip some pairs if we have a good best_delta
            for i in range(n):
                u = p[i]
                for j in range(i + 1, n):
                    v = p[j]
                    
                    # Tabu check
                    move_key = (i, j)
                    if move_key in tabu_list and tabu_list[move_key] > iteration:
                        # Calculate delta to check aspiration
                        delta = get_delta(i, j, p, current_cost, B_p)
                        new_cost = current_cost + delta
                        if new_cost < best_cost:
                            is_aspiration = True
                        else:
                            continue
                    
                    if not is_aspiration:
                        delta = get_delta(i, j, p, current_cost, B_p)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        is_aspiration = False
            
            if best_i == -1:
                # No improving move found (or all tabu)
                # Pick a random non-tabu move
                valid_moves = [(i, j) for i in range(n) for j in range(i+1, n) 
                               if (i, j) not in tabu_list or tabu_list[(i, j)] <= iteration]
                if not valid_moves:
                    valid_moves = [(i, j) for i in range(n) for j in range(i+1, n)]
                
                best_i, best_j = random.choice(valid_moves)
                best_delta = get_delta(best_i, best_j, p, current_cost, B_p)
            
            # Perform move
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            
            # Update B_p
            u, v = p[best_i], p[best_j]
            B_p[:, best_i] = b[:, u]
            B_p[:, best_j] = b[:, v]
            B_p[u, :] = b[u, p]
            B_p[v, :] = b[v, p]
            B_p[u, best_i] = b[u, u]
            B_p[u, best_j] = b[u, v]
            B_p[v, best_i] = b[v, u]
            B_p[v, best_j] = b[v, v]
            
            # Update Tabu
            tabu_list[(best_i, best_j)] = iteration + tabu_tenure
            
            # Update Best
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            iteration += 1
            
            # Stagnation handling
            if no_improve > max_no_improve:
                # Perturb
                for _ in range(5):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                # Recalculate cost and B_p
                current_cost = calc_cost(p)
                B_p = b[:, p]
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve = 0
                tabu_list.clear()
        
        return best_p, best_cost

    # Multi-start Local Search
    best_overall_p = None
    best_overall_cost = np.inf
    
    # Try different initializations
    inits = [
        init_rank(),
        init_random(),
        init_random()
    ]
    
    for init_p in inits:
        if time.time() > deadline - 10.0:
            break
        
        init_cost = calc_cost(init_p)
        if init_cost < best_overall_cost:
            best_overall_cost = init_cost
            best_overall_p = init_p.copy()
        
        # Run local search
        ls_p, ls_cost = local_search(init_p, init_cost)
        
        if ls_cost < best_overall_cost:
            best_overall_cost = ls_cost
            best_overall_p = ls_p.copy()
    
    return best_overall_p.tolist()

# EVOLVE-BLOCK-END
