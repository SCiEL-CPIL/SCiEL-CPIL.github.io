"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140

    # Precompute index lists
    indices = []
    by_a = [[] for _ in range(nm)]
    by_b = [[] for _ in range(mp)]
    by_c = [[] for _ in range(pn)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                idx = len(indices)
                indices.append((a, b, c))
                by_a[a].append(idx)
                by_b[b].append(idx)
                by_c[c].append(idx)

    def compute_error(U, V, W):
        err = 0.0
        for (a, b, c) in indices:
            val = np.dot(U[a], V[b] * W[c])
            err += (val - 1.0)**2
        return err

    def verify_half_integer(U, V, W, tol=1e-10):
        """Check if U,V,W are close to half-integers and give exact decomposition."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        # Check closeness to half-integers
        if np.max(np.abs(U - U_r)) > tol or np.max(np.abs(V - V_r)) > tol or np.max(np.abs(W - W_r)) > tol:
            return None, None, None, float('inf')
        err = compute_error(U_r, V_r, W_r)
        return U_r, V_r, W_r, err

    def solve_als(rank, seed=42, max_iters=300):
        np.random.seed(seed)
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5

        for it in range(max_iters):
            if time.time() > deadline - 30:
                break

            # Update U
            for a in range(nm):
                M = np.zeros(rank)
                for idx in by_a[a]:
                    _, b, c = indices[idx]
                    M += V[b] * W[c]
                norm = np.dot(M, M)
                if norm > 1e-12:
                    U[a] = M / norm

            # Update V
            for b in range(mp):
                M = np.zeros(rank)
                for idx in by_b[b]:
                    a, _, c = indices[idx]
                    M += U[a] * W[c]
                norm = np.dot(M, M)
                if norm > 1e-12:
                    V[b] = M / norm

            # Update W
            for c in range(pn):
                M = np.zeros(rank)
                for idx in by_c[c]:
                    a, b, _ = indices[idx]
                    M += U[a] * V[b]
                norm = np.dot(M, M)
                if norm > 1e-12:
                    W[c] = M / norm

            err = compute_error(U, V, W)
            if err < 1e-10:
                break

        return U, V, W, compute_error(U, V, W)

    def try_mps_decomposition(chi_max=8):
        """
        MPS-based approach: Build the 4-index tensor T[a,b,c,d] where d is a dummy index,
        contract along the bond to get a low-rank decomposition.
        
        Strategy: Represent the tensor as T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        by finding an MPS representation with bond dimension chi and extracting
        the rank-r decomposition from the MPS structure.
        """
        # Build the full 4D tensor: T[a, b, c, r] = delta if (a,b,c) corresponds to (i,j,k)
        # Actually, let's think differently.
        # 
        # The multiplication tensor has 72 non-zero entries.
        # We want to find rank-r decomposition: sum_r U[a,r]*V[b,r]*W[c,r] = 1 for (a,b,c) in indices
        #
        # MPS approach: Think of the tensor as having "sites" corresponding to the three
        # index groups. We can try to find a low-rank structure.
        #
        # Alternative: Use the fact that for 3x3*3x8, we can use block structure.
        # A is 3x3, B is 3x8, C is 3x8.
        # Split B into two 3x4 blocks: B1, B2
        # Then C = A*B1 + A*B2
        # Each 3x3*3x4 can potentially use fewer operations.
        #
        # Let's try a direct MPS-inspired approach:
        # Reshape the problem to find structure.
        
        best_rank = None
        best_result = None
        
        for chi in range(2, chi_max + 1):
            if time.time() > deadline - 40:
                break
            
            # Try to find a rank-r decomposition where r is related to chi
            # The MPS bond structure suggests r <= chi * (some factor)
            # For now, try ranks that are multiples of chi or related
            candidate_ranks = [chi * 2, chi * 3, chi * 4, chi * 5, chi * 6]
            candidate_ranks = sorted(set(r for r in candidate_ranks if r <= 72))
            
            for rank in candidate_ranks:
                if time.time() > deadline - 35:
                    break
                
                for seed in range(3):
                    if time.time() > deadline - 30:
                        break
                    
                    U, V, W, err = solve_als(rank, seed=seed)
                    if err < 1e-6:
                        U_r, V_r, W_r, err_r = verify_half_integer(U, V, W)
                        if err_r < 1e-6:
                            if best_rank is None or rank < best_rank:
                                best_rank = rank
                                best_result = (U_r, V_r, W_r)
        
        return best_result, best_rank

    # First try MPS-inspired search
    mps_result, mps_rank = try_mps_decomposition(chi_max=6)
    
    if mps_result is not None:
        U_r, V_r, W_r = mps_result
        return {
            "U": U_r.tolist(),
            "V": V_r.tolist(),
            "W": W_r.tolist()
        }

    # Fallback to broader search
    for rank in [40, 44, 48, 52, 56, 60, 64, 68, 72]:
        if time.time() > deadline - 40:
            break

        for seed in range(5):
            if time.time() > deadline - 25:
                break

            U, V, W, err = solve_als(rank, seed=seed)
            if err < 1e-6:
                U_r, V_r, W_r, err_r = verify_half_integer(U, V, W)
                if err_r < 1e-6:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
