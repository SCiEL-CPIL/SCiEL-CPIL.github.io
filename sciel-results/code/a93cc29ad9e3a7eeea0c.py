"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Strategy:
    # 1. Start with a low-rank decomposition using random integer/half-integer matrices.
    # 2. Use a "fixing" strategy: fix V and W, solve for U (least squares).
    #    Then round U to nearest integer/half-integer.
    #    Repeat cyclically.
    # 3. If we get an exact match, return it.
    # 4. If not, try increasing rank.
    # 5. Fallback to schoolbook if no low-rank solution found.

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        # Efficient check using einsum
        # T_recon[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T_recon == 1 for all valid (i,j,k)
        # Since the indices are coupled, we can't just do a simple einsum on the full tensor easily
        # without reshaping.
        
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Compute the tensor product
        # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
        
        # This is a specific contraction.
        # Let's compute it explicitly for verification.
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the standard schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Try to find a decomposition with rank < n*m*p
    # Theoretical lower bound is max(n*m, m*p, p*n) = 28, but usually higher.
    # Let's try ranks from 30 up to 83.
    
    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]

    # Heuristic: Try to find a solution with rank significantly lower than schoolbook.
    # If we fail to find one quickly, we return schoolbook.
    
    # We will use a randomized approach with "fixing" and "rounding".
    # Since the problem is small, we can afford some iterations.
    
    for rank in range(30, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random values from {-1, 0, 1} or similar
            # We want to encourage sparsity and small integers.
            U = np.random.choice([-1, 0, 1], (nm, rank))
            V = np.random.choice([-1, 0, 1], (mp, rank))
            W = np.random.choice([-1, 0, 1], (pn, rank))
            
            # Ensure some non-zero entries
            if not np.any(U): U[0,0] = 1
            if not np.any(V): V[0,0] = 1
            if not np.any(W): W[0,0] = 1
            
            # Optimization loop
            # We will fix two matrices and solve for the third via least squares, then round.
            # This is similar to ALS but with rounding.
            
            converged = False
            for it in range(20):
                if time.time() > deadline:
                    break
                    
                # 1. Fix V, W. Solve for U.
                # We want U such that sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # For a fixed (i,j), we have a linear system for U[i*m+j, :]:
                # U[i*m+j, :] @ X = 1, where X[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Wait, the sum is over c.
                # Let A_c = V[:, c] * W[:, c] (element-wise product of vectors of length mp and pn? No.)
                # Let's look at the structure again.
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # For a fixed (i,j), let u_vec = U[i*m+j, :].
                # We want sum_c u_vec[c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k.
                # This is u_vec @ (V[:, j*p+k] * W[:, k*n+i])^T ? No.
                # Let M_kc = V[j*p+k, c] * W[k*n+i, c].
                # Then sum_c u_vec[c] * M_kc = 1 for all k in 0..p-1.
                # This is a linear system: u_vec @ M^T = ones(p)
                # M is (p x rank). M^T is (rank x p).
                # So we solve (M @ M^T) u_vec^T = M @ ones(p) ? No.
                # Let's define B as a matrix where B[k, c] = V[j*p+k, c] * W[k*n+i, c].
                # We want u_vec @ B^T = 1^T.
                # This is underdetermined if rank > p. We want the minimum norm solution or just any solution.
                # Actually, we just need *some* solution.
                # We can use least squares: u_vec = ones(p) @ B / (B^T @ B) ?
                # Or simply: u_vec = pinv(B^T) @ ones(p)
                
                # Let's do this for all (i,j).
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Build B: (p x rank)
                        B = np.zeros((p, rank))
                        for k in range(p):
                            B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        # Solve B^T u = ones(p)
                        # Least squares solution
                        try:
                            u_sol = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                        except:
                            u_sol = np.zeros(rank)
                        
                        # Round to nearest integer or half-integer
                        # The task allows integer/half-integer entries.
                        # Rounding to nearest 0.5
                        u_rounded = np.round(u_sol * 2) / 2
                        new_U[i * m + j, :] = u_rounded
                
                U = new_U

                # 2. Fix U, W. Solve for V.
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For fixed (j,k), let v_vec = V[j*p+k, :].
                # We want sum_c v_vec[c] * U[i*m+j, c] * W[k*n+i, c] = 1 for all i.
                # Let C_ic = U[i*m+j, c] * W[k*n+i, c].
                # We want v_vec @ C^T = ones(n).
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        C = np.zeros((n, rank))
                        for i in range(n):
                            C[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        try:
                            v_sol = np.linalg.lstsq(C.T, np.ones(n), rcond=None)[0]
                        except:
                            v_sol = np.zeros(rank)
                        
                        v_rounded = np.round(v_sol * 2) / 2
                        new_V[j * p + k, :] = v_rounded
                V = new_V

                # 3. Fix U, V. Solve for W.
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For fixed (k,i), let w_vec = W[k*n+i, :].
                # We want sum_c w_vec[c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j.
                # Let D_jc = U[i*m+j, c] * V[j*p+k, c].
                # We want w_vec @ D^T = ones(m).
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        D = np.zeros((m, rank))
                        for j in range(m):
                            D[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        try:
                            w_sol = np.linalg.lstsq(D.T, np.ones(m), rcond=None)[0]
                        except:
                            w_sol = np.zeros(rank)
                        
                        w_rounded = np.round(w_sol * 2) / 2
                        new_W[k * n + i, :] = w_rounded
                W = new_W

                # Check for convergence/exactness
                if verify(U, V, W):
                    converged = True
                    break

            if converged:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break

        # If we found a solution for this rank, we might try lower?
        # No, we are iterating upwards. If we find one at rank 30, we stop.
        if best_rank == rank:
            break

    # Final check: if we didn't find a better solution, use schoolbook
    if best_rank == n * m * p:
        best_U, best_V, best_W = schoolbook()

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
