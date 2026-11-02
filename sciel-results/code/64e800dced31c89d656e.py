import time
import numpy as np
import random

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for return
    
    n, m, p = DIMENSIONS
    full_rank = n * m * p  # 48
    
    # Initialize best result as schoolbook
    best_rank = full_rank
    U_best = [[0] * full_rank for _ in range(n * m)]
    V_best = [[0] * full_rank for _ in range(m * p)]
    W_best = [[0] * full_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][col] = 1
                V_best[j * p + k][col] = 1
                W_best[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_best, "V": V_best, "W": W_best}
    
    if time.time() - start_time > deadline - 2:
        return best_result

    # Search for rank 47
    # Strategy: Randomized search with fixing U and V to sparse random matrices
    # and solving for W. If W is "nice" (integer/half-integer), we found a solution.
    
    def try_find_decomposition(rank):
        if time.time() - start_time > deadline - 2:
            return None
            
        target_tensor = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target_tensor[u_idx, v_idx, w_idx] = 1.0
        
        mat_target = target_tensor.reshape(n * m, m * p * p * n)
        
        # We want U (nm x r), V (mp x r), W (pn x r)
        # sum_a U[:,a] (outer) V[:,a] (outer) W[:,a] = Target
        # This is equivalent to:
        # For each a, T_a = U[:,a] * V[:,a]^T (size nm x mp)
        # Then sum_a T_a * W[:,a]^T? No.
        # T = sum_a (U[:,a] * V[:,a]^T) * W[:,a]^T ? No, W is a vector.
        # The tensor is a sum of rank-1 matrices in the (nm x mp) and (pn) modes?
        # Actually, T_{u,v,w} = sum_a U_{u,a} V_{v,a} W_{w,a}.
        # Let M_a = U[:,a] * V[:,a]^T. Then T = sum_a M_a * W[:,a]^T?
        # Element (u,v,w) is sum_a M_a[u,v] W_a[w].
        # This is a linear combination of matrices M_a weighted by vectors W_a.
        
        # We can fix U and V randomly, and solve for W?
        # No, it's non-linear in U, V, W.
        # However, if we fix U and V, the equation is linear in W?
        # T_{u,v,w} = sum_a U_{u,a} V_{v,a} W_{w,a}
        # Let A_a[u,v] = U_{u,a} V_{v,a}.
        # Then T_{u,v,w} = sum_a A_a[u,v] W_{w,a}.
        # This is linear in W.
        # So: Fix U, V. Solve for W.
        
        trials = 0
        max_trials = 500
        
        while trials < max_trials:
            if time.time() - start_time > deadline - 2:
                return None
            trials += 1
            
            # Generate sparse random U and V
            # Use density 0.5
            U_rand = np.zeros((n * m, rank))
            V_rand = np.zeros((m * p, rank))
            
            # Ensure U and V are not all zero
            for a in range(rank):
                # Pick random non-zero entries
                for idx in np.random.choice(n * m, size=max(1, n * m // 4), replace=False):
                    U_rand[idx, a] = np.random.choice([-1, 1])
                for idx in np.random.choice(m * p, size=max(1, m * p // 4), replace=False):
                    V_rand[idx, a] = np.random.choice([-1, 1])
            
            # Construct the linear system for W
            # We need to solve for W (pn x rank)
            # Equations: for all u, v, w:
            # sum_a (U_rand[u,a] * V_rand[v,a]) * W[w,a] = T_target[u,v,w]
            
            # This is a large system. pn * rank unknowns.
            # (n*m * m*p) equations.
            # 6 * 24 = 144 equations.
            # 16 * 47 = 752 unknowns.
            # Under-determined.
            # We can use least squares? No, we need exact solution.
            # We need to find W such that T = sum_a (U_a * V_a^T) * W_a^T?
            
            # Let's restructure.
            # T is a 3-tensor.
            # Flatten u, v. T is (nm * mp) x (pn).
            # T_flat (nm*mp x pn) = sum_a (U_a * V_a^T).flatten() * W_a^T ?
            # No.
            # T_{(u,v), w} = sum_a (U_a[u] V_a[v]) W_a[w].
            # Let B_a be the vector (U_a * V_a^T).flatten().
            # Then T_flat = sum_a B_a * W_a^T?
            # T_flat[ (u,v), w ] = sum_a B_a[ (u,v) ] W_a[ w ].
            # Yes.
            # So T_flat (size (nm*mp) x pn) = sum_a B_a W_a^T.
            # This is a sum of rank-1 matrices.
            # We fixed U, V, so B_a are fixed vectors.
            # We need to find W_a (columns of W) such that sum_a B_a W_a^T = T_flat.
            
            # This is a linear system in the entries of W.
            # Let W be (pn x rank).
            # The equation is: for each a, the contribution is B_a * W_a^T.
            # Let W_a be the column a of W.
            # Sum_a B_a W_a^T = T_flat.
            
            # This is equivalent to:
            # [B_0 B_1 ... B_{r-1}] * W^T = T_flat?
            # No. Sum_a B_a W_a^T.
            # Let M_B be the matrix with columns B_a? No, B_a are vectors of length nm*mp.
            # So we have rank vectors B_a in R^{nm*mp}.
            # We want to express T_flat (which is in R^{nm*mp x pn}) as sum_a B_a W_a^T.
            # This is possible if the columns of T_flat (for each w) lie in the span of {B_a}.
            
            # Let's check if the columns of T_flat are in the span of {B_a}.
            # If yes, we can solve for W_a.
            
            B_matrix = np.zeros((n * m * m * p, rank))
            for a in range(rank):
                # B_a is flatten(U_a * V_a^T)
                M_a = np.outer(U_rand[:, a], V_rand[:, a]) # nm x mp
                B_matrix[:, a] = M_a.flatten()
                
            # T_flat is (nm*mp) x pn
            T_flat = target_tensor.reshape(n * m * m * p, p * n)
            
            # Check if each column of T_flat is in the column space of B_matrix.
            # B_matrix is (nm*mp) x rank.
            # We want to find W (pn x rank) such that T_flat = B_matrix * W^T.
            # So W^T = (B_matrix^T B_matrix)^-1 B_matrix^T T_flat?
            # Only if T_flat is in the range of B_matrix.
            
            # Let's project T_flat onto the column space of B_matrix.
            # If the projection equals T_flat, we have a solution.
            
            # Use QR or SVD of B_matrix
            # B_matrix shape (N, R) with N=144, R=47.
            # Since R < N, we can use least squares to find W^T.
            # Then check residual.
            
            try:
                # Solve B_matrix * W_trans = T_flat
                # W_trans is (rank x pn)
                W_trans, residuals, rank_b, s = np.linalg.lstsq(B_matrix, T_flat, rcond=None)
                
                # Check residual
                residual = np.linalg.norm(B_matrix @ W_trans - T_flat)
                
                if residual < 1e-8:
                    # We have a solution!
                    # W is (pn x rank)
                    W_sol = W_trans.T
                    # Check if entries are nice (integer or half-integer)
                    # Multiply by 2 to check for integers
                    W_check = W_sol * 2
                    if np.allclose(W_check, np.round(W_check)):
                        # Valid solution
                        U_sol = U_rand
                        V_sol = V_rand
                        W_final = W_sol / 2.0 # Wait, if W_check is int, W_sol is half-int.
                        # The task allows integer/half-integer.
                        
                        # Convert to lists
                        U_list = U_sol.tolist()
                        V_list = V_sol.tolist()
                        W_list = W_final.tolist()
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
            except np.linalg.LinAlgError:
                pass
        
        return None

    # Try rank 47
    res_47 = try_find_decomposition(47)
    if res_47 is not None:
        return res_47
        
    # Try rank 46 if time permits
    if time.time() - start_time < deadline - 10:
        res_46 = try_find_decomposition(46)
        if res_46 is not None:
            return res_46

    return best_result
