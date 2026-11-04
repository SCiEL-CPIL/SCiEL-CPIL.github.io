import numpy as np
import time
from itertools import product
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def to_lists(mat):
        return mat.tolist()

    # Build target tensor T in R^{nm x mp x pn}
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def decompose_to_tensor(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)

    def verify(U, V, W, atol=1e-9):
        T = decompose_to_tensor(U, V, W)
        return np.allclose(T, T_target, atol=atol)

    # Schoolbook rank-18 fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r)); V = np.zeros((mp, r)); W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    def get_best_result():
        # Return the best valid decomposition found so far
        # We store candidates in a list and pick the one with lowest rank
        if not candidates:
            return schoolbook()
        # Sort by rank
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    candidates = []

    # --- Strategy: Structured Linear Search ---
    # We hypothesize that a low-rank decomposition exists with integer/half-integer entries.
    # The tensor is small (6x9x6).
    # We attempt to find rank r by fixing the columns of W to be from a small dictionary of
    # "simple" vectors (sparse, small magnitude) and solving for U and V.
    
    # Since the problem is symmetric in U, V, W up to permutation, fixing W's structure
    # is a valid reduction.
    
    def try_rank_with_fixed_W_structure(r, max_candidates=100):
        """
        Attempts to find a decomposition of rank r by iterating over possible simple
        configurations for W, then solving the resulting linear system for U and V.
        """
        if time.time() > deadline - 10:
            return None
        
        # Dictionary of candidate vectors for W columns.
        # Given the small dimensions, we can afford a moderate search space.
        # Vectors in W are in R^6.
        # We look for sparse vectors with entries in {-1, -0.5, 0, 0.5, 1}.
        
        # To keep the search tractable, we restrict to vectors with at most 2 non-zero entries
        # initially, then expand if needed.
        
        # Actually, a more robust approach for "rank minimal" is to use the known
        # structure of Strassen-like algorithms or just brute-force the linear system
        # if we assume U and V are also sparse/simple.
        
        # Let's try a different angle: 
        # If we assume the decomposition uses only integer entries (a common property
        # for minimal rank over Q if it exists), we can search for integer solutions.
        
        # However, the most reliable "jump" strategy is to check if rank 16 is possible
        # via a randomized linear algebra approach that is more robust than GD.
        
        # Method: Randomized Subspace Search
        # 1. Generate random W (6 x r).
        # 2. This defines a subspace for each slice of the tensor.
        # 3. Actually, T = sum w_t (u_t \otimes v_t)^T.
        # Flatten T to (nm*mp) x pn? No.
        # T_{a,b,c} = sum_t U_{a,t} V_{b,t} W_{c,t}.
        
        # Let's fix W to be random Gaussian, then solve for U and V?
        # This is bilinear.
        
        # Better: Alternate Minimization with Exact Linear Solves.
        # 1. Init U, V, W randomly.
        # 2. Fix V, W. Solve for U:
        #    T_{a,b,c} = sum_t U_{a,t} (V_{b,t} W_{c,t})
        #    For each a, T_{a,:,:} = sum_t U_{a,t} (V_{:,t} \otimes W_{:,t})^T
        #    This is a linear system: T_flat_a = U_a @ M_VW
        #    Where M_VW is (mp*pn) x r.
        #    If r <= mp*pn, we can solve U_a = T_flat_a @ M_VW^+ (pseudo-inverse).
        # 3. Similarly for V and W.
        # 4. Check if the resulting U, V, W produce T.
        # 5. If not, refine or restart.
        
        # This is "Alternating Least Squares" (ALS). It is much more stable than GD
        # for small systems and converges to local minima faster.
        
        def als_decompose(r, max_iter=200, tol=1e-8):
            if time.time() > deadline - 5:
                return None
                
            # Initialize
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            prev_err = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                # Solve for U
                # T_{a,b,c} = sum_t U_{a,t} V_{b,t} W_{c,t}
                # Reshape T to (nm, mp*pn)
                T_flat = T_target.reshape(nm, mp*pn)
                
                # Build M_VW: (mp*pn) x r
                # M_VW[b*p*pn + c*pn + c, t] = V[b,t] * W[c,t]
                M_VW = np.zeros((mp*pn, r))
                for t in range(r):
                    M_VW[:, t] = (V[:, t] * W[:, t]).flatten() # Wait, V is mp, W is pn.
                    # V[:, t] is length mp. W[:, t] is length pn.
                    # Outer product V[:, t] (mp x 1) * W[:, t] (1 x pn) -> mp x pn matrix.
                    # Flatten to mp*pn vector.
                    M_VW[:, t] = np.outer(V[:, t], W[:, t]).flatten()
                
                # Solve U @ M_VW = T_flat
                # U = T_flat @ pinv(M_VW)
                try:
                    U_new = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)[0].T
                except:
                    return None
                
                # Solve for V
                T_flat_v = T_target.reshape(mp, nm*pn)
                M_UW = np.zeros((nm*pn, r))
                for t in range(r):
                    M_UW[:, t] = np.outer(U_new[:, t], W[:, t]).flatten()
                try:
                    V_new = np.linalg.lstsq(M_UW.T, T_flat_v.T, rcond=None)[0].T
                except:
                    return None
                
                # Solve for W
                T_flat_w = T_target.reshape(pn, nm*mp)
                M_UV = np.zeros((nm*mp, r))
                for t in range(r):
                    M_UV[:, t] = np.outer(U_new[:, t], V_new[:, t]).flatten()
                try:
                    W_new = np.linalg.lstsq(M_UV.T, T_flat_w.T, rcond=None)[0].T
                except:
                    return None
                
                U, V, W = U_new, V_new, W_new
                
                # Calculate Error
                T_approx = decompose_to_tensor(U, V, W)
                err = np.linalg.norm(T_approx - T_target)
                
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            # Final Check
            if verify(U, V, W, atol=1e-6):
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
                
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if verify(U_i, V_i, W_i):
                    return (U_i, V_i, W_i)
                
                # If rounding fails, return the float solution if it's "clean" enough?
                # The task requires integer/half-integer.
                # So if rounding fails, this candidate is invalid for the strict contract,
                # BUT if the floats are very close to half-integers, we might need to force it.
                # For now, only accept if rounded version verifies.
                
            return None

        # Try ALS for rank 16
        # We need multiple restarts because ALS gets stuck in local minima.
        num_restarts = 50
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                break
            np.random.seed(seed * 12345 + 6789)
            res = als_decompose(r, max_iter=100)
            if res:
                return res
        return None

    # --- Execution Plan ---
    
    # 1. Try Rank 16 (Theoretical minimum is likely 16 or 17)
    # Literature suggests rank(2x3x3) is 17? Or 16?
    # Let's check 16 first.
    result_16 = try_rank_with_fixed_W_structure(16)
    if result_16:
        candidates.append((16, result_16))
        # If 16 works, we can stop? No, we want minimal. 15 is theoretically possible?
        # Lower bound for 2x3x3:
        # Flattening ranks:
        # (2*3) x (3*3) x (3*2) -> 6x9x6
        # Rank of flattening (6*9) x 6 is <= 6.
        # Rank of flattening (6*6) x 9 is <= 6.
        # Rank of flattening (9*6) x 6 is <= 6.
        # These don't force rank > 16.
        # However, Strassen's bound or similar might.
        # If 16 works, it's a great improvement over 17/18.
        
    # 2. Try Rank 17
    if not candidates or candidates[0][0] > 17:
        result_17 = try_rank_with_fixed_W_structure(17)
        if result_17:
            candidates.append((17, result_17))
            
    # 3. Fallback Rank 18 (Schoolbook)
    if not candidates:
        candidates.append((18, schoolbook()))
        
    # 4. Final Polish: If we have a candidate, try to see if we can reduce rank further?
    # No, we searched 16 and 17.
    
    # Select best
    best_rank, best_U, best_V, best_W = get_best_result()
    
    return {"U": to_lists(best_U), "V": to_lists(best_V), "W": to_lists(best_W)}
# EVOLVE-BLOCK-END
