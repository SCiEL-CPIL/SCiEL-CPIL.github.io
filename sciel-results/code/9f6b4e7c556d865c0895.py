import time
import numpy as np

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 118.0

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_system_matrices(rank, val_set):
        # We want to find U, V, W such that sum U[i,j,c] V[j,k,c] W[k,i,c] = 1
        # This is a 3-tensor decomposition.
        # We can view this as finding rank `rank` decomposition of the matrix multiplication tensor.
        # The tensor has indices (i, j, k) -> 1.
        
        # A common approach for small ranks is to use the "border rank" or exact rank search.
        # For 2x3x5, rank 29 is schoolbook - 1. Rank 28 is even better.
        # Known lower bounds: max(2*3, 3*5, 5*2) = 15.
        # Alon-Boaz-Parter-Shok (ABPS) bound: for n*m*p, rank >= (nm+np+mp - (n+m+p-1)^2/4) / (n+m+p-1) * something? 
        # Actually, for 2x3x5, the rank is known to be 28 or 29? 
        # Let's check literature in my head. 2x2x2 is 7. 2x2x3 is 11? 
        # 2x3x3 is 17? 2x3x4 is 24? 2x3x5 is 28? 
        # If rank 28 is possible, we should find it.
        
        # Strategy: Randomized search over U and V, solve for W.
        # Since we need exact integer/half-integer solutions, we can restrict U and V to small integers.
        # Then W will be rational. We check if W has small denominators.
        
        # To make this faster, we can try to fix some structure.
        # But random search with rational reconstruction is robust.
        
        A_list = []
        b_list = []
        
        # The system is linear in W if U and V are fixed.
        # Equations: for each (i,j,k), sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        
        # We iterate over random U, V
        # U: 6 x R, V: 15 x R
        # W: 10 x R
        
        # We can sample U, V from {-1, 0, 1} with some density.
        # Then solve linear system for W.
        
        # The linear system has 30 equations and 10*R unknowns.
        # If R < 3, it's underdetermined. But R is likely > 3.
        # For R=28, 280 unknowns, 30 equations. Many solutions.
        # We just need one that has small denominators.
        
        # To increase the chance of small denominators, we can scale the problem.
        # Or we can look for U, V such that the inverse (or pseudo-inverse) gives nice numbers.
        
        # Alternative: Use the fact that we can permute and scale the tensor.
        # The tensor is symmetric in a way? No, but we can permute indices.
        
        # Let's try a few random U, V and see if W is "nice".
        
        pass

    def try_rank(rank):
        # We will try to find a solution of given rank
        # Using random sparse U, V and solving for W
        
        # Number of attempts based on remaining time
        # Let's do a fixed number of attempts, say 1000, if time permits
        
        for _ in range(1000):
            if time.time() > deadline:
                return None
            
            # Generate random U and V
            # Values in {-1, 0, 1}
            density = 0.3
            U = np.zeros((nm, rank), dtype=int)
            V = np.zeros((mp, rank), dtype=int)
            
            mask_u = np.random.rand(nm, rank) < density
            mask_v = np.random.rand(mp, rank) < density
            
            U[mask_u] = np.random.choice([-1, 1], size=(nm, rank))[mask_u]
            V[mask_v] = np.random.choice([-1, 1], size=(mp, rank))[mask_v]
            
            # Build linear system for W
            # Unknowns: W[k, i, c] for k in 0..4, i in 0..1, c in 0..rank-1
            # Total unknowns: 10 * rank
            # Equations: 30
            
            A = np.zeros((n * m * p, pn * rank))
            b = np.ones(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        coeffs = u_row * v_row
                        start_idx = (k * n + i) * rank
                        A[idx, start_idx:start_idx + rank] = coeffs
                        idx += 1
            
            # Solve A x = b
            # Since A is underdetermined (for rank >= 3), use least squares
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            W_sol = x.reshape(pn, rank)
            
            # Check if W_sol is close to rational numbers with small denominators
            # We check denominators 1, 2, 3, 4, 5, 6
            found = False
            for denom in [1, 2, 3, 4, 5, 6, 10, 12]:
                # Check if W_sol * denom is close to integer
                W_scaled = W_sol * denom
                if np.allclose(W_scaled, np.round(W_scaled), atol=1e-6):
                    W_int = np.round(W_scaled).astype(int)
                    W_list = (W_int / denom).tolist()
                    U_list = U.tolist()
                    V_list = V.tolist()
                    
                    # Verify exactly
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Maybe numerical error, try to clean up
                        # Re-verify with exact arithmetic if possible? 
                        # For now, if verify fails, try next denom or next random
                        pass
        
        return None

    # Try ranks from low to high? Or high to low?
    # We want the minimum rank.
    # Schoolbook is 30.
    # Let's try 28, 29. If found, return.
    # If not, return schoolbook.
    
    # Lower bound is 15, but likely higher.
    # Let's try 25, 26, 27, 28, 29.
    # But time is limited. 28 and 29 are most likely.
    
    for r in [28, 29]:
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result:
            return result
    
    return schoolbook()
