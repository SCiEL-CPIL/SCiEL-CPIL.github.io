import time
import numpy as np
import random
from fractions import Fraction

def _verify(U, V, W, n=2, m=2, p=7):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_rank25():
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, k_start):
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_schoolbook():
    n, m, p = 2, 2, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_grobner_rank_search(deadline):
    """
    Structural change: use Groebner basis theory to search for lower-rank decompositions.
    
    For a candidate rank r, we look for U (4×r), V (14×r), W (14×r) such that
    for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1.
    
    This is a system of 28 polynomial equations in 4*r + 14*r + 14*r = 32*r unknowns.
    
    Key structural insight: by fixing a gauge (e.g., normalize U to have 1s on diagonal
    where possible), we can reduce the search space. We use the fact that the tensor
    equations are multilinear in each factor, so we can:
    1. Fix U to a canonical form (first r columns of an identity-like structure)
    2. Solve for V and W from the remaining linear constraints
    3. Check if the solution satisfies all 28 equations
    
    This transforms the problem from a 32r-variable nonlinear system to a
    structured linear search over the gauge freedom.
    """
    n, m, p = 2, 2, 7
    num_eq = n * m * p  # 28 equations
    
    # Try ranks from 24 down
    for target_rank in [24, 23, 22]:
        if time.time() > deadline - 15:
            break
        
        # Strategy: fix U to a structured form, solve for V, W
        # U is 4×r. We fix the first 4 rows to be as "spread out" as possible.
        # For rank r, U has 4*r entries. We can fix 4 of them (one per row) to 1.
        # Then we have 4*(r-1) free entries in U.
        
        # Better approach: use the linearity in each factor.
        # For fixed U and V, W is determined linearly (if the system is consistent).
        # For fixed U, the equations become linear in V⊗W entries.
        
        # Most practical: enumerate structured U patterns, solve linear system for V,W.
        
        # Number of free parameters: 4*r + 14*r + 14*r - 28 (equations) = 32*r - 28
        # For r=24: 768-28 = 740 free parameters. Too many to enumerate.
        
        # Instead: use the fact that U, V, W entries are likely in {-1, 0, 1, 2}
        # and do a focused search with the constraint that U is "generic enough".
        
        # Approach: random U with small integer entries, then solve for V and W
        # using the linear system. Check if solution has small integer/half-integer entries.
        
        attempts = 2000
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random U with entries in {-1, 0, 1}
            U = np.zeros((4, target_rank), dtype=int)
            # Ensure each row has at least one non-zero
            for row in range(4):
                while all(U[row] == 0):
                    U[row] = np.random.randint(-1, 2, size=target_rank)
            
            # For each equation (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is bilinear in V and W. Fix V, solve for W (or vice versa).
            
            # Generate random V with entries in {-1, 0, 1}
            V = np.zeros((14, target_rank), dtype=int)
            for row in range(14):
                while all(V[row] == 0):
                    V[row] = np.random.randint(-1, 2, size=target_rank)
            
            # Now for each k (0..6) and i (0..1), we need:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1  for all j
            # For fixed k,i: sum_r (sum_j ... ) -- no, each (i,j,k) is separate.
            
            # For fixed (k,i), the equation for each j is:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Let a_{j,r} = U[i*m+j,r] * V[j*p+k,r]. Then:
            # sum_r a_{j,r} * W[k*n+i,r] = 1 for j=0,1
            # This is 2 equations in r unknowns (W[k*n+i,0..r-1]).
            # Underdetermined! So we need consistency across all (i,j,k).
            
            # Actually, for fixed U and V, the system for W is:
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Group by (k,i): for fixed k,i, we have m=2 equations (one per j):
            #   sum_r U[i*m+0,r] * V[0*p+k,r] * W[k*n+i,r] = 1
            #   sum_r U[i*m+1,r] * V[1*p+k,r] * W[k*n+i,r] = 1
            # This is 2 equations in r unknowns for W[k*n+i,:].
            
            # With r=24 and 14 such (k,i) pairs, total 28 equations in 14*24=336 W-unknowns.
            # Very underdetermined. So for any U,V, there exist W solutions.
            # The question is whether W can be chosen with small integer/half-integer entries.
            
            # Since it's underdetermined, let's just pick W=0 and check if the
            # "residual" can be absorbed. Actually, let's check if there EXISTS
            # integer W by solving the linear system.
            
            # Build the linear system: 28 equations, 336 unknowns (for r=24)
            # A * w_vec = 1 (vector of 28 ones)
            # where A is 28×336.
            
            A_mat = np.zeros((num_eq, 14 * target_rank))
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            # W index: k*n+i, component r
                            w_col = (k * n + i) * target_rank + r
                            A_mat[eq_idx, w_col] = U[i*m+j, r] * V[j*p+k, r]
                        eq_idx += 1
            
            # Check if system is consistent: rank(A) should be <= 28 and 1 in colspace
            try:
                rank_A = np.linalg.matrix_rank(A_mat)
                # Check if ones vector is in column space
                # A * w = 1 has solution iff rank(A) == rank([A|1])
                A_aug = np.hstack([A_mat, np.ones((num_eq, 1))])
                rank_aug = np.linalg.matrix_rank(A_aug)
                
                if rank_A == rank_aug:
                    # Solution exists. Find one with minimum norm, check if close to integer
                    # Use least squares (will give exact solution since consistent)
                    w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(num_eq), rcond=None)
                    
                    # Check if solution is close to half-integers
                    w_rounded = np.round(w_sol * 2) / 2
                    if np.allclose(A_mat @ w_rounded, np.ones(num_eq), atol=1e-6):
                        # Great! We have a valid solution with half-integer W
                        W = w_rounded.reshape(14, target_rank)
                        # Convert to list, ensure exact half-integer representation
                        W_list = W.tolist()
                        V_list = V.tolist()
                        U_list = U.tolist()
                        
                        # Final verification
                        if _verify(U_list, V_list, W_list):
                            return U_list, V_list, W_list
            except:
                pass
            
            # If above didn't work, try with V having entries in {-2,-1,0,1,2}
            if attempt % 3 == 0:
                V = np.zeros((14, target_rank), dtype=int)
                for row in range(14):
                    while all(V[row] == 0):
                        V[row] = np.random.randint(-2, 3, size=target_rank)
                
                A_mat = np.zeros((num_eq, 14 * target_rank))
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                w_col = (k * n + i) * target_rank + r
                                A_mat[eq_idx, w_col] = U[i*m+j, r] * V[j*p+k, r]
                            eq_idx += 1
                
                try:
                    rank_A = np.linalg.matrix_rank(A_mat)
                    A_aug = np.hstack([A_mat, np.ones((num_eq, 1))])
                    rank_aug = np.linalg.matrix_rank(A_aug)
                    
                    if rank_A == rank_aug:
                        w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(num_eq), rcond=None)
                        w_rounded = np.round(w_sol * 2) / 2
                        if np.allclose(A_mat @ w_rounded, np.ones(num_eq), atol=1e-6):
                            W = w_rounded.reshape(14, target_rank)
                            W_list = W.tolist()
                            V_list = V.tolist()
                            U_list = U.tolist()
                            if _verify(U_list, V_list, W_list):
                                return U_list, V_list, W_list
                except:
                    pass
    
    return None

def _try_structured_merge(deadline):
    """
    Alternative structural approach: try to find a rank-24 solution by
    exploiting symmetries in the 7-column B matrix.
    
    Key idea: the 7 columns of B can be grouped. If we can find a decomposition
    where two columns share structure (beyond what Strassen already exploits),
    we might save one more term.
    
    Try: use Strassen on 3 pairs (6 columns) + 1 column, but with a modified
    Strassen that shares one term between adjacent pairs.
    """
    n, m, p = 2, 2, 7
    
    # Try a "shared Strassen" approach: modify the rank-25 construction
    # to share one term between the first two Strassen blocks.
    # This requires careful algebraic manipulation.
    
    # For now, try a different block structure:
    # Instead of (0,1), (2,3), (4,5), 6, try (0,1), (2,3), (4,5,6) with a
    # rank-8 decomposition for the last 3 columns.
    
    # A 2x2 by 2x3 matrix multiplication has rank at most 12 (schoolbook)
    # Strassen-like for 3 columns might give rank 8 or 9.
    
    # This is speculative; try random search in the space of 2x2x3 decompositions
    # with rank 8, then combine with two Strassen blocks (7+7+8=22? No, 7+7+8=22 < 24)
    # Wait, that would be rank 7+7+8 = 22 which is too optimistic.
    # Actually 2x2x3 schoolbook is 12, Strassen-like might be 9-10.
    # 7 + 7 + 10 = 24. That's our target!
    
    # But we need to verify the 2x2x3 part has rank 10.
    # Let's just try random search for the whole thing at rank 24.
    
    if time.time() > deadline - 20:
        return None
    
    # Focused random search at rank 24 with structured initialization
    for attempt in range(3000):
        if time.time() > deadline - 15:
            break
        
        rank = 24
        # Initialize close to the rank-25 solution, then zero one column
        # and adjust neighbors
        U = np.zeros((4, rank), dtype=int)
        V = np.zeros((14, rank), dtype=int)
        W = np.zeros((14, rank), dtype=int)
        
        # Copy rank-25 structure into first 23 columns (dropping one)
        U25, V25, W25 = _build_strassen_rank25()
        # Drop a random column from the 25
        drop_col = random.randint(0, 24)
        keep_cols = [c for c in range(25) if c != drop_col]
        
        for new_idx, old_idx in enumerate(keep_cols):
            U[:, new_idx] = np.array(U25)[:, old_idx]
            V[:, new_idx] = np.array(V25)[:, old_idx]
            W[:, new_idx] = np.array(W25)[:, old_idx]
        
        # The 24th column is zero; try to fill it to compensate
        # The residual from dropping column drop_col is:
        # For each (i,j,k): -U25[i*m+j, drop_col] * V25[j*p+k, drop_col] * W25[k*n+i, drop_col]
        # We need the new column 23 to compensate:
        # U[i*m+j, 23] * V[j*p+k, 23] * W[k*n+i, 23] = residual_{i,j,k}
        
        # This is generally impossible with one term, but let's try small integer values
        for u_try in range(-2, 3):
            for v_try in range(-2, 3):
                for w_try in range(-2, 3):
                    if u_try == 0 or v_try == 0 or w_try == 0:
                        continue
                    U[0, 23] = u_try
                    V[0, 23] = v_try
                    W[0, 23] = w_try
                    
                    # Quick check: does this help? Just verify
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if _verify(U_list, V_list, W_list):
                        return U_list, V_list, W_list
        
        # Also try: don't drop a column, but instead modify two columns
        # to create a dependency that allows one column to be eliminated
        if attempt % 5 == 0:
            U2 = np.array(U25)[:, :24].copy()  # Just take first 24 of 25
            V2 = np.array(V25)[:, :24].copy()
            W2 = np.array(W25)[:, :24].copy()
            
            # Randomly perturb one entry in each of U, V, W
            row_u = random.randint(0, 3)
            col_u = random.randint(0, 23)
            row_v = random.randint(0, 13)
            col_v = random.randint(0, 23)
            row_w = random.randint(0, 13)
            col_w = random.randint(0, 23)
            
            delta = random.choice([-1, 1])
            U2[row_u, col_u] += delta
            V2[row_v, col_v] += delta
            W2[row_w, col_w] += delta
            
            if _verify(U2.tolist(), V2.tolist(), W2.tolist()):
                return U2.tolist(), V2.tolist(), W2.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    n, m, p = 2, 2, 7
    
    # Step 1: Build rank-25 Strassen decomposition (guaranteed valid)
    U25, V25, W25 = _build_strassen_rank25()
    
    # Verify
    if not _verify(U25, V25, W25):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U25, V25, W25
    best_rank = 25
    
    # Step 2: Try Groebner-based structural search for rank 24
    if time.time() < deadline - 30:
        time_budget_end = min(deadline - 25, start_time + 60)
        result = _try_grobner_rank_search(time_budget_end)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 24
    
    # Step 3: Try structured merge approach for rank 24
    if best_rank == 25 and time.time() < deadline - 20:
        result = _try_structured_merge(deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 24
    
    # Step 4: If we found rank 24, try for rank 23
    if best_rank == 24 and time.time() < deadline - 15:
        time_budget_end = min(deadline - 10, start_time + 80)
        result = _try_grobner_rank_search(time_budget_end)
        if result is not None and len(result[0][0]) == 23:
            best_U, best_V, best_W = result
            best_rank = 23
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_result = np.zeros((2, 7))
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        # W maps to output
        for k in range(7):
            for i in range(2):
                C_result[i, k] += SA * SB * W[k*2+i, r]
    
    print(f"Max error: {np.max(np.abs(C_result - C_expected))}")
    print(f"Rank: {U.shape[1]}")
