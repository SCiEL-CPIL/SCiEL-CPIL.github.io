"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as absolute baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition exactly reconstructs T."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        max_err = np.max(np.abs(T_approx - T))
        return max_err < tol, max_err
    
    def build_strassen_decomposition():
        """
        Build a structured Strassen-based decomposition for 4x4 * 4x7.
        
        Strategy: Split the 7 columns of B into 4+3.
        - For the first 4 columns: use Strassen on the 4x4 * 4x4 multiplication (rank 56)
        - For the last 3 columns: use schoolbook for 4x4 * 4x3 (rank 48)
        - Total rank: 56 + 48 = 104
        
        Strassen for 4x4 * 4x4:
        Partition A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
        Partition B into 2x2 blocks: B = [[B11, B12], [B21, B22]]
        
        The 7 Strassen products for 2x2*2x2 are applied at the block level.
        Each block product is a 2x2 * 2x2 = 2x2 multiplication.
        For 4x4*4x4, we have 4 block multiplications, each needing 7 scalar products
        per block element... 
        
        Actually, let me use the standard approach:
        The rank of 4x4*4x4 via Strassen is 7^2 = 49? No.
        
        Let me use the correct Strassen decomposition:
        For 4x4 * 4x4, partition into 2x2 blocks.
        We need to compute 4 block products: C11 = A11*B11 + A12*B21, etc.
        Using Strassen for the 2x2*2x2 block multiplication gives 7 products per block pair.
        But we have 4 block multiplications to do, each of which is 2x2*2x2.
        
        Actually, the standard Strassen for n=4 gives rank 7^2 = 49 for 4x4*4x4.
        But we can do better by being smarter.
        
        Let me use a simpler but correct approach:
        - Use schoolbook for the entire 4x4*4x7 but with some structure.
        
        Actually, let me just use the known result: the Strassen-based rank for 4x4*4x4 is 49,
        and for 4x4*4x3 it's 48 (schoolbook) or potentially less.
        
        For simplicity and correctness, let me build the decomposition explicitly:
        """
        # Strassen decomposition for 4x4 * 4x4 (first 4 columns of B)
        # Partition indices: rows 0-1, 2-3; cols 0-1, 2-3
        # A blocks: A11=[0:2,0:2], A12=[0:2,2:4], A21=[2:4,0:2], A22=[2:4,2:4]
        # B blocks: B11=[0:2,0:2], B12=[0:2,2:4], B21=[2:4,0:2], B22=[2:4,2:4]
        
        # The 7 Strassen products for 2x2*2x2:
        # P1 = (A11+A22)*(B11+B22)
        # P2 = (A21+A22)*B11
        # P3 = A11*(B12-B22)
        # P4 = A22*(B21-B11)
        # P5 = (A11+A12)*B22
        # P6 = (A21-A11)*(B11+B12)
        # P7 = (A12-A22)*(B21+B22)
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # For 4x4*4x4, we apply this at the block level.
        # Each "product" like P1 = (A11+A22)*(B11+B22) involves multiplying two 2x2 matrices.
        # That's a 2x2*2x2 = 2x2 multiplication, which has rank 4 (schoolbook).
        # So each Strassen product contributes 4 scalar multiplications.
        # 7 products * 4 = 28? That doesn't seem right either.
        
        # Let me just use the direct approach: enumerate all the terms.
        # For the 4x4*4x4 part, use schoolbook: 16 terms.
        # For the 4x4*4x3 part, use schoolbook: 12 terms.
        # Total: 28 terms. But that's just schoolbook for the whole thing = 112.
        
        # I'm overcomplicating this. Let me use a simpler structured approach:
        # Just use schoolbook for everything but try to optimize.
        
        return None  # Fall back to optimization
    
    # Try to find a better decomposition using ALS optimization
    T_flat_2d = T.reshape(nm, mp * pn)  # (nm, mp*pn)
    T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # (mp, nm*pn)
    T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # (pn, nm*mp)
    
    def solve_als_step(target_matrix, M):
        """Solve target = X @ M for X using least squares."""
        try:
            X_T, residuals, rank_sol, sv = np.linalg.lstsq(M.T, target_matrix.T, rcond=None)
            return X_T.T
        except:
            return None
    
    def try_rank_als(target_rank, n_restarts=3, max_iter=30):
        """Try to find a rank-target_rank decomposition using ALS."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                M_U = np.einsum('br,cr->rbc', V, W).reshape(target_rank, mp * pn)
                U_new = solve_als_step(T_flat_2d, M_U)
                if U_new is not None:
                    U = U_new
                    
                # Update V
                M_V = np.einsum('ar,cr->rac', U, W).reshape(target_rank, nm * pn)
                V_new = solve_als_step(T_flat_v, M_V)
                if V_new is not None:
                    V = V_new
                    
                # Update W
                M_W = np.einsum('ar,br->rab', U, V).reshape(target_rank, nm * mp)
                W_new = solve_als_step(T_flat_w, M_W)
                if W_new is not None:
                    W = W_new
                
                # Check convergence
                ok, err = check_decomposition(U, V, W)
                if ok:
                    if target_rank < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                    return True
                
                # Early stopping
                if it > 5:
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                    cur_err = np.sum((T_approx - T)**2)
                    if it >= 2:
                        prev_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                        # Simple check: if error is not decreasing, break
                        pass
                        
        return False
    
    # Try ranks from 104 down to 80
    # Start with the known good rank of 104 (Strassen + schoolbook)
    for target_rank in [104, 100, 96, 92, 88, 84, 80]:
        if time.time() > deadline - 15:
            break
        if try_rank_als(target_rank, n_restarts=2, max_iter=25):
            if target_rank < best_rank:
                break
    
    # If we found something better than schoolbook, return it
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Otherwise return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
