import numpy as np
import time
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline (20s safety margin for 150s limit)
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()

    # Verification function
    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def check_half_integer(U, V, W):
        """Check if all entries are half-integers (multiples of 0.5)"""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    def snap_to_half_int(A):
        return np.round(np.array(A, dtype=float) * 2) / 2

    # Build rank-28 decomposition
    def build_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            # M1
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M2
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = -1
            col += 1
            # M3
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M4
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = -1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M5
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            col += 1
            # M6
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (0 + j_off)][col] = -1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M7
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = -1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            col += 1

        # Block 1: j=0 (schoolbook: 2*5 = 10 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
        # Block 2: j=1,2, k=0,1 (Strassen: 7 terms)
        add_strassen(0, 1, 0)
        
        # Block 3: j=1,2, k=2 (schoolbook: 2*2 = 4 terms)
        for i in range(n):
            for j in [1, 2]:
                U[i * m + j][col] = 1
                V[j * p + 2][col] = 1
                W[2 * n + i][col] = 1
                col += 1
                
        # Block 4: j=1,2, k=3,4 (Strassen: 7 terms)
        add_strassen(0, 1, 3)
        
        assert col == rank, f"Column count mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    # Try to find rank-27 using structured sparsity + linear solve
    def try_find_rank_27_structured():
        """
        Fix U and V with specific sparsity patterns, then solve linear system for W.
        
        For each (k,i) pair, W[k*n+i, :] is a vector of length rank.
        The constraint for (i,j,k) is:
            sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        For fixed (k,i), this gives m=3 linear equations in the rank-dim vector W[k*n+i, :].
        We have p*n = 10 such vectors, each of length rank, so 10*rank unknowns.
        Total equations: n*m*p = 30.
        
        For rank=27: 270 unknowns, 30 equations -> underdetermined, should have solutions.
        We need to find U, V such that the resulting system has a solution with
        half-integer entries.
        
        Strategy: Try structured U, V patterns (sparse, with small integer/half-integer entries)
        and solve the linear system.
        """
        if time_left() < 10:
            return None
        
        rank = 27
        
        # Try several structured patterns for U and V
        np.random.seed(42)
        
        for pattern_idx in range(30):
            if time_left() < 5:
                break
            
            # Generate sparse U and V with small half-integer entries
            # U has shape (n*m, rank) = (6, 27)
            # V has shape (m*p, rank) = (15, 27)
            
            # Pattern: each row of U has ~3-5 nonzeros, each row of V has ~2-4 nonzeros
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            
            # Fill U with sparse pattern
            for row in range(n * m):
                num_nonzeros = np.random.randint(3, 6)
                cols = np.random.choice(rank, size=num_nonzeros, replace=False)
                for c in cols:
                    U[row, c] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
            
            # Fill V with sparse pattern
            for row in range(m * p):
                num_nonzeros = np.random.randint(2, 5)
                cols = np.random.choice(rank, size=num_nonzeros, replace=False)
                for c in cols:
                    V[row, c] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
            
            # Now solve for W
            # For each (k, i) pair, we have a linear system:
            #   sum_{j=0}^{m-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let A_{k,i} be a (m x rank) matrix where A_{k,i}[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Then A_{k,i} @ W[k*n+i, :] = ones(m)
            
            W = np.zeros((p * n, rank))
            feasible = True
            
            for k in range(p):
                for i in range(n):
                    # Build the m x rank matrix A
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A @ w = ones(m) for w (length rank)
                    # This is underdetermined (m=3 < rank=27), so use least squares
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        feasible = False
                        break
                    
                    # Check if the solution actually satisfies the equations
                    A_w = A @ w
                    if np.max(np.abs(A_w - 1.0)) > 1e-8:
                        feasible = False
                        break
                    
                    W[k * n + i, :] = w
            
            if not feasible:
                continue
            
            # Snap to half-integers
            W_snap = np.round(W * 2) / 2
            
            # Verify the full decomposition with snapped W
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W_snap[k * n + i, :])
                        err += (val - 1.0) ** 2
            
            if err < 1e-6:
                return {
                    "U": np.round(U * 2) / 2,
                    "V": np.round(V * 2) / 2,
                    "W": W_snap
                }
        
        return None

    # Try to find rank-27 using rational linear algebra (slice method)
    def try_find_rank_27():
        """
        Use the slice method: fix one factor (say W) to random rational values,
        then the problem becomes finding a rank-27 decomposition of a matrix
        (the "slice"), which can be attacked with linear algebra over Q.
        
        More specifically: if we fix W's columns to be e_r (standard basis),
        then U and V must satisfy: sum_r U[:,r] V[:,r]^T (with W[:,r] = e_r)
        gives the tensor. But this is over-constrained.
        
        Alternative: Use the fact that for n=2, we can view the tensor as
        a linear map from (3x5 matrices) to (2x5 matrices) and use
        border rank / flattening arguments.
        
        Practical approach: Try to find a rank-27 solution by starting from
        rank-28 and attempting to merge two terms, or by direct search with
        rational constraints.
        """
        rank = 27
        
        # Strategy: Start from rank-28 and try to eliminate one term
        # by expressing it as a linear combination of the others.
        # This is equivalent to finding a rank-27 decomposition.
        
        # Use iterative refinement with rational arithmetic
        # Initialize with random rational values (small integers / 2)
        
        best_err = float('inf')
        best_solution = None
        
        np.random.seed(12345)
        
        # Try multiple random initializations
        for attempt in range(3):
            if time_left() < 5:
                break
            
            # Initialize with random small rational values
            U = np.random.randint(-2, 3, size=(n * m, rank)).astype(float) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, rank)).astype(float) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, rank)).astype(float) * 0.5
            
            # ALS with rational snapping
            max_iters = 30
            for it in range(max_iters):
                if time_left() < 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M_ij = np.zeros((p, rank))
                        for k in range(p):
                            M_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M_jk = np.zeros((n, rank))
                        for i in range(n):
                            M_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(M_jk, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M_ki = np.zeros((m, rank))
                        for j in range(m):
                            M_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(M_ki, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-10:
                    break
            
            if best_err < 1e-10:
                break
        
        if best_solution is not None and best_err < 1e-8:
            U_sol, V_sol, W_sol = best_solution
            
            # Try to snap to half-integers
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            # Verify snapped solution
            err_snap = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_snap[i * m + j, :], V_snap[j * p + k, :] * W_snap[k * n + i, :])
                        err_snap += (val - 1.0) ** 2
            
            if err_snap < 1e-6:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        
        return None

    # First, try the structured sparsity approach (faster and more targeted)
    if time_left() > 20:
        res_27s = try_find_rank_27_structured()
        if res_27s:
            U = np.array(res_27s["U"], dtype=float)
            V = np.array(res_27s["V"], dtype=float)
            W = np.array(res_27s["W"], dtype=float)
            if verify(res_27s["U"], res_27s["V"], res_27s["W"], 27):
                return res_27s
    
    # Try the ALS-based rank-27 search if time permits
    if time_left() > 30:
        res_27 = try_find_rank_27()
        if res_27:
            # Verify
            U = np.array(res_27["U"], dtype=float)
            V = np.array(res_27["V"], dtype=float)
            W = np.array(res_27["W"], dtype=float)
            if verify(res_27["U"], res_27["V"], res_27["W"], 27):
                return res_27
    
    # Fallback to rank 28
    res_28 = build_rank_28()
    
    # Verify rank-28 solution
    U = np.array(res_28["U"], dtype=float)
    V = np.array(res_28["V"], dtype=float)
    W = np.array(res_28["W"], dtype=float)
    if verify(res_28["U"], res_28["V"], res_28["W"], 28):
        return res_28
    
    # Ultimate fallback: schoolbook rank 30
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
