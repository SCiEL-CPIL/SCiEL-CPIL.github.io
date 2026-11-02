import numpy as np
import time
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline (10s safety margin)
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()

    # Verification function
    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
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

    # Try to find rank-27 using rational linear algebra (slice method)
    def try_find_rank_27():
        rank = 27
        
        np.random.seed(12345)
        
        best_err = float('inf')
        best_solution = None
        
        # Try multiple random initializations
        for attempt in range(5):
            if time_left() < 10:
                break
            
            # Initialize with random small rational values
            U = np.random.randint(-2, 3, size=(n * m, rank)).astype(float) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, rank)).astype(float) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, rank)).astype(float) * 0.5
            
            # ALS with rational snapping
            max_iters = 50
            for it in range(max_iters):
                if time_left() < 10:
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

    # Try rank-27 using rank-28 warm start with deflation
    def try_find_rank_27_warm_start():
        """
        Start from rank-28, try to find a direction that allows rank reduction.
        For each pair of terms, check if their sum is approximately rank-1.
        """
        rank28 = build_rank_28()
        U28 = np.array(rank28["U"], dtype=float)
        V28 = np.array(rank28["V"], dtype=float)
        W28 = np.array(rank28["W"], dtype=float)
        
        # Try to merge pairs of terms
        # For terms a and b, check if U_a⊗V_a⊗W_a + U_b⊗V_b⊗W_b ≈ rank-1
        # This is equivalent to checking if the 6-vectors are linearly dependent in a certain way
        
        # Simpler: try to find a linear combination of all 28 terms that gives
        # a valid rank-27 decomposition by setting one coefficient to zero
        
        # Use gradient-based search: minimize ||T - sum_{r=1}^{27} u_r⊗v_r⊗w_r||^2
        # Initialize with first 27 terms of the rank-28 solution
        rank = 27
        U = U28[:, :rank].copy()
        V = V28[:, :rank].copy()
        W = W28[:, :rank].copy()
        
        # Add the 28th term distributed among the 27 terms
        # This is a heuristic initialization
        U[:, -1] += U28[:, rank] * 0.5
        V[:, -1] += V28[:, rank] * 0.5
        W[:, -1] += W28[:, rank] * 0.5
        
        best_err = float('inf')
        best_solution = None
        
        max_iters = 100
        for it in range(max_iters):
            if time_left() < 10:
                break
            
            # ALS updates
            for i in range(n):
                for j in range(m):
                    M_ij = np.zeros((p, rank))
                    for k in range(p):
                        M_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    M_jk = np.zeros((n, rank))
                    for i in range(n):
                        M_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M_jk, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
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
        
        if best_solution is not None and best_err < 1e-8:
            U_sol, V_sol, W_sol = best_solution
            
            # Snap to half-integers
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
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

    # First, try the warm-start rank-27 search (more likely to succeed)
    if time_left() > 40:
        res_27 = try_find_rank_27_warm_start()
        if res_27:
            if verify(res_27["U"], res_27["V"], res_27["W"], 27):
                return res_27
    
    # Try the random-init rank-27 search
    if time_left() > 30:
        res_27 = try_find_rank_27()
        if res_27:
            if verify(res_27["U"], res_27["V"], res_27["W"], 27):
                return res_27
    
    # Fallback to rank 28
    res_28 = build_rank_28()
    
    # Verify rank-28 solution
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
