import numpy as np
import time
from fractions import Fraction
import sys
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline (10s safety margin)

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

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
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    # Build rank-28 decomposition (known good)
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

    # Try to find lower rank using structured search
    # Key idea: use the flattening approach. For each fixed k, we have a matrix
    # A_k of size (n*m) x (m*p) where A_k[i*m+j, j'*p+k] = delta_{j,j'}
    # The tensor rank is related to the sum of ranks of these flattenings.
    # 
    # Alternative: Try rank 27 by solving a system of equations.
    # We'll use a different parameterization: assume W has a specific structure.
    
    def try_rank_27():
        """
        Try to find rank-27 decomposition by exploiting the structure.
        
        For n=2, the tensor can be viewed as a map from M(3,5) to M(2,5).
        We can try to find a decomposition where the W matrix has a specific form.
        
        Strategy: Fix W to have a structured form (e.g., block diagonal or sparse),
        then solve for U and V linearly.
        """
        rank = 27
        
        # Try different structures for W
        # Structure 1: W has entries in {-1, 0, 1} with limited support
        
        np.random.seed(42)
        
        best_err = float('inf')
        best_solution = None
        
        # Try multiple random structured initializations
        for attempt in range(50):
            if time_left() < 10:
                break
            
            # Initialize with random sparse half-integer values
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Each column has a few non-zero entries
            for r in range(rank):
                # U column: 2-4 nonzeros
                num_u = np.random.randint(2, 5)
                idx_u = np.random.choice(n * m, size=num_u, replace=False)
                for idx in idx_u:
                    U[idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
                
                # V column: 2-4 nonzeros
                num_v = np.random.randint(2, 5)
                idx_v = np.random.choice(m * p, size=num_v, replace=False)
                for idx in idx_v:
                    V[idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
                
                # W column: 2-4 nonzeros
                num_w = np.random.randint(2, 5)
                idx_w = np.random.choice(p * n, size=num_w, replace=False)
                for idx in idx_w:
                    W[idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
            
            # ALS refinement
            max_iters = 20
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
                
                if err < 1e-12:
                    break
            
            if best_err < 1e-10:
                break
        
        if best_solution is not None and best_err < 1e-8:
            U_sol, V_sol, W_sol = best_solution
            
            # Snap to half-integers
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            # Verify
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

    # Try rank 27
    if time_left() > 60:
        res_27 = try_rank_27()
        if res_27:
            if verify(res_27["U"], res_27["V"], res_27["W"], 27):
                return res_27

    # Fallback to rank 28
    res_28 = build_rank_28()
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
