import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    def make_schoolbook():
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def verify_exact(U_mat, V_mat, W_mat):
        """Exact verification using the tensor condition."""
        rank = U_mat.shape[1]
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                expected = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                actual = 0.0
                                for c in range(rank):
                                    actual += W_mat[k*n+i, c] * U_mat[i2*m+j, c] * V_mat[j2*p+k2, c]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    def verify_random(U_mat, V_mat, W_mat, n_tests=10):
        """Quick verification using random test matrices."""
        rank = U_mat.shape[1]
        for _ in range(n_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(rank):
                A_vec = A.flatten()
                B_vec = B.flatten()
                u_val = U_mat[:, c] @ A_vec
                v_val = V_mat[:, c] @ B_vec
                W_col = W_mat[:, c]
                C_computed += W_col.reshape(p, n).T * (u_val * v_val)
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True
    
    def alternating_minimize(U, V, W, rank, n_iters=40):
        """Alternating least squares to minimize the error."""
        constraints = []
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                target = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                constraints.append((i, k, i2, j, j2, k2, target))
        
        for it in range(n_iters):
            # Fix U, V; solve for W row by row
            for i in range(n):
                for k in range(p):
                    A_sys = []
                    b_sys = []
                    for (ci, ck, ci2, cj, cj2, ck2, target) in constraints:
                        if ci == i and ck == k:
                            row = np.array([U[ci2*m+cj, c] * V[cj2*p+ck2, c] for c in range(rank)])
                            A_sys.append(row)
                            b_sys.append(target)
                    if A_sys:
                        A_sys = np.array(A_sys)
                        b_sys = np.array(b_sys)
                        W_row, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W[k*n+i, :] = W_row
            
            # Fix V, W; solve for U row by row
            for i2 in range(n):
                for j in range(m):
                    A_sys = []
                    b_sys = []
                    for (ci, ck, ci2, cj, cj2, ck2, target) in constraints:
                        if ci2 == i2 and cj == j:
                            row = np.array([W[ck*n+ci, c] * V[cj2*p+ck2, c] for c in range(rank)])
                            A_sys.append(row)
                            b_sys.append(target)
                    if A_sys:
                        A_sys = np.array(A_sys)
                        b_sys = np.array(b_sys)
                        U_row, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        U[i2*m+j, :] = U_row
            
            # Fix U, W; solve for V row by row
            for j2 in range(m):
                for k2 in range(p):
                    A_sys = []
                    b_sys = []
                    for (ci, ck, ci2, cj, cj2, ck2, target) in constraints:
                        if cj2 == j2 and ck2 == k2:
                            row = np.array([W[ck*n+ci, c] * U[ci2*m+cj, c] for c in range(rank)])
                            A_sys.append(row)
                            b_sys.append(target)
                    if A_sys:
                        A_sys = np.array(A_sys)
                        b_sys = np.array(b_sys)
                        V_row, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        V[j2*p+k2, :] = V_row
        
        return U, V, W
    
    deadline = time.time() + 115
    
    U_sb, V_sb, W_sb = make_schoolbook()
    best = {'U': U_sb, 'V': V_sb, 'W': W_sb, 'rank': rank_school}
    
    # Try to find lower rank decompositions
    for rank in range(12, rank_school):
        if time.time() > deadline - 25:
            break
        
        max_restarts = 4 if rank <= 16 else 2
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 25:
                break
            
            U, V, W = np.random.randn(n*m, rank), np.random.randn(m*p, rank), np.random.randn(p*n, rank)
            U, V, W = alternating_minimize(U, V, W, rank, n_iters=30)
            
            if verify_exact(U, V, W):
                best = {'U': U, 'V': V, 'W': W, 'rank': rank}
                found = True
                break
        
        if found:
            continue
        else:
            if rank > 18:
                break
    
    return {"U": best['U'].tolist(), "V": best['V'].tolist(), "W": best['W'].tolist()}
# EVOLVE-BLOCK-END
