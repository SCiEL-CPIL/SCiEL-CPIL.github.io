import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    # Target tensor indices
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0

    def verify(U, V, W):
        T_approx = np.einsum('ut,vt,wt->uvw', U, V, W)
        diff = np.abs(T_approx - T_target)
        return np.all(diff < 1e-9)

    # Schoolbook solution (Rank 40)
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_solution = {'U': U_school, 'V': V_school, 'W': W_school}
    best_rank = rank_schoolbook

    # Strategy: Construct a rank 36 solution using split 5 = 2 + 3
    # R(2, 4, 5) <= R(2, 4, 2) + R(2, 4, 3)
    # R(2, 4, 2) <= 2 * R(2, 2, 2) = 14
    # R(2, 4, 3) <= 2 * R(2, 2, 3)
    # If R(2, 2, 3) = 11, then R(2, 4, 3) <= 22
    # Total rank <= 14 + 22 = 36

    # Helper to build 2x2xn decomposition
    def build_2x2x2():
        # Strassen decomposition
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: i,j for U; j,k for V; k,i for W
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen coefficients
        # m1 = (A11-A22)(B21+B22)
        # m2 = (A11+A12)B22
        # m3 = A11(B11-B22)
        # m4 = A22(B11+B22)
        # m5 = (A12-A22)(B11+B12)
        # m6 = (A11+A22)(B21-B22)
        # m7 = (A12-A11)(B11+B21)
        
        # Mapping to tensor indices
        # A is 2x2, B is 2x2, C is 2x2
        # U[i, t] is coeff for A[i]
        # V[j, t] is coeff for B[j]
        # W[k, t] is coeff for C[k]
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # m1
        U[0, 0] = 1; U[3, 0] = -1
        V[2, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        
        # m2
        U[0, 1] = 1; U[1, 1] = 1
        V[3, 1] = 1
        W[1, 1] = 1
        
        # m3
        U[0, 2] = 1
        V[0, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1
        
        # m4
        U[3, 3] = 1
        V[0, 3] = 1; V[1, 3] = 1
        W[2, 3] = 1
        
        # m5
        U[1, 4] = 1; U[3, 4] = -1
        V[0, 4] = 1; V[1, 4] = 1
        W[1, 4] = 1; W[2, 4] = -1
        
        # m6
        U[0, 5] = 1; U[3, 5] = 1
        V[2, 5] = 1; V[3, 5] = -1
        W[2, 5] = 1
        
        # m7
        U[1, 6] = 1; U[0, 6] = -1
        V[0, 6] = 1; V[2, 6] = 1
        W[3, 6] = 1
        
        return U, V, W

    def build_2x2x3():
        # Search for rank 11 decomposition of 2x2x3
        # If not found, return None
        # Target tensor T: (4, 6, 6)
        T = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T[i*2+j, j*3+k, k*2+i] = 1.0
        
        r = 11
        # Simple random search with ALS
        for _ in range(20):
            U = np.random.randn(4, r)
            V = np.random.randn(6, r)
            W = np.random.randn(6, r)
            
            for it in range(50):
                # Update U
                T_flat = T.reshape(4, 36)
                M = np.einsum('vt,wt->tvp', V, W).reshape(r, 36)
                try:
                    U, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                    U = U.T
                except: break
                
                # Update V
                T_mat_v = T.transpose(1, 0, 2).reshape(6, 24)
                N = np.einsum('ut,wt->utw', U, W).reshape(r, 24)
                try:
                    V, _, _, _ = np.linalg.lstsq(N.T, T_mat_v.T, rcond=None)
                    V = V.T
                except: break
                
                # Update W
                T_mat_w = T.transpose(2, 0, 1).reshape(6, 24)
                P = np.einsum('ut,vt->utv', U, V).reshape(r, 24)
                try:
                    W, _, _, _ = np.linalg.lstsq(P.T, T_mat_w.T, rcond=None)
                    W = W.T
                except: break
                
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_approx = np.einsum('ut,vt,wt->uvw', U_snap, V_snap, W_snap)
            if np.all(np.abs(T_approx - T) < 1e-9):
                return U_snap, V_snap, W_snap
                
        return None

    U_222, V_222, W_222 = build_2x2x2()
    U_223, V_223, W_223 = build_2x2x3()
    
    if U_223 is not None:
        # Combine
        r_total = 14 + 22
        U = np.zeros((8, r_total))
        V = np.zeros((20, r_total))
        W = np.zeros((10, r_total))
        
        # Block 1: 2x2x2 (for k=0,1)
        # Map 2x2x2 to 2x4x2 part of 2x4x5
        # i in 0,1; j in 0,1,2,3; k in 0,1
        # We have two 2x2x2 blocks: j=0,1 and j=2,3
        # Block 1: j=0,1
        U[0:4, 0:7] = U_222
        V[0:4, 0:7] = V_222 # j=0,1 -> indices 0,1,2,3 in V (j*2+k)
        # V index for j=0,1 and k=0,1 is j*2+k.
        # j=0, k=0 -> 0; j=0, k=1 -> 1; j=1, k=0 -> 2; j=1, k=1 -> 3.
        # So V[0:4] is correct.
        # W index for k=0,1 and i=0,1 is k*2+i.
        # k=0, i=0 -> 0; k=0, i=1 -> 1; k=1, i=0 -> 2; k=1, i=1 -> 3.
        # So W[0:4] is correct.
        W[0:4, 0:7] = W_222
        
        # Block 2: j=2,3
        # V index for j=2,3 and k=0,1 is j*2+k.
        # j=2, k=0 -> 4; j=2, k=1 -> 5; j=3, k=0 -> 6; j=3, k=1 -> 7.
        # So V[4:8] corresponds to j=2,3.
        U[0:4, 7:14] = U_222
        V[4:8, 7:14] = V_222
        W[0:4, 7:14] = W_222
        
        # Block 3: 2x2x3 (for k=2,3,4)
        # We have two 2x2x3 blocks: j=0,1 and j=2,3
        # Block 3a: j=0,1
        # U is same
        U[0:4, 14:25] = U_223
        # V index for j=0,1 and k=2,3,4 is j*5+k.
        # j=0, k=2 -> 2; j=0, k=3 -> 3; j=0, k=4 -> 4.
        # j=1, k=2 -> 7; j=1, k=3 -> 8; j=1, k=4 -> 9.
        # So V indices are [2,3,4, 7,8,9].
        V[2:5, 14:25] = V_223[0:3, :]
        V[7:10, 14:25] = V_223[3:6, :]
        # W index for k=2,3,4 and i=0,1 is k*2+i.
        # k=2, i=0 -> 4; k=2, i=1 -> 5.
        # k=3, i=0 -> 6; k=3, i=1 -> 7.
        # k=4, i=0 -> 8; k=4, i=1 -> 9.
        # So W indices are [4,5, 6,7, 8,9].
        W[4:10, 14:25] = W_223
        
        # Block 3b: j=2,3
        U[0:4, 25:36] = U_223
        # V index for j=2,3 and k=2,3,4 is j*5+k.
        # j=2, k=2 -> 12; j=2, k=3 -> 13; j=2, k=4 -> 14.
        # j=3, k=2 -> 17; j=3, k=3 -> 18; j=3, k=4 -> 19.
        # So V indices are [12,13,14, 17,18,19].
        V[12:15, 25:36] = V_223[0:3, :]
        V[17:20, 25:36] = V_223[3:6, :]
        # W is same as 3a
        W[4:10, 25:36] = W_223
        
        if verify(U, V, W):
            best_solution = {'U': U, 'V': V, 'W': W}
            best_rank = 36

    return {
        "U": best_solution['U'].tolist(),
        "V": best_solution['V'].tolist(),
        "W": best_solution['W'].tolist()
    }
