"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 145  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi,(ga,gb,gc) in enumerate(blocks):
            for t in range(7):
                col = bi*7+t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li,lj), col] = U_s[li*2+lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj,lk), col] = V_s[lj*2+lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk,li), col] = W_s[lk*2+li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()
    
    # Verify rank-28 works
    if not verify(U28, V28, W28, tol=1e-8):
        # fallback to schoolbook
        U = np.zeros((n*m, 32)); V = np.zeros((m*p, 32)); W = np.zeros((p*n, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
        return to_lists(U, V, W, 32)

    # === TENSOR SPLITTING STRATEGY ===
    # Split the 2x4x4 tensor into two 1x4x4 sub-tensors (one per row of A)
    # T1: row 0 of A (j=0,1,2,3) contributes to output indices where i=0
    # T2: row 1 of A (j=0,1,2,3) contributes to output indices where i=1
    # 
    # For sub-problem 1x4x4: multiply 1x4 matrix by 4x4 matrix to get 4x1 matrix
    # Indices: a = j (0..3), b = j*p+k (0..15), c = k (0..3)
    # T_sub[j, j*p+k, k] = 1
    #
    # Rank of 1x4x4 tensor: We need to find minimal rank.
    # The tensor has 16 non-zero entries. Let's find its rank via decomposition.
    
    def find_rank_1x4x4(max_rank=16, time_budget=30):
        """Find minimal rank decomposition of 1x4x4 multiplication tensor."""
        start = time.time()
        m_sub, p_sub = 4, 4
        # Build the 1x4x4 tensor
        T_sub = np.zeros((m_sub, m_sub * p_sub, p_sub))
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[j, j * p_sub + k, k] = 1.0
        
        def verify_sub(U, V, W, tol=1e-10):
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            return np.max(np.abs(recon - T_sub)) < tol
        
        # Try ranks from 13 upward (theoretical lower bound is 16, but let's check)
        # Actually, let's try to find the minimal rank using optimization
        
        best_result = None
        for r in range(13, 17):  # Try ranks 13-16
            if time.time() - start > time_budget:
                break
            
            # Use random restarts with gradient descent
            for seed in range(200):
                if time.time() - start > time_budget:
                    break
                rng = np.random.RandomState(seed)
                U = rng.randn(m_sub, r) * 0.3
                V = rng.randn(m_sub * p_sub, r) * 0.3
                W = rng.randn(p_sub, r) * 0.3
                
                for it in range(3000):
                    recon = np.einsum('at,bt,ct->abc', U, V, W)
                    F = recon - T_sub
                    err = float(np.sum(F * F))
                    if err < 1e-14:
                        # Snap to half-integers
                        U = np.round(U * 2) / 2
                        V = np.round(V * 2) / 2
                        W = np.round(W * 2) / 2
                        if verify_sub(U, V, W):
                            best_result = (r, U, V, W)
                            return best_result
                        break
                    
                    gU = np.einsum('abc,bt,ct->at', F, V, W)
                    gV = np.einsum('abc,at,ct->bt', F, U, W)
                    gW = np.einsum('abc,at,bt->ct', F, U, V)
                    lr = 0.05 / (1.0 + err)
                    U -= lr * gU
                    V -= lr * gV
                    W -= lr * gW
        
        return best_result

    # Try tensor splitting: find rank for each 1x4x4 sub-problem
    result_sub = find_rank_1x4x4(max_rank=16, time_budget=60)
    
    if result_sub is not None:
        r_sub, U_sub, V_sub, W_sub = result_sub
        total_rank = 2 * r_sub
        
        # Build the full 2x4x4 decomposition from two 1x4x4 decompositions
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        for i in range(n):
            offset = i * r_sub
            for j in range(m):
                U_full[i * m + j, offset:offset + r_sub] = U_sub[j, :]
            for k in range(p):
                for j in range(m):
                    V_full[j * p + k, offset:offset + r_sub] = V_sub[j * p + k, :]
            for k in range(p):
                W_full[k * n + i, offset:offset + r_sub] = W_sub[k, :]
        
        if verify(U_full, V_full, W_full, tol=1e-8):
            return to_lists(U_full, V_full, W_full, total_rank)

    # === Fallback: Attempt rank-27 by finding and eliminating a linear dependency ===
    def find_dependency(U, V, W, r):
        """Find a linear dependency among the r rank-1 tensors."""
        dim = (n*m) * (m*p) * (p*n)
        M = np.zeros((dim, r))
        for t in range(r):
            M[:, t] = np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t]).flatten()
        try:
            _, s, vh = np.linalg.svd(M, full_matrices=False)
            if s[-1] < 1e-10:
                null_vec = vh[-1, :]
                idx = np.argmax(np.abs(null_vec))
                return null_vec, idx
        except np.linalg.LinAlgError:
            pass
        return None

    dep = find_dependency(U28, V28, W28, 28)
    if dep is not None:
        coeffs, rem_idx = dep
        keep = [i for i in range(28) if i != rem_idx]
        U27 = U28[:, keep]
        V27 = V28[:, keep]
        W27 = W28[:, keep]
        if verify(U27, V27, W27, tol=1e-8):
            return to_lists(U27, V27, W27, 27)

    # Fallback: verified rank-28 Strassen
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
