"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    schoolbook_rank = n * m * p  # 32
    deadline = time.time() + 130  # safety margin before 150s kill

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
        # C[i,j] -> index j*2 + i (trace convention k*n+i with n=2, k=j, i=i)
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
    if not verify(U28, V28, W28, tol=1e-8):
        U = np.zeros((n*m, schoolbook_rank)); V = np.zeros((m*p, schoolbook_rank)); W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
        return to_lists(U, V, W, schoolbook_rank)

    # Attempt to find rank 27
    # Strategy: Fix first 26 Strassen terms. Find 2 terms to complete the tensor.
    # If the residual has rank 1, we can merge? No.
    # If we can find ANY 27-term decomposition, we return it.
    
    def try_rank_27(seed):
        rng = np.random.RandomState(seed)
        r = 27
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Initialize first 26 with Strassen
        U[:, :26] = U28[:, :26]
        V[:, :26] = V28[:, :26]
        W[:, :26] = W28[:, :26]
        
        # Initialize 27th with noise
        U[:, 26] = rng.randn(n*m) * 0.1
        V[:, 26] = rng.randn(m*p) * 0.1
        W[:, 26] = rng.randn(p*n) * 0.1
        
        # Optimize
        from scipy.optimize import minimize
        
        def objective(params):
            Uf = params[0:n*m].reshape(n*m, r)
            Vf = params[n*m:n*m+m*p].reshape(m*p, r)
            Wf = params[n*m+m*p:n*m+m*p+p*n].reshape(p*n, r)
            recon = np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
            return np.sum((recon - T)**2)
            
        # This is too large for general minimize.
        # Use Gauss-Newton style update manually.
        
        Uf = U.copy(); Vf = V.copy(); Wf = W.copy()
        
        for it in range(500):
            if time.time() > deadline - 5:
                break
            recon = np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
            F = recon - T
            err = np.sum(F**2)
            if err < 1e-12:
                break
            
            # Gradients
            gU = np.einsum('abc,bt,ct->at', F, Vf, Wf)
            gV = np.einsum('abc,at,ct->bt', F, Uf, Wf)
            gW = np.einsum('abc,at,bt->ct', F, Uf, Vf)
            
            lr = 0.01 / (1 + err)
            Uf -= lr * gU
            Vf -= lr * gV
            Wf -= lr * gW
        
        # Snap to half-integers
        Uf = np.round(Uf * 2) / 2
        Vf = np.round(Vf * 2) / 2
        Wf = np.round(Wf * 2) / 2
        
        if verify(Uf, Vf, Wf, tol=1e-8):
            return Uf, Vf, Wf
        return None

    # Try a few seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if time.time() > deadline - 10:
            break
        res = try_rank_27(seed)
        if res is not None:
            Uf, Vf, Wf = res
            return to_lists(Uf, Vf, Wf, 27)

    # Fallback: verified rank-28 Strassen
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
