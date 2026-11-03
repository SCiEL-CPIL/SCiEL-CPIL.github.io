import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build rank-44 solution
    rank0 = 44
    U0 = np.zeros((dim_U, rank0))
    V0 = np.zeros((dim_V, rank0))
    W0 = np.zeros((dim_W, rank0))
    
    col = 0
    for k in range(p):
        U0[0, col] = 1
        V0[k, col] = 1
        W0[k*n, col] = 1
        col += 1
        U0[m, col] = 1
        V0[k, col] = 1
        W0[k*n+1, col] = 1
        col += 1
    
    for b in range(4):
        k0 = 2*b
        k1 = 2*b+1
        # P1
        U0[1, col] = 1; U0[m+2, col] = 1
        V0[p+k0, col] = 1; V0[2*p+k1, col] = 1
        W0[k0*n, col] = 1; W0[k1*n+1, col] = 1
        col += 1
        # P2
        U0[m+1, col] = 1; U0[m+2, col] = 1
        V0[p+k0, col] = 1
        W0[k0*n+1, col] = 1; W0[k1*n+1, col] = -1
        col += 1
        # P3
        U0[1, col] = 1
        V0[p+k1, col] = 1; V0[2*p+k1, col] = -1
        W0[k1*n, col] = 1; W0[k1*n+1, col] = 1
        col += 1
        # P4
        U0[m+2, col] = 1
        V0[2*p+k0, col] = 1; V0[p+k0, col] = -1
        W0[k0*n, col] = 1; W0[k0*n+1, col] = 1
        col += 1
        # P5
        U0[1, col] = 1; U0[2, col] = 1
        V0[2*p+k1, col] = 1
        W0[k0*n, col] = -1; W0[k1*n, col] = 1
        col += 1
        # P6
        U0[m+1, col] = 1; U0[1, col] = -1
        V0[p+k0, col] = 1; V0[p+k1, col] = 1
        W0[k1*n+1, col] = 1
        col += 1
        # P7
        U0[2, col] = 1; U0[m+2, col] = -1
        V0[2*p+k0, col] = 1; V0[2*p+k1, col] = 1
        W0[k0*n, col] = 1
        col += 1
    
    # Verify
    def verify(U, V, W):
        r = U.shape[1]
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    for i in range(n):
                        for k in range(p):
                            val = sum(W[k*n+i, r_] * U[i*m+j0, r_] * V[j0*p+k, r_] for r_ in range(r))
                            expected = 1.0 if (i==i0 and k==k0) else 0.0
                            if abs(val - expected) > 1e-9:
                                return False
        return True
    
    assert verify(U0, V0, W0)
    
    # Try to reduce rank
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    for r in range(rank0 - 1, 30, -1):
        found = False
        for remove in range(r+1):
            # Try removing term 'remove'
            idx = [i for i in range(r+1) if i != remove]
            U_r = U0[:, idx]
            V_r = V0[:, idx]
            W_r = W0[:, idx]
            if verify(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                found = True
                break
        if not found:
            break
    
    # Convert to lists
    U_list = [[int(round(x)) for x in row] for row in best_U]
    V_list = [[int(round(x)) for x in row] for row in best_V]
    W_list = [[int(round(x)) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
