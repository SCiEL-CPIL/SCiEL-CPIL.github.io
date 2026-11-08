import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = 4, 4, 7

    # Build the multiplication tensor T[A,B,C] where C = A*B
    # T is a 3-tensor: T[i*m+j, j2*p+k, k*n+i] = delta(j,j2)*delta... 
    # Actually the standard matrix multiplication tensor has entries:
    # T[i,j,k; j2,j3,k2] ... let me think more carefully.
    # 
    # For matrix multiplication: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor has indices: (i,j) for A, (j,k) for B, (i,k) for C
    # So T[(i,m,j),(j,p,k),(k,n,i)] = 1 if the j in A and B match
    # 
    # In our indexing:
    # A is flattened: A_flat[i*m + j]
    # B is flattened: B_flat[j*p + k]  
    # C is flattened: C_flat[k*n + i]
    # 
    # The tensor T[a_idx, b_idx, c_idx] = 1 iff a_idx=i*m+j, b_idx=j*p+k, c_idx=k*n+i
    # i.e., a_idx%m == b_idx//p and c_idx%n == a_idx//m and c_idx//n == b_idx%p
    
    nA, nB, nC = n*m, m*p, p*n  # 16, 28, 28
    
    # The tensor T has T[a,b,c] = 1 if (a%m == b//p) and (a//m == c%n) and (b%p == c//n)
    # Otherwise T[a,b,c] = 0
    
    # A rank-1 term (u,v,w) contributes u[a]*v[b]*w[c]
    # We need sum over all terms of u_t[a]*v_t[b]*w_t[c] = T[a,b,c] for all a,b,c
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)

    # Standard schoolbook decomposition: rank = n * m * p = 112
    # This is guaranteed correct.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                add_term(u, v, w)
    
    rank = len(terms)
    
    # Build U, V, W as numpy arrays
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    # Verify correctness
    def verify_decomposition(U_arr, V_arr, W_arr, tol=1e-9):
        for a in range(nA):
            i_a, j_a = a // m, a % m
            for b in range(nB):
                j_b, k_b = b // p, b % p
                for c in range(nC):
                    k_c, i_c = c // n, c % n
                    expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                    actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    if not verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
