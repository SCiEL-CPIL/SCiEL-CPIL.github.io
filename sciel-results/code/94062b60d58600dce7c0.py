import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Schoolbook rank is 64
    # Strassen-based rank is 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # We have 8 blocks of 2x2x2
    # j0 in {0, 2}
    # k0 in {0, 2, 4, 6}
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Check deadline
            if _time_left() < 1.0:
                # If we run out of time, we might not finish, but the construction is fast.
                # If we are here, we are in the loop.
                pass
            
            # Local indices:
            # i in {0, 1}
            # j in {j0, j0+1}
            # k in {k0, k0+1}
            
            # Strassen terms for 2x2x2
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # Let's define a helper to add a term
            def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
                # u_coeffs: dict {(i,j): val}
                # v_coeffs: dict {(j,k): val}
                # w_coeffs: dict {(k,i): val}
                for (i, j), val in u_coeffs.items():
                    if val != 0:
                        U[i*m + j][col_idx] += val
                for (j, k), val in v_coeffs.items():
                    if val != 0:
                        V[j*p + k][col_idx] += val
                for (k, i), val in w_coeffs.items():
                    if val != 0:
                        W[k*n + i][col_idx] += val
            
            # Term 1: M1 = (A00 + A11)(B00 + B11)
            # A00: i=0, j=j0. A11: i=1, j=j0+1
            # B00: j=j0, k=k0. B11: j=j0+1, k=k0+1
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # W indices: k*n + i
            # C00: k=k0, i=0 -> W[k0*n+0]
            # C01: k=k0+1, i=0 -> W[(k0+1)*n+0]
            # C10: k=k0, i=1 -> W[k0*n+1]
            # C11: k=k0+1, i=1 -> W[(k0+1)*n+1]
            
            # M1 contributes:
            # C00: 1
            # C11: 1
            add_term(
                {(0, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1
            
            # Term 2: M2 = (A10 + A11)B00
            # A10: i=1, j=j0. A11: i=1, j=j0+1
            # B00: j=j0, k=k0
            # Contributes to C10 (1), C11 (-1)
            add_term(
                {(1, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1},
                {(k0, 1): 1, (k0+1, 1): -1},
                col
            )
            col += 1
            
            # Term 3: M3 = A00(B01 - B11)
            # A00: i=0, j=j0
            # B01: j=j0, k=k0+1. B11: j=j0+1, k=k0+1
            # Contributes to C01 (1), C11 (1)
            add_term(
                {(0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(k0+1, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1
            
            # Term 4: M4 = A11(B10 - B00)
            # A11: i=1, j=j0+1
            # B10: j=j0+1, k=k0. B00: j=j0, k=k0
            # Contributes to C00 (1), C10 (1)
            add_term(
                {(1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(k0, 0): 1, (k0, 1): 1},
                col
            )
            col += 1
            
            # Term 5: M5 = (A00 + A01)B11
            # A00: i=0, j=j0. A01: i=0, j=j0+1
            # B11: j=j0+1, k=k0+1
            # Contributes to C00 (-1), C01 (1)
            add_term(
                {(0, j0): 1, (0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(k0, 0): -1, (k0+1, 0): 1},
                col
            )
            col += 1
            
            # Term 6: M6 = (A10 - A00)(B00 + B01)
            # A10: i=1, j=j0. A00: i=0, j=j0
            # B00: j=j0, k=k0. B01: j=j0, k=k0+1
            # Contributes to C11 (1)
            add_term(
                {(1, j0): 1, (0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(k0+1, 1): 1},
                col
            )
            col += 1
            
            # Term 7: M7 = (A01 - A11)(B10 + B11)
            # A01: i=0, j=j0+1. A11: i=1, j=j0+1
            # B10: j=j0+1, k=k0. B11: j=j0+1, k=k0+1
            # Contributes to C00 (1)
            add_term(
                {(0, j0+1): 1, (1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1},
                col
            )
            col += 1
    
    # Validation step to ensure correctness
    # If validation fails, fallback to schoolbook (rank 64)
    # This is fast enough for small dimensions
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1

    return {
        "U": U,
        "V": V,
        "W": W
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
