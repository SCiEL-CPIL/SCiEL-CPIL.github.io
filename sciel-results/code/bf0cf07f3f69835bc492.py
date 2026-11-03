import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the Strassen decomposition for 4x4x4
    # Strassen for 2x2x2
    # U, V, W for 2x2x2
    # i: 0..3, j: 0..3, k: 0..3
    # i = a*2 + b, j = c*2 + d, k = e*2 + f
    # A[i,j] = A[a,c]_{b,d}
    # B[j,k] = B[c,e]_{d,f}
    # C[i,k] = C[a,e]_{b,f}
    
    # Strassen 2x2x2 factors
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to define U, V, W for each M
    # M1: A00*B00, A00*B11, A11*B00, A11*B11
    # This is a sum of 4 terms.
    # We can represent M1 as a combination of 4 rank-1 tensors.
    # But Strassen uses 7 multiplications, each of which is a dot product.
    # Each dot product is a rank-4 tensor (2x2x2).
    # So M1 is a sum of 4 rank-1 tensors in the 2x2x2 space? No.
    # M1 is a scalar. The tensor for M1 is the sum of the tensors for the 4 products.
    
    # Let's define the factors for a single 2x2x2 multiplication (schoolbook rank 4)
    # U_sc[i, r], V_sc[j, r], W_sc[k, r]
    # r = a*2 + c (block index), s = b*2 + d (element index)
    # This is getting too complex to derive manually.
    
    # Instead, we will use the random search but with a better heuristic.
    # We know the rank is at most 65.
    # We will try to find a decomposition of rank 65.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        a = i // m
        b = i % m
        for j in range(m*p):
            b_j = j // p
            c = j % p
            if b == b_j:
                for k in range(p*n):
                    c_k = k // n
                    a_k = k % n
                    if c == c_k and a == a_k:
                        T[i, j, k] = 1.0

    best_rank = 80
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U0 = np.zeros((n*m, 80))
    V0 = np.zeros((m*p, 80))
    W0 = np.zeros((p*n, 80))
    col = 0
    for i in range(n*m):
        a = i // m
        b = i % m
        for j in range(m*p):
            b_j = j // p
            c = j % p
            if b == b_j:
                for k in range(p*n):
                    c_k = k // n
                    a_k = k % n
                    if c == c_k and a == a_k:
                        U0[i, col] = 1
                        V0[j, col] = 1
                        W0[k, col] = 1
                        col += 1
    best_U, best_V, best_W = U0, V0, W0

    start_time = time.time()
    max_time = 140
    
    # Try to find a rank 65 solution
    # We will use a local search starting from a random initialization
    # but we will bias the initialization to be sparse.
    
    for r in range(65, 19, -5):
        if time.time() - start_time > max_time:
            break
        
        for restart in range(10):
            if time.time() - start_time > max_time:
                break
            
            U = np.random.randint(-1, 2, (n*m, r)).astype(float)
            V = np.random.randint(-1, 2, (m*p, r)).astype(float)
            W = np.random.randint(-1, 2, (p*n, r)).astype(float)
            
            # Local search
            for iter in range(5000):
                if time.time() - start_time > max_time:
                    break
                
                # Update one random entry
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    i, rr = np.random.randint(n*m), np.random.randint(r)
                    U[i, rr] = np.round(U[i, rr] * 2) / 2 + np.random.choice([-0.5, 0, 0.5])
                elif mat_idx == 1:
                    j, rr = np.random.randint(m*p), np.random.randint(r)
                    V[j, rr] = np.round(V[j, rr] * 2) / 2 + np.random.choice([-0.5, 0, 0.5])
                else:
                    k, rr = np.random.randint(p*n), np.random.randint(r)
                    W[k, rr] = np.round(W[k, rr] * 2) / 2 + np.random.choice([-0.5, 0, 0.5])
                
                if iter % 50 == 0:
                    T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                    T_approx_int = np.round(T_approx * 2)
                    if np.allclose(T_approx_int, T * 2, atol=1e-9):
                        best_rank = r
                        best_U = np.round(U * 2) / 2
                        best_V = np.round(V * 2) / 2
                        best_W = np.round(W * 2) / 2
                        break
            
            if best_rank == r:
                break

    U_list = [[val for val in row] for row in best_U]
    V_list = [[val for val in row] for row in best_V]
    W_list = [[val for val in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
