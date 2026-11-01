import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second buffer
    
    n, m, p = DIMENSIONS
    
    # Target Tensor Construction
    # A: n x m, B: m x p, C: p x n
    # Indices:
    # A: i (0..n-1), j (0..m-1) -> flat a = i*m + j
    # B: j (0..m-1), k (0..p-1) -> flat b = j*p + k
    # C: k (0..p-1), i (0..n-1) -> flat c = k*n + i  (Trace convention W[k*n+i, r])
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies C is stored as p x n?
    # "multiplying a 2 by 2 matrix by a 2 by 3 matrix"
    # A is 2x2, B is 2x3.
    # C = A * B is 2x3.
    # So C has dimensions n x p?
    # n=2, p=3. C is 2x3.
    # But the task says W shape is (p*n, rank).
    # And "W[k*n+i,r]".
    # If C is 2x3 (n x p), then index is i (0..1), k (0..2).
    # Flat index i*p + k?
    # But the convention says W[k*n+i].
    # This implies the storage is column-major for C? Or C is transposed?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually trace convention for matrix multiplication $C = AB$ where $C$ is $n \times p$?
    # Let's check the current program's W usage.
    # set_w(k, i, r, val): W[k*n+i][r] = val
    # In the current program:
    # n=2, m=2, p=3.
    # set_w(0, 0, 0, 1) -> W[0*2+0] = W[0].
    # set_w(1, 1, 0, 1) -> W[1*2+1] = W[3].
    # set_w(0, 1, 1, 1) -> W[0*2+1] = W[1].
    # set_w(1, 1, 1, -1) -> W[1*2+1] = W[3].
    #
    # Let's map C indices.
    # C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # i in 0..1, k in 0..2.
    # In current program:
    # M1: (A00+A11)(B00+B11).
    # A00 -> i=0, j=0. A11 -> i=1, j=1.
    # B00 -> j=0, k=0. B11 -> j=1, k=1.
    # Product M1 = A00B00 + A00B11 + A11B00 + A11B11.
    # Contributions to C:
    # C[0,0] gets A00B00.
    # C[0,1] gets A00B11? No, B11 is j=1, k=1. A00 is i=0, j=0. Mismatch in j.
    # Wait, the decomposition is $C_{ik} = \sum_r U_{ij,r} V_{jk,r} W_{ik,r}$?
    # No, the tensor decomposition is $T_{a,b,c} = \sum_r U_{a,r} V_{b,r} W_{c,r}$.
    # $a$ corresponds to $A_{ij}$, $b$ to $B_{jk}$, $c$ to $C_{ik}$.
    # The sum over $r$ gives the coefficient for the term $A_{ij} B_{jk}$ in $C_{ik}$.
    # Actually, $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
    # So $T_{a,b,c} = 1$ if $a=(i,j), b=(j,k), c=(i,k)$?
    # Let's check the current program's W indices again.
    # set_w(k, i, r, val) -> W[k*n+i].
    # This maps $(k,i)$ to index $k*n+i$.
    # If $C$ is $n \times p$ (2x3), then $i \in \{0,1\}, k \in \{0,1,2\}$.
    # Index $k*n+i$ means $k$ is the major index (0,1,2) and $i$ is minor (0,1).
    # So $W$ index 0: k=0, i=0 -> C[0,0].
    # W index 1: k=0, i=1 -> C[1,0].
    # W index 2: k=1, i=0 -> C[0,1].
    # W index 3: k=1, i=1 -> C[1,1].
    # W index 4: k=2, i=0 -> C[0,2].
    # W index 5: k=2, i=1 -> C[1,2].
    #
    # Let's check M1 in current program:
    # set_w(0, 0, 0, 1) -> W[0] = 1. (C[0,0])
    # set_w(1, 1, 0, 1) -> W[3] = 1. (C[1,1])
    # This matches $C_{00}$ and $C_{11}$ getting contributions from M1.
    # M1 = (A00+A11)(B00+B11).
    # A00B00 contributes to C[0,0].
    # A11B11 contributes to C[1,1].
    # A00B11 (j mismatch) and A11B00 (j mismatch) do not contribute to C directly in standard multiplication?
    # Wait, in tensor decomposition, $U_{a,r} V_{b,r} W_{c,r}$ sums over $r$.
    # For a fixed $a,b,c$, the value is $\sum_r U_{a,r} V_{b,r} W_{c,r}$.
    # This value must equal 1 if $b$'s $j$ matches $a$'s $j$ and $c$'s $i$ matches $a$'s $i$ and $c$'s $k$ matches $b$'s $k$.
    # i.e., $a=(i,j), b=(j,k), c=(i,k)$.
    # In M1:
    # r=0.
    # U[0,0]=1 (A00), U[3,0]=1 (A11). (Indices: A00->0, A11->3? A is 2x2. 0:00, 1:01, 2:10, 3:11. Yes.)
    # V[0,0]=1 (B00), V[4,0]=1 (B11). (B is 2x3. 0:00, 1:01, 2:02, 3:10, 4:11, 5:12. Yes.)
    # W[0,0]=1 (C00), W[3,0]=1 (C11). (C is 2x3 stored as k*n+i. 0:00, 1:10, 2:01, 3:11, 4:02, 5:12. Yes.)
    #
    # Check term a=0 (A00), b=0 (B00), c=0 (C00).
    # Sum r=0: U[0,0]V[0,0]W[0,0] = 1*1*1 = 1. Correct.
    # Check term a=3 (A11), b=4 (B11), c=3 (C11).
    # Sum r=0: U[3,0]V[4,0]W[3,0] = 1*1*1 = 1. Correct.
    # Check term a=0 (A00), b=4 (B11), c=2 (C01).
    # j in a is 0, j in b is 1. Mismatch. Should be 0.
    # Sum r=0: U[0,0]V[4,0]W[2,0] = 1*1*0 = 0. Correct.
    
    # So the tensor T is defined as:
    # T[a, b, c] = 1 if j_a == j_b and i_a == i_c and k_b == k_c.
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    # I will construct this tensor and search for rank 10.

    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1
        return T

    T = build_tensor(n, m, p)
    
    best_solution = None
    best_rank = 11
    
    # Try to find rank 10
    target_rank = 10
    max_iters = 500
    tol = 1e-10
    
    # Random restarts
    num_restarts = 50
    time_per_restart = 100 / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Scale to avoid overflow/underflow
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        converged = False
        for it in range(max_iters):
            # Update U
            # T = sum U_r V_r W_r
            # Fix V, W. Solve for U.
            # T_flat = (V * W) * U?
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a, T[a,:,:] = sum_r U[a,r] (V[:,r] W[:,r])^T?
            # Let M_r = V[:,r] W[:,r] (outer product).
            # T[a,:,:] = sum_r U[a,r] M_r.
            # Vectorize: T_a = sum_r U[a,r] vec(M_r).
            # This is a linear system for U[a,:].
            # But we can do it for all a at once.
            # T (n*m, m*p, p*n) -> (n*m, (m*p)*(p*n))
            # RHS = T.reshape(n*m, -1)
            # Design matrix X ( (m*p)*(p*n) , target_rank )
            # X[:, r] = vec(V[:,r] W[:,r])
            
            X_u = np.zeros((m*p*p*n, target_rank))
            for rr in range(target_rank):
                M = np.outer(V[:,rr], W[:,rr])
                X_u[:,rr] = M.flatten()
            
            try:
                U, _, _, _ = np.linalg.lstsq(X_u, T.reshape(n*m, -1).T, rcond=None)
                U = U.T
            except:
                break
                
            # Update V
            X_v = np.zeros((n*m*p*n, target_rank))
            for rr in range(target_rank):
                M = np.outer(U[:,rr], W[:,rr])
                X_v[:,rr] = M.flatten()
            
            try:
                V, _, _, _ = np.linalg.lstsq(X_v, T.reshape(m*p, -1).T, rcond=None)
                V = V.T
            except:
                break
                
            # Update W
            X_w = np.zeros((n*m*m*p, target_rank))
            for rr in range(target_rank):
                M = np.outer(U[:,rr], V[:,rr])
                X_w[:,rr] = M.flatten()
            
            try:
                W, _, _, _ = np.linalg.lstsq(X_w, T.reshape(p*n, -1).T, rcond=None)
                W = W.T
            except:
                break
                
            # Check convergence
            if it % 10 == 0:
                # Reconstruct and check error
                # Error = || T - sum U V W ||
                # Approximate by checking a few entries or full norm
                pass # Skip full norm for speed in inner loop, check at end
        
        # Check solution
        # Reconstruct
        T_rec = np.zeros((n*m, m*p, p*n))
        for rr in range(target_rank):
            T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
        
        err = np.linalg.norm(T - T_rec)
        if err < 1e-6:
            # Check if entries are close to integers/half-integers
            # Scale factors?
            # The decomposition is not unique. We can scale columns.
            # Try to find scaling that makes entries integer/half-integer.
            # Simple heuristic: check if entries are close to k/2
            is_half_int = True
            for mat in [U, V, W]:
                for val in mat:
                    # Check if val is close to integer or half-integer
                    nearest = round(val * 2) / 2
                    if abs(val - nearest) > 1e-3:
                        is_half_int = False
                        break
                if not is_half_int:
                    break
            
            if is_half_int:
                # Snap to half-integers
                U_s = np.round(U * 2) / 2
                V_s = np.round(V * 2) / 2
                W_s = np.round(W * 2) / 2
                
                # Verify exactness
                T_rec_exact = np.zeros((n*m, m*p, p*n))
                for rr in range(target_rank):
                    T_rec_exact += np.einsum('a,b,c->abc', U_s[:,rr], V_s[:,rr], W_s[:,rr])
                
                if np.allclose(T_rec_exact, T, atol=1e-9):
                    best_solution = (U_s, V_s, W_s)
                    best_rank = target_rank
                    break # Found rank 10

    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return known rank 11 solution
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        # 1. (A00+A11)(B00+B11) -> M1
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # 2. (A10+A11)B00 -> M2
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # 3. A00(B01-B11) -> M3
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # 4. A11(B10-B00) -> M4
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # 5. (A00+A01)B11 -> M5
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # 6. (A10-A00)(B00+B01) -> M6
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # 7. (A01-A11)(B10+B11) -> M7
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # 8. A00 B02 -> M8
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # 9. A01 B12 -> M9
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # 10. A10 B02 -> M10
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # 11. A11 B12 -> M11
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}
