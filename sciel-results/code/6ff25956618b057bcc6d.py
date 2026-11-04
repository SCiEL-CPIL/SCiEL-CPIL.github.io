import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 110
    
    # Target tensor for 2x3x3
    n2, m2, p2 = 2, 3, 3
    T2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
    # Precompute matricizations for 2x3x3
    T2_12 = T2.reshape(n2*m2, m2*p2*p2*n2) # Wrong reshape for ALS
    # T2[i, j, k]
    # Mat12: rows i, cols (j, k) -> index j*p2 + k
    T2_mat12 = T2.reshape(n2*m2, m2*p2 * p2*n2) # No, T2 is (6, 9, 6).
    # T2[i, j, k] corresponds to U[i], V[j], W[k].
    # T2_mat12[i, j*p2+k] = T2[i, j, k]
    T2_mat12 = np.zeros((n2*m2, m2*p2 * p2*n2)) # This is not right.
    # The second index of T is j*p2+k. The third is k*n2+i.
    # We need to map T2[i, j, k] to a matrix where row is i (or i*m2+j?)
    # In the tensor T, index 0 is i*m2+j.
    # So T2_mat12 should be of shape (n2*m2, m2*p2 * p2*n2)?
    # No, the second index of the tensor is j*p2+k.
    # The third index is k*n2+i.
    # We want to solve for U (size n2*m2 x r).
    # T2[i*m2+j, j*p2+k, k*n2+i] = sum_r U[i*m2+j, r] V[j*p2+k, r] W[k*n2+i, r]
    # This is not linear in U if we fix V and W?
    # It IS linear in U.
    # For a fixed i, j:
    # T2[i*m2+j, j*p2+k, k*n2+i] depends on k.
    # Let's group by the first index A = i*m2+j.
    # The other indices are B = j*p2+k and C = k*n2+i.
    # Note that B and C are not independent. They share j and k.
    # This coupling makes the matricization non-trivial.
    # Standard ALS assumes T[a,b,c] = sum U[a,r] V[b,r] W[c,r].
    # Here a = i*m2+j, b = j*p2+k, c = k*n2+i.
    # The indices are coupled.
    # However, we can still use ALS by treating the coupled indices as a single flattened index for the "other" two modes?
    # No, ALS updates U by fixing V and W.
    # T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r]).
    # Let M[r, (b,c)] = V[b,r] W[c,r].
    # Then T[a, (b,c)] = sum_r U[a,r] M[r, (b,c)].
    # We need to form the matrix M of size r x (m2*p2 * p2*n2).
    # But (b,c) are coupled. We only need to consider valid (b,c) pairs.
    # The valid pairs are those where b = j*p2+k and c = k*n2+i for some i,j,k.
    # For a given a = i*m2+j, the valid (b,c) pairs are determined by j (from a) and k.
    # b = j*p2+k, c = k*n2+i.
    # So for a fixed a, we have p2=3 equations.
    # This is a very sparse system.
    # We can solve for U[a, :] using these 3 equations.
    # This is much faster than full matricization.
    
    def als_step_233(T, U, V, W, r):
        # Update U
        for i in range(n2):
            for j in range(m2):
                a = i*m2 + j
                # Equations for k in 0..2
                # T[a, j*p2+k, k*n2+i] = sum_r U[a,r] V[j*p2+k, r] W[k*n2+i, r]
                # Let y_k = T[a, j*p2+k, k*n2+i]
                # Let A_k[r] = V[j*p2+k, r] * W[k*n2+i, r]
                # We want U[a, :] such that dot(U[a, :], A_k) = y_k for all k.
                # This is a system of 3 equations with r unknowns.
                # If r > 3, it's underdetermined. We want the minimum norm solution?
                # Or we can use least squares if we consider all i,j?
                # No, U[a,:] is independent for each a.
                # So for each a, we solve min ||A U[a,:]^T - y||^2.
                # A is 3 x r. y is 3 x 1.
                # Solution: U[a,:]^T = (A^T A)^-1 A^T y.
                A = np.zeros((p2, r))
                y = np.zeros(p2)
                for k in range(p2):
                    b = j*p2 + k
                    c = k*n2 + i
                    A[k, :] = V[b, :] * W[c, :]
                    y[k] = T[a, b, c]
                try:
                    # Solve A x = y
                    # x is r x 1
                    U[a, :] = np.linalg.lstsq(A, y, rcond=None)[0]
                except:
                    pass
                    
        # Update V
        for j in range(m2):
            for k in range(p2):
                b = j*p2 + k
                # Equations for i in 0..1
                # T[i*m2+j, b, k*n2+i] = sum_r U[i*m2+j, r] V[b, r] W[k*n2+i, r]
                # Let y_i = T[i*m2+j, b, k*n2+i]
                # Let A_i[r] = U[i*m2+j, r] * W[k*n2+i, r]
                # We want V[b, :] such that dot(V[b, :], A_i) = y_i for all i.
                A = np.zeros((n2, r))
                y = np.zeros(n2)
                for i in range(n2):
                    a = i*m2 + j
                    c = k*n2 + i
                    A[i, :] = U[a, :] * W[c, :]
                    y[i] = T[a, b, c]
                try:
                    V[b, :] = np.linalg.lstsq(A, y, rcond=None)[0]
                except:
                    pass
                    
        # Update W
        for k in range(p2):
            for i in range(n2):
                c = k*n2 + i
                # Equations for j in 0..2
                # T[i*m2+j, j*p2+k, c] = sum_r U[i*m2+j, r] V[j*p2+k, r] W[c, r]
                # Let y_j = T[i*m2+j, j*p2+k, c]
                # Let A_j[r] = U[i*m2+j, r] * V[j*p2+k, r]
                # We want W[c, :] such that dot(W[c, :], A_j) = y_j for all j.
                A = np.zeros((m2, r))
                y = np.zeros(m2)
                for j in range(m2):
                    a = i*m2 + j
                    b = j*p2 + k
                    A[j, :] = U[a, :] * V[b, :]
                    y[j] = T[a, b, c]
                try:
                    W[c, :] = np.linalg.lstsq(A, y, rcond=None)[0]
                except:
                    pass
                    
        return U, V, W

    def calculate_error(T, U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - T_rec)**2)

    def snap_and_verify(T, U, V, W, r):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        T_rec = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        if np.allclose(T, T_rec, atol=1e-9, rtol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    best_rank = 54 # Schoolbook
    best_U, best_V, best_W = None, None, None
    
    # Try ranks for 2x3x3
    for r_block in range(11, 19):
        if time.time() > deadline - 20:
            break
        r_full = r_block * 3
        if r_full >= best_rank:
            break
            
        found = False
        for trial in range(200):
            if time.time() > deadline - 20:
                break
            np.random.seed(42 + r_block * 1000 + trial)
            U2 = np.random.randn(n2*m2, r_block)
            V2 = np.random.randn(m2*p2, r_block)
            W2 = np.random.randn(p2*n2, r_block)
            
            for it in range(500):
                if time.time() > deadline - 20:
                    break
                U2, V2, W2 = als_step_233(T2, U2, V2, W2, r_block)
                err = calculate_error(T2, U2, V2, W2)
                if err < 1e-6:
                    snap_ok, U2_s, V2_s, W2_s = snap_and_verify(T2, U2, V2, W2, r_block)
                    if snap_ok:
                        # Build full
                        U_full = np.zeros((n*m, r_full))
                        V_full = np.zeros((m*p, r_full))
                        W_full = np.zeros((p*n, r_full))
                        
                        for b in range(3):
                            U_full[:, b*r_block:(b+1)*r_block] = U2_s
                            for j in range(m):
                                for k in range(p):
                                    if k // 3 == b:
                                        k_loc = k % 3
                                        V_full[j*p+k, b*r_block:(b+1)*r_block] = V2_s[j*3+k_loc, :]
                            for k in range(p):
                                if k // 3 == b:
                                    k_loc = k % 3
                                    for i in range(n):
                                        W_full[k*n+i, b*r_block:(b+1)*r_block] = W2_s[k_loc*2+i, :]
                                        
                        # Verify full
                        T_full = np.zeros((n*m, m*p, p*n))
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_full[i*m+j, j*p+k, k*n+i] = 1.0
                                    
                        T_rec_full = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
                        if np.allclose(T_full, T_rec_full, atol=1e-9, rtol=1e-9):
                            best_rank = r_full
                            best_U, best_V, best_W = U_full, V_full, W_full
                            found = True
                            break
                if found:
                    break
            if found:
                break
                
        if found:
            break

    if best_U is None:
        # Fallback to schoolbook
        U_sb = np.zeros((n*m, n*m*p))
        V_sb = np.zeros((m*p, n*m*p))
        W_sb = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = 54

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
