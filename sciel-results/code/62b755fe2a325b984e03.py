"""Program to find rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import numpy as np
from itertools import product

def construct():
    n, m, p = 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Compute lower bounds from matrix unfoldings
    T_u1 = T.reshape(nm, mp * pn)
    T_u2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_u3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    lb1 = np.linalg.matrix_rank(T_u1)
    lb2 = np.linalg.matrix_rank(T_u2)
    lb3 = np.linalg.matrix_rank(T_u3)
    lower_bound = max(lb1, lb2, lb3)
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.int64)
    V_school = np.zeros((mp, rank_school), dtype=np.int64)
    W_school = np.zeros((pn, rank_school), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # For each candidate rank, try to find U, V, W such that sum_r U[:,r] V[r,:] W[r,:] = T (in tensor sense)
    # This is equivalent to: for all a,b,c: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # We can fix V and solve for U and W, or use iterative methods
    # For exact integer/half-integer solutions, we need a different approach
    
    # Try small ranks from lower_bound up to schoolbook
    for rank in range(lower_bound, rank_school + 1):
        # Try to find a decomposition of this rank
        # Use a greedy/iterative approach: pick rank-1 terms that best approximate the residual
        # Start with T, subtract rank-1 tensors greedily
        
        residual = T.astype(np.float64).copy()
        U_list = []
        V_list = []
        W_list = []
        
        for r in range(rank):
            # Find the best rank-1 tensor to subtract
            # A rank-1 tensor is u ⊗ v ⊗ w where u, v, w are vectors
            # We want to minimize ||residual - (1-alpha) * u⊗v⊗w||
            # For simplicity, try to find u,v,w that maximize alignment with residual
            
            # Use SVD-like approach on unfoldings
            # Unfold residual along mode 1: shape (nm, mp*pn)
            res_u1 = residual.reshape(nm, mp * pn)
            # Get top left singular vector
            try:
                U_svd, S, Vt_svd = np.linalg.svd(res_u1, full_matrices=False)
                u_vec = U_svd[:, 0]
                # Reshape Vt_svd[0] back to (mp, pn)
                vw = Vt_svd[0].reshape(mp, pn)
                # Now decompose vw into v and w using SVD
                U_svd2, S2, Vt_svd2 = np.linalg.svd(vw, full_matrices=False)
                v_vec = U_svd2[:, 0]
                w_vec = Vt_svd2[0]
            except:
                break
            
            # Scale to best fit: find alpha that minimizes ||residual - alpha * u⊗v◗w||
            # alpha = <residual, u⊗v⊗w> / ||u⊗v⊗w||^2
            uvw = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
            numerator = np.sum(residual * uvw)
            denominator = np.sum(uvw * uvw)
            if denominator < 1e-12:
                break
            alpha = numerator / denominator
            
            # Subtract
            residual = residual - alpha * uvw
            
            # Check if residual is close to zero
            if np.max(np.abs(residual)) < 1e-9:
                # Found exact decomposition
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
                if len(U_list) <= best_rank:
                    best_rank = len(U_list)
                    best_U = np.array(U_list).T
                    best_V = np.array(V_list).T
                    best_W = np.array(W_list).T
                break
            else:
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
        
        if len(U_list) < best_rank and np.max(np.abs(residual)) < 1e-9:
            best_rank = len(U_list)
            best_U = np.array(U_list).T
            best_V = np.array(V_list).T
            best_W = np.array(W_list).T
            break
    
    # Try to round to integer/half-integer if close
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    U_out = round_to_half(best_U)
    V_out = round_to_half(best_V)
    W_out = round_to_half(best_W)
    
    # Verify the decomposition
    def verify(U, V, W, T_target):
        n, m, p = 4, 4, 9
        nm, mp, pn = n*m, m*p, p*n
        # Compute sum_r U[:,r] V[r,:] W[r,:] in tensor form
        result = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.max(np.abs(result - T_target.astype(np.float64))) < 1e-6
    
    if not verify(U_out, V_out, W_out, T):
        # Fall back to schoolbook
        U_out = U_school
        V_out = V_school
        W_out = W_school
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
