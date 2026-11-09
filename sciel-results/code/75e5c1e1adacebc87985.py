import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_222():
    alpha = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    beta = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]
    return alpha, beta, recon

def _rank11_223():
    alpha = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    beta = [
        [(0, 1), (4, 1)],
        [(0, 1)],
        [(1, 1), (4, -1)],
        [(3, 1), (0, -1)],
        [(4, 1)],
        [(0, 1), (1, 1)],
        [(3, 1), (4, 1)],
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]
    
    alpha.append([(0, 1)])
    beta.append([(2, 1)])
    recon.append([(0, 2, 1)])
    
    alpha.append([(1, 1)])
    beta.append([(5, 1)])
    recon.append([(0, 2, 1)])
    
    alpha.append([(2, 1)])
    beta.append([(2, 1)])
    recon.append([(1, 2, 1)])
    
    alpha.append([(3, 1)])
    beta.append([(5, 1)])
    recon.append([(1, 2, 1)])
    
    return alpha, beta, recon

def _build():
    n, m, p = 4, 4, 6
    outer_alpha, outer_beta, outer_recon = _rank11_223()
    inner_alpha, inner_beta, inner_recon = _strassen_222()
    
    R_outer = len(outer_alpha)
    R_inner = len(inner_alpha)
    R = R_outer * R_inner
    
    Umat = np.zeros((16, R))
    Vmat = np.zeros((24, R))
    Wmat = np.zeros((24, R))
    
    for so in range(R_outer):
        for si in range(R_inner):
            col = so * R_inner + si
            
            for (block_o, coeff_o) in outer_alpha[so]:
                ar, ac = block_o // 2, block_o % 2
                for (block_i, coeff_i) in inner_alpha[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gi = ar * 2 + ir
                    gk = ac * 2 + ic
                    Umat[gi * 4 + gk, col] += coeff_o * coeff_i
            
            for (block_o, coeff_o) in outer_beta[so]:
                br, bc = block_o // 3, block_o % 3
                for (block_i, coeff_i) in inner_beta[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gk = br * 2 + ir
                    gj = bc * 2 + ic
                    Vmat[gk * 6 + gj, col] += coeff_o * coeff_i
            
            for (out_r, out_c, coeff_o) in outer_recon[so]:
                for (in_r, in_c, coeff_i) in inner_recon[si]:
                    gi = out_r * 2 + in_r
                    gj = out_c * 2 + in_c
                    Wmat[gj * 4 + gi, col] += coeff_o * coeff_i
    
    return Umat, Vmat, Wmat

def _verify(U, V, W):
    n, m, p = 4, 4, 6
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ikr,kjr,jir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-6)

def _reduce_rank(U, V, W):
    R = U.shape[1]
    if R <= 1:
        return U, V, W
    
    # Flatten tensors to check linear independence
    # T_r is a vector of length n*m*p
    # T_r[i, k, j] = U[i,k,r] * V[k,j,r] * W[j,i,r]
    n, m, p = 4, 4, 6
    dim = n * m * p
    T_vectors = np.zeros((R, dim))
    
    for r in range(R):
        U_r = U[:, r]
        V_r = V[:, r]
        W_r = W[:, r]
        
        # Compute T_r
        # T_r is the tensor defined by U_r, V_r, W_r
        # We can compute it by einsum
        U3 = U_r.reshape(n, m)
        V3 = V_r.reshape(m, p)
        W3 = W_r.reshape(p, n)
        T_r = np.einsum('ik,jk->ijk', U3, V3) # Wait, W is not used in this product?
        # The tensor is sum_r U_r V_r W_r
        # Each term is a rank-1 tensor? No.
        # The term is U_r \otimes V_r \otimes W_r?
        # T_{ijk} = sum_r U_{ir} V_{kr} W_{jr} ? No.
        # T_{ijk} = sum_r U_{ikr} V_{kjr} W_{jir}
        # So for a fixed r, the contribution is U_{ikr} V_{kjr} W_{jir}
        
        T_r_flat = np.zeros((n, m, p))
        for i in range(n):
            for k in range(m):
                u_val = U[i*m+k, r]
                if u_val == 0: continue
                for j in range(p):
                    v_val = V[k*p+j, r]
                    if v_val == 0: continue
                    w_val = W[j*n+i, r]
                    if w_val == 0: continue
                    T_r_flat[i, k, j] += u_val * v_val * w_val
        
        T_vectors[r] = T_r_flat.flatten()
    
    # Check rank of T_vectors
    # If rank < R, we can reduce
    # We want to find a basis for the column space of T_vectors
    # Then express each T_r in terms of the basis
    # This gives a new decomposition with rank = rank(T_vectors)
    
    # Use SVD to find basis
    U_svd, s, Vt = np.linalg.svd(T_vectors, full_matrices=False)
    rank = np.sum(s > 1e-9)
    
    if rank >= R:
        return U, V, W
    
    # The basis vectors are the first 'rank' columns of U_svd
    # But we need to reconstruct U, V, W.
    # This is not straightforward because the decomposition is not unique.
    # However, if the T_vectors are linearly dependent, there exists a lower rank decomposition.
    # We can try to find it by solving a linear system.
    
    # For now, return the original if we can't easily reconstruct.
    # A better approach: use the fact that we can remove dependent columns.
    
    return U, V, W

def construct():
    deadline = time.time() + 110.0
    U, V, W = _build()
    
    if not _verify(U, V, W):
        R = 4 * 4 * 6
        U = np.zeros((16, R))
        V = np.zeros((24, R))
        W = np.zeros((24, R))
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(6):
                    U[i*4 + k, idx] = 1
                    V[k*6 + j, idx] = 1
                    W[j*4 + i, idx] = 1
                    idx += 1
    
    if time.time() < deadline - 5:
        U, V, W = _reduce_rank(U, V, W)
        if not _verify(U, V, W):
            U, V, W = _build()
    
    def to_list(M):
        out = []
        for row in M:
            r = []
            for x in row:
                rx = round(x)
                r.append(int(rx) if abs(x - rx) < 1e-9 else float(x))
            out.append(r)
        return out
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
