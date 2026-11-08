import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) match
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Flatten T for ALS
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try ranks from 20 upwards
    # Known rank for 2x4x4 is 20
    for rank in range(20, 33):
        if time.time() > deadline - 10:
            break
            
        found = _search_decomposition(T, rank, n, m, p, deadline)
        if found is not None:
            U, V, W = found
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _search_decomposition(T, rank, n, m, p, deadline):
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Number of restarts
    num_restarts = 50
    max_iter = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            return None
            
        # Random initialization
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # ALS
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
                
            # Fix V, W -> solve U
            # T = sum U[:,c] outer V[:,c] outer W[:,c]
            # Reshape T to (dim_u, dim_v * dim_w)
            # T_reshaped = T.reshape(dim_u, -1)
            # M_vw = V.T @ W -> No, this is not quite right for ALS directly on tensors.
            # Standard ALS for tensor decomposition:
            # U = argmin || T - U * V * W ||
            # For fixed V, W, the problem is linear in U.
            # T_flat (size dim_u * dim_v * dim_w)
            # Let's use the matricization approach.
            
            # We want to minimize || T - sum_c u_c v_c^T w_c^T ||
            
            # Efficient way:
            # 1. Compute M_vw = V.T @ W? No.
            # Let's use the fact that T is small.
            
            # For each column c of U, we have a vector u_c.
            # The term is u_c (v_c outer w_c)^T
            # Let Z = V * W (element-wise product? No)
            # Z_c = v_c outer w_c (size dim_v * dim_w)
            # T = sum_c u_c outer Z_c (size dim_u * dim_v * dim_w)
            
            # Matricize T along mode 0: T_0 (size dim_u, dim_v * dim_w)
            # T_0 = U @ (V * W) where * is the Kronecker-like product of columns?
            # Actually, T_0 = sum_c u_c (v_c \otimes w_c)^T
            # Let B = [v_1 \otimes w_1, ..., v_r \otimes w_r] (size dim_v*dim_w, rank)
            # Then T_0 = U @ B.T
            
            # So U = T_0 @ B @ inv(B.T @ B)
            
            # Construct B
            # B[:, c] = V[:, c] * W[:, c] (element-wise product of vectors, flattened)
            # v_c is (dim_v,), w_c is (dim_w,)
            # v_c outer w_c is (dim_v, dim_w)
            # Flatten to (dim_v*dim_w,)
            
            B = np.zeros((dim_v * dim_w, rank))
            for c in range(rank):
                B[:, c] = (V[:, c] * W[:, c]).flatten() # Wait, V[:,c] is col c. W[:,c] is col c.
                # Element-wise product of two vectors is not outer product.
                # Outer product v_c \otimes w_c has entries v_c[i] * w_c[j].
                # Flatten of outer product is V[:,c] * W[:,c]? No.
                # If v = [a,b], w = [c,d], outer = [[ac, ad], [bc, bd]]. Flatten = [ac, ad, bc, bd].
                # np.outer(v, w).flatten()
                
            # Let's do it properly
            B = np.zeros((dim_v * dim_w, rank))
            for c in range(rank):
                outer = np.outer(V[:, c], W[:, c])
                B[:, c] = outer.flatten()
                
            T_0 = T.reshape(dim_u, -1)
            
            # Solve U = T_0 @ B @ (B.T @ B)^-1
            BtB = B.T @ B
            BtB_inv = np.linalg.pinv(BtB)
            U = T_0 @ B @ BtB_inv
            
            # Fix U, W -> solve V
            # Matricize T along mode 1: T_1 (size dim_v, dim_u * dim_w)
            # T_1 = V @ C.T
            # C[:, c] = u_c \otimes w_c
            C = np.zeros((dim_u * dim_w, rank))
            for c in range(rank):
                outer = np.outer(U[:, c], W[:, c])
                C[:, c] = outer.flatten()
                
            T_1 = np.transpose(T, (1, 0, 2)).reshape(dim_v, -1)
            CtC = C.T @ C
            CtC_inv = np.linalg.pinv(CtC)
            V = T_1 @ C @ CtC_inv
            
            # Fix U, V -> solve W
            # Matricize T along mode 2: T_2 (size dim_w, dim_u * dim_v)
            # T_2 = W @ D.T
            # D[:, c] = u_c \otimes v_c
            D = np.zeros((dim_u * dim_v, rank))
            for c in range(rank):
                outer = np.outer(U[:, c], V[:, c])
                D[:, c] = outer.flatten()
                
            T_2 = np.transpose(T, (2, 0, 1)).reshape(dim_w, -1)
            DtD = D.T @ D
            DtD_inv = np.linalg.pinv(DtD)
            W = T_2 @ D @ DtD_inv
            
            # Check convergence
            # Reconstruct T_approx
            # T_approx = sum_c U[:,c] outer V[:,c] outer W[:,c]
            # T_approx_flat = (U @ (V * W).T).flatten? No.
            # T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
            
            if it % 10 == 0:
                T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
                error = np.linalg.norm(T - T_approx)
                if error < 1e-6:
                    # Try to snap to half-integers
                    snapped_U, snapped_V, snapped_W = _snap_to_half_int(U, V, W)
                    if _verify(snapped_U, snapped_V, snapped_W, T):
                        return snapped_U, snapped_V, snapped_W

def _snap_to_half_int(U, V, W):
    # Multiply by 2, round, divide by 2
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    return U_snap, V_snap, W_snap

def _verify(U, V, W, T):
    # Check if U, V, W exactly decompose T
    # Use integer arithmetic if possible to avoid floating point issues
    # Convert to integers by multiplying by 2
    U_int = np.round(U * 2).astype(np.int64)
    V_int = np.round(V * 2).astype(np.int64)
    W_int = np.round(W * 2).astype(np.int64)
    
    # Reconstruct
    # T_approx = (1/8) * sum_c (U_int[:,c] outer V_int[:,c] outer W_int[:,c])
    # We want T_approx == T
    
    T_approx = np.zeros_like(T, dtype=np.int64)
    for c in range(U.shape[1]):
        outer_vw = np.outer(V_int[:, c], W_int[:, c])
        outer_u = U_int[:, c]
        T_approx += np.einsum('i,ij->ij', outer_u, outer_vw) # This adds to T_approx[:, :, :]? No.
        # T_approx[i,j,k] += U_int[i,c] * V_int[j,c] * W_int[k,c]
        # Using einsum: 'ic,jc,kc->ijk'
        
    # Let's just use float for verification but with strict tolerance
    T_approx_float = np.einsum('ic,jc,kc->ijk', U, V, W)
    diff = np.abs(T - T_approx_float)
    return np.all(diff < 1e-9)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
