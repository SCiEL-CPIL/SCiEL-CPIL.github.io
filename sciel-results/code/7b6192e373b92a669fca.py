"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Precompute non-zero entries of the tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    # Otherwise 0
    non_zero_uvw = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zero_uvw.append((u, v, w))
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((dim_u, rank_sb))
    V_sb = np.zeros((dim_v, rank_sb))
    W_sb = np.zeros((dim_w, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def compute_error(U, V, W):
        """Compute error over ALL entries, not just non-zeros"""
        # We need to check sum_r U[u,r]V[v,r]W[w,r] == T[u,v,w]
        # T[u,v,w] is 1 for non_zero_uvw, 0 otherwise.
        # This is expensive if done naively for all 3600 entries.
        # But we can vectorize.
        # However, for the optimization loop, we use linear solves.
        # This function is for final verification.
        
        # Efficient check:
        # We know T is sparse.
        # Error = sum_{u,v,w} (sum_r U[u,r]V[v,r]W[w,r] - T[u,v,w])^2
        # = sum_{u,v,w in NZ} (sum_r ... - 1)^2 + sum_{u,v,w not in NZ} (sum_r ... - 0)^2
        
        # We can compute the full tensor product?
        # T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        # This creates a 12x20x15 array, which is small (3600 elements).
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Create target tensor
        T_target = np.zeros((dim_u, dim_v, dim_w))
        for u, v, w in non_zero_uvw:
            T_target[u, v, w] = 1.0
            
        return np.sum((T_recon - T_target) ** 2)

    def solve_for_W(U, V):
        """Solve for W given U, V using least squares for each w"""
        W = np.zeros((dim_w, U.shape[1]))
        R = U.shape[1]
        
        # Precompute M_w matrices?
        # M_w has shape (dim_u * dim_v, R)
        # M_w[u*dim_v + v, r] = U[u, r] * V[v, r]
        
        for w in range(dim_w):
            # Build M_w
            # We can do this efficiently
            # M_w = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
            M_w = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
            
            # Build RHS
            # RHS[u*dim_v + v] = T[u, v, w]
            # T[u, v, w] is 1 if (u,v,w) in non_zero_uvw, else 0
            # We can precompute a lookup table for T[:, :, w]
            # But doing it in the loop is fine if dim_u*dim_v is small (240)
            
            # Optimization: Precompute the structure of T
            # We already have non_zero_uvw.
            # Let's just construct the RHS vector
            rhs = np.zeros(dim_u * dim_v)
            # We need to map (u,v) to index
            # We can iterate non_zero_uvw and pick those with this w
            # This is inefficient.
            
            # Better: Precompute T_slices
            # T_slices[w] is a vector of length dim_u*dim_v
            # Let's do this outside the loop.
            
            # For now, let's just use the precomputed T_target if available?
            # No, let's precompute T_slices once.
            
            # We will pass T_slices as an argument or closure variable.
            # Let's refactor to precompute T_slices.
            
            pass # Placeholder, will be replaced

    # Precompute T_slices for efficiency
    T_slices_w = [np.zeros(dim_u * dim_v) for _ in range(dim_w)]
    T_slices_u = [np.zeros(dim_v * dim_w) for _ in range(dim_u)]
    T_slices_v = [np.zeros(dim_u * dim_w) for _ in range(dim_v)]
    
    for u, v, w in non_zero_uvw:
        T_slices_w[w][u * dim_v + v] = 1.0
        T_slices_u[u][v * dim_w + w] = 1.0
        T_slices_v[v][u * dim_w + w] = 1.0

    def solve_W_block(U, V):
        W = np.zeros((dim_w, U.shape[1]))
        R = U.shape[1]
        for w in range(dim_w):
            M_w = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
            rhs = T_slices_w[w]
            # Solve M_w x = rhs
            try:
                W[w, :] = np.linalg.lstsq(M_w, rhs, rcond=None)[0]
            except:
                pass
        return W

    def solve_U_block(V, W):
        U = np.zeros((dim_u, V.shape[1]))
        R = V.shape[1]
        for u in range(dim_u):
            # Equation: sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w]
            # Variables: U[u, 1..R]
            # Coefficients for U[u,r]: V[v,r] W[w,r]
            # Matrix M_u shape (dim_v * dim_w, R)
            # M_u[v*dim_w + w, r] = V[v,r] W[w,r]
            M_u = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
            rhs = T_slices_u[u]
            try:
                U[u, :] = np.linalg.lstsq(M_u, rhs, rcond=None)[0]
            except:
                pass
        return U

    def solve_V_block(U, W):
        V = np.zeros((dim_v, U.shape[1]))
        R = U.shape[1]
        for v in range(dim_v):
            # Equation: sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w]
            # Variables: V[v, 1..R]
            # Coefficients for V[v,r]: U[u,r] W[w,r]
            # Matrix M_v shape (dim_u * dim_w, R)
            # M_v[u*dim_w + w, r] = U[u,r] W[w,r]
            M_v = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
            rhs = T_slices_v[v]
            try:
                V[v, :] = np.linalg.lstsq(M_v, rhs, rcond=None)[0]
            except:
                pass
        return V

    # Try to find lower rank decompositions
    for rank in range(20, rank_sb + 1):
        if time.time() > deadline:
            break
        
        found = False
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 100 + attempt)
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # Block coordinate descent
            for _ in range(50):
                W = solve_W_block(U, V)
                U = solve_U_block(V, W)
                V = solve_V_block(U, W)
                
                # Check error occasionally
                if _ % 10 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-9:
                        break
            
            err = compute_error(U, V, W)
            if err < 1e-9:
                # Try rounding
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if compute_error(U_rounded, V_rounded, W_rounded) < 1e-9:
                    best_result = (U_rounded, V_rounded, W_rounded)
                    best_rank = rank
                    found = True
                    break
    
    # Convert to lists
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
