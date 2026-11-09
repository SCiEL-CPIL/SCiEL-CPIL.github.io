"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    
    The schoolbook decomposition has rank n*m*p = 60.
    We attempt to find a lower rank decomposition using gradient descent
    on the tensor reconstruction loss, followed by a rounding check to
    half-integers. If no valid lower rank is found within the time budget,
    we fall back to the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    # Build the target tensor T
    # T[u, v, w] = 1 if there exists (i, j, k) such that:
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W

    def check_half_integer(U, V, W):
        """
        Checks if rounding U, V, W to nearest 0.5 yields an exact decomposition.
        Returns rounded matrices if valid, else None.
        """
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Reconstruct tensor: T_rec[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        T_rec = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
        
        if np.allclose(T_rec, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    def optimize_rank(rank, seed=42):
        """
        Attempts to find a rank-`rank` decomposition via gradient descent.
        Returns (U, V, W) if a valid half-integer solution is found, else None.
        """
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        
        # Initialize with small random values
        scale = 0.1
        U = np.random.randn(dim_u, rank) * scale
        V = np.random.randn(dim_v, rank) * scale
        W = np.random.randn(dim_w, rank) * scale
        
        lr = 0.01
        steps = 500
        
        for step in range(steps):
            if time.time() > deadline:
                return None
                
            # Calculate Reconstruction
            Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
            
            # Error
            Err = T - Rec
            
            # Gradient
            # dL/dU = 2 * sum_{v,w} Err[u,v,w] V[v,r] W[w,r]
            # Using einsum: 'uvw,vr,wr->ur'
            try:
                G_U = np.einsum('uvw,vr,wr->ur', Err, V, W)
                G_V = np.einsum('uvw,ur,wr->vr', Err, U, W)
                G_W = np.einsum('uvw,ur,vr->wr', Err, U, V)
            except Exception:
                return None

            # Update
            U -= lr * G_U
            V -= lr * G_V
            W -= lr * G_W
            
            # Periodic check for convergence to half-integers
            if step % 50 == 0:
                result = check_half_integer(U, V, W)
                if result is not None:
                    return result

        # Final check
        return check_half_integer(U, V, W)

    # Try to find lower rank decompositions
    # We try ranks from 59 down to 50.
    # In practice, rank 59 might be hard to find with random init, but we try.
    # If we find one, we return it.
    
    # Note: The problem asks for rank-minimal. It is unknown if < 60 is possible
    # with half-integer coefficients for 2x3x10.
    # However, we will attempt a search.
    
    best_result = None
    best_rank = n * m * p
    
    # Try a few ranks
    for rank in range(59, 49, -1):
        if time.time() > deadline:
            break
            
        found = None
        for restart in range(5):
            if time.time() > deadline:
                break
            seed = 42 + restart
            res = optimize_rank(rank, seed=seed)
            if res is not None:
                found = res
                break
                
        if found:
            U, V, W = found
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to Schoolbook
    U, V, W = get_schoolbook()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
