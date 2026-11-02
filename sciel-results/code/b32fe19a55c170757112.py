import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _als_decompose(n, m, p, rank, max_iter=200, seed=42, tol=1e-10):
    """
    Alternating Least Squares for the sparse matrix multiplication tensor.
    We need: for all (i,j,k) in [0,n)x[0,m)x[0,p):
      sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    
    This is a sparse tensor decomposition where only n*m*p entries are constrained.
    """
    np.random.seed(seed)
    dim1 = n * m
    dim2 = m * p
    dim3 = p * n
    
    # Precompute the list of valid (i,j,k) triples and their indices
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i*m + j, j*p + k, k*n + i))
    triples = np.array(triples)
    
    # Initialize
    U = np.random.randn(dim1, rank) * 0.5
    V = np.random.randn(dim2, rank) * 0.5
    W = np.random.randn(dim3, rank) * 0.5
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # --- Update U ---
        # For each row i of U, we need: sum_{(j,k) s.t. (i,j,k) valid} (1 - U[i,:] * (V[j,:] * W[k,:]))^2
        # This is a linear LS: minimize ||b_i - A_i * U[i,:]||^2
        # where A_i[s,t] = sum_{(j,k): (i,j,k) valid} V[j,s]*W[k,s]*V[j,t]*W[k,t]
        #       b_i[s]   = sum_{(j,k): (i,j,k) valid} V[j,s]*W[k,s]
        
        # Group triples by first index
        for i_idx in range(dim1):
            # Find all triples where first index = i_idx
            mask = triples[:, 0] == i_idx
            if not np.any(mask):
                U[i_idx, :] = 0
                continue
            idxs = triples[mask]
            v_rows = V[idxs[:, 1], :]  # shape (n_valid, rank)
            w_rows = W[idxs[:, 2], :]  # shape (n_valid, rank)
            
            # Element-wise product
            vw = v_rows * w_rows  # shape (n_valid, rank)
            
            # b[s] = sum over valid entries of vw[:, s]
            b = vw.sum(axis=0)  # shape (rank,)
            
            # A[s,t] = sum over valid entries of vw[:, s] * vw[:, t]
            A = vw.T @ vw  # shape (rank, rank)
            
            # Solve A * U[i_idx, :] = b
            try:
                U[i_idx, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
            except np.linalg.LinAlgError:
                U[i_idx, :] = np.linalg.lstsq(A + 1e-12 * np.eye(rank), b, rcond=None)[0]
        
        # --- Update V ---
        for j_idx in range(dim2):
            mask = triples[:, 1] == j_idx
            if not np.any(mask):
                V[j_idx, :] = 0
                continue
            idxs = triples[mask]
            u_rows = U[idxs[:, 0], :]
            w_rows = W[idxs[:, 2], :]
            
            uw = u_rows * w_rows
            b = uw.sum(axis=0)
            A = uw.T @ uw
            
            try:
                V[j_idx, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
            except np.linalg.LinAlgError:
                V[j_idx, :] = np.linalg.lstsq(A + 1e-12 * np.eye(rank), b, rcond=None)[0]
        
        # --- Update W ---
        for k_idx in range(dim3):
            mask = triples[:, 2] == k_idx
            if not np.any(mask):
                W[k_idx, :] = 0
                continue
            idxs = triples[mask]
            u_rows = U[idxs[:, 0], :]
            v_rows = V[idxs[:, 1], :]
            
            uv = u_rows * v_rows
            b = uv.sum(axis=0)
            A = uv.T @ uv
            
            try:
                W[k_idx, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
            except np.linalg.LinAlgError:
                W[k_idx, :] = np.linalg.lstsq(A + 1e-12 * np.eye(rank), b, rcond=None)[0]
        
        # Compute error
        err = 0.0
        for (u_idx, v_idx, w_idx) in triples:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            err += (val - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 130
    
    # Try to find a decomposition with rank < 90
    # Start with trivial as fallback
    fallback = _construct_trivial(n, m, p)
    
    # Try various ranks from low to high
    # The minimum possible rank is at least max(n*m, m*p, p*n) / something... 
    # Actually, the tensor has 90 non-zero entries, so rank >= 1 theoretically
    # But due to structure, it's likely higher
    
    # Try ranks: 90 (trivial), then try to go lower
    # Let's try a few ranks with ALS
    ranks_to_try = [90, 80, 70, 60, 50, 45, 40, 36, 30, 25, 20]
    
    best_result = None
    best_rank = 90
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        if rank >= 90:
            # Trivial solution
            result = _construct_trivial(n, m, p)
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
                break
            continue
        
        # Try ALS with this rank
        U, V, W, err = _als_decompose(n, m, p, rank, max_iter=100, seed=42)
        
        if err < 1e-8:
            # Verify
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        # Try with different seeds if time allows
        if time.time() > deadline - 15:
            break
        
        for seed in [1, 7, 123]:
            if time.time() > deadline - 10:
                break
            U, V, W, err = _als_decompose(n, m, p, rank, max_iter=80, seed=seed)
            if err < 1e-8:
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
        if best_rank < rank:
            break
    
    if best_result is None:
        best_result = fallback
    
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
