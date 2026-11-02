import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Build the tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Pre-compute flattened views of T for ALS
    # T_flat_w: shape (nm*mp, pn). Col c is T[:,:,c] flattened.
    T_flat_w = T.reshape(nm * mp, pn)
    # T_flat_u: shape (mp*pn, nm). Col a is T[a,:,:] flattened.
    # T[a, b, c] -> We need to group by a.
    # T.reshape(nm, mp*pn) -> Row a is T[a,:,:] flattened.
    # Transpose to get (mp*pn, nm).
    T_flat_u = T.reshape(nm, mp * pn).T
    # T_flat_v: shape (pn*nm, mp). Col b is T[:,b,:] flattened.
    # T.reshape(mp, pn*nm) -> Row b is T[:,b,:] flattened?
    # T is (nm, mp, pn).
    # To get T[:, b, :] for all b, we can transpose T to (mp, nm, pn), then reshape to (mp, nm*pn), then transpose to (nm*pn, mp).
    T_trans_v = np.transpose(T, (1, 0, 2)) # (mp, nm, pn)
    T_flat_v = T_trans_v.reshape(mp, pn * nm).T # (pn*nm, mp)

    # Try to find lower rank decompositions
    # We try ranks from schoolbook_rank down to a lower bound.
    # For 3x5x7, schoolbook is 105.
    # We will try a few ranks.
    
    # Helper to verify and convert
    def verify_and_convert(U, V, W, rank):
        T_computed = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_computed += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        err = np.linalg.norm(T_computed - T)
        if err > 1e-8:
            return None
        
        # Try to round to simple rationals (integers or half-integers)
        # If they are already close to integers, round them.
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        
        # Check if rounding to integers works
        T_verify_int = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_verify_int += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
        if np.linalg.norm(T_verify_int - T) < 1e-10:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Try half-integers
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        
        T_verify_half = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_verify_half += np.einsum('a,b,c->abc', U_h[:,r], V_h[:,r], W_h[:,r])
        if np.linalg.norm(T_verify_half - T) < 1e-10:
            # Convert to list, handling .5
            def to_list(mat):
                res = []
                for row in mat:
                    r_list = []
                    for val in row:
                        if abs(val - round(val)) < 1e-10:
                            r_list.append(int(round(val)))
                        else:
                            r_list.append(val)
                    res.append(r_list)
                return res
            return {"U": to_list(U_h), "V": to_list(V_h), "W": to_list(W_h)}
            
        return None

    # ALS function
    def try_als(rank, seed):
        np.random.seed(seed)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        for iteration in range(50): # Reduced iterations for speed
            if time.time() > deadline - 5:
                return None
            
            # Solve for W
            # A_w[:, r] = U[:, r] * V[:, r]
            A_w = np.zeros((nm * mp, rank))
            for r in range(rank):
                A_w[:, r] = (U[:, r] * V[:, r]).reshape(-1)
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(A_w, T_flat_w, rcond=None)
                W = W_new.T
            except:
                return None
            
            # Solve for U
            # A_u[:, r] = V[:, r] * W[:, r]
            A_u = np.zeros((mp * pn, rank))
            for r in range(rank):
                A_u[:, r] = (V[:, r] * W[:, r]).reshape(-1)
            
            try:
                U_new, _, _, _ = np.linalg.lstsq(A_u, T_flat_u, rcond=None)
                U = U_new.T
            except:
                return None
            
            # Solve for V
            # A_v[:, r] = W[:, r] * U[:, r]
            A_v = np.zeros((pn * nm, rank))
            for r in range(rank):
                A_v[:, r] = (W[:, r] * U[:, r]).reshape(-1)
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(A_v, T_flat_v, rcond=None)
                V = V_new.T
            except:
                return None
            
            # Check error
            T_computed = np.zeros((nm, mp, pn))
            for r in range(rank):
                T_computed += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err = np.linalg.norm(T_computed - T)
            if err < 1e-9:
                return (U, V, W)
                
        return None

    # Search for lower ranks
    # We know schoolbook is 105.
    # Let's try a few ranks below 105.
    # If we find one, we return it.
    # Given the time limit, we can't search exhaustively.
    # We will try ranks: 100, 95, 90...
    
    ranks_to_try = list(range(100, 80, -5))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        for trial in range(2):
            if time.time() > deadline - 5:
                break
            result = try_als(rank, rank * 10 + trial)
            if result:
                U, V, W = result
                converted = verify_and_convert(U, V, W, rank)
                if converted:
                    return converted
                    # Note: We return immediately if we find a valid lower rank.
                    # This is greedy but acceptable given the constraints.
                    
    return best_result
