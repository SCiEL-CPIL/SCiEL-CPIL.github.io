"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_rational(arr, tol=1e-9):
    """Snap values to nearest integer or half-integer if close enough."""
    result = np.array(arr, dtype=float)
    # Try integers first
    int_snapped = np.round(result)
    if np.max(np.abs(result - int_snapped)) < tol:
        return int_snapped.astype(int).tolist()
    # Try half-integers
    half_snapped = np.round(result * 2) / 2
    if np.max(np.abs(result - half_snapped)) < tol:
        # Convert to list, using int where possible
        lst = []
        for row in half_snapped:
            r = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(int(round(v * 2)) / 2.0)
            lst.append(r)
        return lst
    # Return as-is (floats)
    return result.tolist()

def _construct_rank5_2x3x3():
    """
    Construct a rank-5 decomposition for 2x3 * 3x3 = 2x3 multiplication.
    Uses ALS with structured initialization and early stopping.
    """
    n, m, p = 2, 3, 3
    rank = 5
    nm, mp, pn = n*m, m*p, p*n
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for trial in range(80):
        seed = trial * 7 + 42
        np.random.seed(seed)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        for it in range(300):
            # Update W
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            M = np.einsum('ijc,jkc->kic', U3, V3)
            for k in range(p):
                for i in range(n):
                    vec = M[k, i, :]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-12:
                        W[k*n+i, :] = vec / norm_sq
        
            # Update U
            W3 = W.reshape(p, n, rank)
            N = np.einsum('jkc,kic->ijc', V3, W3)
            for i in range(n):
                for j in range(m):
                    vec = N[i, j, :]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-12:
                        U[i*m+j, :] = vec / norm_sq
        
            # Update V
            U3 = U.reshape(n, m, rank)
            P = np.einsum('ijc,kic->jkc', U3, W3)
            for j in range(m):
                for k in range(p):
                    vec = P[j, k, :]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-12:
                        V[j*p+k, :] = vec / norm_sq
            
            # Early stop check
            if it % 20 == 19:
                recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.max(np.abs(recon - 1.0))
                if err < 1e-10:
                    break
        
        # Check error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(recon - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W, True
    return best_U, best_V, best_W, False

def _construct_rank6_2x3x1():
    """
    Construct a rank-6 decomposition for 2x3 * 3x1 = 2x1 multiplication.
    This is essentially just the schoolbook: 6 independent products.
    C[0] = a1*b1 + a2*b2 + a3*b3
    C[1] = a4*b1 + a5*b2 + a6*b3
    """
    n, m, p = 2, 3, 1
    rank = 6
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return U, V, W, True

def _compose_decompositions(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Compose two matrix multiplication decompositions.
    First: n1 x m1 * m1 x p1 = n1 x p1 (using rank r1)
    Second: p1 x m2 * m2 x p2 = p1 x p2 (using rank r2)
    Result: n1 x m1 * m1 x (p1*p2/m1)... 
    
    Actually, this isn't quite right for our case. Let me think differently.
    
    For 2x3 * 3x7, we split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1).
    C = A*B1 + A*B2 + A*B3
    
    For A*B1: use rank-5 2x3x3 decomposition
    For A*B2: use rank-5 2x3x3 decomposition (shifted)
    For A*B3: use rank-6 2x3x1 decomposition
    
    The composition is straightforward: just concatenate the columns.
    """
    pass

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Strategy: Compose rank-5 2x3x3 and rank-6 2x3x1
    # Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A*B1 + A*B2 + A*B3
    
    # Get rank-5 decomposition for 2x3x3
    U5, V5, W5, ok5 = _construct_rank5_2x3x3()
    
    # Get rank-6 decomposition for 2x3x1
    U6, V6, W6, ok6 = _construct_rank6_2x3x1()
    
    if ok5 and ok6:
        # Compose: total rank = 5 + 5 + 6 = 16
        rank_composed = 5 + 5 + 6
        U_comp = np.zeros((nm, rank_composed))
        V_comp = np.zeros((mp, rank_composed))
        W_comp = np.zeros((pn, rank_composed))
        
        # Block 1: A * B1 where B1 is columns 0-2 of B
        # U1 = U5 (2x3 flattened to 6x5)
        # V1: for B1 (3x3), V[j*3+k, c] for k in 0..2
        # V_comp[j*7+k, c] = V5[j*3+k, c] for k in 0..2
        for j in range(3):
            for k in range(3):
                V_comp[j*7+k, 0:5] = V5[j*3+k, :]
        
        # W1 = W5 (3x2 flattened to 6x5)
        # W_comp[k*2+i, c] = W5[k*2+i, c] for k in 0..2
        for k in range(3):
            for i in range(2):
                W_comp[k*2+i, 0:5] = W5[k*2+i, :]
        
        # U_comp[:, 0:5] = U5
        U_comp[:, 0:5] = U5
        
        # Block 2: A * B2 where B2 is columns 3-5 of B
        # U2 = U5 (same A)
        U_comp[:, 5:10] = U5
        
        # V2: for B2 (3x3), columns 3-5 of B
        # V_comp[j*7+k, 5:10] = V5[j*3+(k-3), :] for k in 3..5
        for j in range(3):
            for k in range(3, 6):
                V_comp[j*7+k, 5:10] = V5[j*3+(k-3), :]
        
        # W2 = W5 (same output structure)
        for k in range(3):
            for i in range(2):
                W_comp[k*2+i, 5:10] = W5[k*2+i, :]
        
        # Block 3: A * B3 where B3 is column 6 of B
        # U3 = U6 (2x3 flattened to 6x6)
        U_comp[:, 10:16] = U6
        
        # V3: for B3 (3x1), only column 6
        # V_comp[j*7+6, 10:16] = V6[j*1+0, :] for j in 0..2
        for j in range(3):
            V_comp[j*7+6, 10:16] = V6[j*1+0, :]
        
        # W3 = W6 (1x2 flattened to 2x6)
        # W_comp[6*2+i, 10:16] = W6[0*2+i, :] for i in 0..1
        for i in range(2):
            W_comp[6*2+i, 10:16] = W6[0*2+i, :]
        
        # Verify
        if _verify_decomposition(U_comp, V_comp, W_comp, n, m, p, rank_composed, tol=1e-4):
            # Snap to rationals
            U_list = _snap_to_rational(U_comp)
            V_list = _snap_to_rational(V_comp)
            W_list = _snap_to_rational(W_comp)
            
            # Re-verify with snapped values
            if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank_composed, tol=1e-4):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank_composed
    
    # Try to improve further with ALS if time permits
    if time.time() < deadline - 30:
        # Try ranks from best_rank down to 10
        for rank in range(best_rank - 1, 9, -1):
            if time.time() > deadline - 25:
                break
            
            found = False
            for attempt in range(4):
                if time.time() > deadline - 20:
                    break
                
                # Use ALS with structured init from composition if available
                np.random.seed(rank * 137 + attempt * 31)
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
                
                max_iters = 400 if rank <= 14 else 250
                for it in range(max_iters):
                    if time.time() > deadline - 15:
                        break
                    
                    U3 = U.reshape(n, m, rank)
                    V3 = V.reshape(m, p, rank)
                    M = np.einsum('ijc,jkc->kic', U3, V3)
                    for k in range(p):
                        for i in range(n):
                            vec = M[k, i, :]
                            norm_sq = np.dot(vec, vec)
                            if norm_sq > 1e-12:
                                W[k*n+i, :] = vec / norm_sq
                    
                    W3 = W.reshape(p, n, rank)
                    N = np.einsum('jkc,kic->ijc', V3, W3)
                    for i in range(n):
                        for j in range(m):
                            vec = N[i, j, :]
                            norm_sq = np.dot(vec, vec)
                            if norm_sq > 1e-12:
                                U[i*m+j, :] = vec / norm_sq
                    
                    U3 = U.reshape(n, m, rank)
                    P = np.einsum('ijc,kic->jkc', U3, W3)
                    for j in range(m):
                        for k in range(p):
                            vec = P[j, k, :]
                            norm_sq = np.dot(vec, vec)
                            if norm_sq > 1e-12:
                                V[j*p+k, :] = vec / norm_sq
                    
                    # Early stop
                    if it % 30 == 29:
                        recon_chk = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                        err_chk = np.max(np.abs(recon_chk - 1.0))
                        if err_chk < 1e-10:
                            break
                
                # Check convergence
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.max(np.abs(recon - 1.0))
                
                if err < 1e-6:
                    U_list = _snap_to_rational(U)
                    V_list = _snap_to_rational(V)
                    W_list = _snap_to_rational(W)
                    if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank, tol=1e-4):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            
            if found:
                continue
    
    return best
