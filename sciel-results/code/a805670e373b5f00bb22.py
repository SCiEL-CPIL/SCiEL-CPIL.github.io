"""Matrix multiplication 3,3,8 rank minimization via randomized decomposition search."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Tensor T[i,j,k,r] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # T[i,j,k] should equal 1 if this is the (i,j,k) entry of the multiplication tensor
    # Actually the tensor for matmul: T_{i,j,k} = delta (the multiplication tensor)
    # The decomposition: T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, let me re-read the convention.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n+i, r] - this is the "official trace convention"
    # The tensor: for matmul C = A*B where A is n x m, B is m x p, C is n x p
    # T[i,j,k] = 1 (the multiplication tensor has 1s at (i,j,k) for the product)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify by checking a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                expected = 1.0
                if abs(s - expected) > tol:
                    return False
    return True

def _to_fraction_list(mat, max_denom=1000):
    """Convert numpy array to list of Fractions, snapping near-rationals."""
    result = []
    for row in mat:
        r = []
        for v in row:
            v = float(v)
            if abs(v) < 1e-12:
                r.append(Fraction(0))
            else:
                # Try to find a good rational approximation
                fr = Fraction(v).limit_denominator(max_denom)
                # Check if the approximation is good
                if abs(fr - v) < 1e-8:
                    r.append(fr)
                else:
                    r.append(Fraction(v))
        result.append(r)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    best_rank = n * m * p  # 72, schoolbook
    best_result = None
    
    # Schoolbook as initial best
    rank0 = n * m * p
    U0 = [[0]*rank0 for _ in range(nm)]
    V0 = [[0]*rank0 for _ in range(mp)]
    W0 = [[0]*rank0 for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    best_result = {"U": U0, "V": V0, "W": W0}
    best_rank = rank0
    
    # Build the multiplication tensor T of shape (n, m, p) where T[i,j,k]=1
    # We want to find rank-r decomposition:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Strategy: For a given rank r, randomly initialize U and V, then solve for W
    # using the constraint that for all (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k*n+i,c] = 1
    
    # For fixed U, V: For each (k,i) pair, we have:
    # sum_c W[k*n+i,c] * (sum over j of U[i,j,c]*V[j,k,c]) = 1
    # This gives a system: for each (k,i): sum_c W[k*n+i,c] * S[k,i,c] = 1
    # where S[k,i,c] = sum_j U[i,j,c]*V[j,k,c]
    # But we have pn = 24 equations (one per (k,i)) and r unknowns per (k,i)... 
    # Actually W has shape (pn, r), so for each (k,i) we have r unknowns W[k*n+i, 0..r-1]
    # and 1 equation. So it's underdetermined if r > 1.
    
    # Better approach: Randomly generate U (nm x r) and V (mp x r), then for each (k,i),
    # we need W[k*n+i, :] such that dot(W[k*n+i,:], S_ki) = 1 where S_ki[c] = sum_j U[i*m+j,c]*V[j*p+k,c]
    # If S_ki is nonzero, we can pick W[k*n+i,:] = S_ki / ||S_ki||^2 (minimum norm solution)
    # Then verify.
    
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Try various ranks
        # Start from lower ranks and go up
        # Known: rank of 3x3 * 3x8... let's try ranks from 20 upward
        # Actually, let's try a range of ranks
        ranks_to_try = []
        # Try some specific ranks
        for r in range(20, 73):
            if time.time() > deadline - 5:
                break
            ranks_to_try.append(r)
        
        for r in ranks_to_try:
            if r >= best_rank:
                continue
            if time.time() > deadline - 5:
                break
            
            # Multiple random trials for this rank
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Random U and V
                U_rand = np.random.randn(nm, r)
                V_rand = np.random.randn(mp, r)
                
                # For each (k, i), compute S_ki[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                # S has shape (pn, r) where S[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                S = np.zeros((pn, r))
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            S[k*n+i, :] += U_rand[i*m+j, :] * V_rand[j*p+k, :]
                
                # Now for each (k,i) row of S, we need W[k*n+i, :] such that
                # dot(W[k*n+i, :], S[k*n+i, :]) = 1
                # Minimum norm solution: W[k*n+i, :] = S[k*n+i, :] / (S[k*n+i, :] . S[k*n+i, :])
                W_rand = np.zeros((pn, r))
                valid = True
                for idx in range(pn):
                    s = S[idx, :]
                    norm_sq = np.dot(s, s)
                    if norm_sq < 1e-10:
                        valid = False
                        break
                    W_rand[idx, :] = s / norm_sq
                
                if not valid:
                    continue
                
                # Verify the decomposition
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_val = 0.0
                            for c in range(r):
                                s_val += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                            if abs(s_val - 1.0) > 1e-6:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    # Found a valid decomposition of rank r!
                    # Convert to exact fractions
                    U_frac = _to_fraction_list(U_rand)
                    V_frac = _to_fraction_list(V_rand)
                    W_frac = _to_fraction_list(W_rand)
                    
                    # Final exact verification with fractions
                    exact_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s_val = Fraction(0)
                                for c in range(r):
                                    s_val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                                if s_val != 1:
                                    exact_ok = False
                                    break
                            if not exact_ok:
                                break
                        if not exact_ok:
                            break
                    
                    if exact_ok:
                        best_rank = r
                        best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                        # Keep searching for even lower rank
                    else:
                        # Try with higher precision or different random seed
                        pass
    
    return best_result
