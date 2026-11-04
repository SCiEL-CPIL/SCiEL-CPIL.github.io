"""Matrix multiplication 2x5x5 tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best = (rank_sb, U_sb, V_sb, W_sb)
    
    def to_fraction_matrix(M, tol=1e-6):
        """Convert float matrix to fractions (half-integers) if close."""
        result = np.zeros_like(M, dtype=object)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if abs(v) < tol:
                    result[i, j] = Fraction(0)
                else:
                    # Try to find closest half-integer
                    v2 = v * 2
                    r = round(v2)
                    if abs(v2 - r) < tol * 2:
                        result[i, j] = Fraction(r, 2)
                    else:
                        # Try integer
                        r = round(v)
                        if abs(v - r) < tol:
                            result[i, j] = Fraction(r)
                        else:
                            result[i, j] = Fraction(v).limit_denominator(1000)
        return result
    
    def verify_decomp(U, V, W, tol=1e-4):
        """Check if U, V, W satisfy the tensor equation."""
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_c U[i*m+j... ] 
        # Tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should equal delta (i.e., T[i,j,k] = 1 if this is the identity pattern)
        # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T[i,j,k] has T[i,j,k] = 1 for all valid i,j,k
        # Decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Wait, the convention is W[k*n+i, c]
        # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal 1 for all i,j,k
        
        rank = U.shape[1]
        # Compute for a sample of (i,j,k) to verify
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        errors.append((i,j,k,val))
        return len(errors) == 0, errors
    
    def try_rank(r, max_iters=2000, seed=42):
        """Try to find a rank-r decomposition using iterative methods."""
        if r >= best[0]:
            return None
        rng = np.random.RandomState(seed)
        
        # Initialize with random sparse matrices
        for attempt in range(3):
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Random initialization: each column gets a few non-zeros
            for c in range(r):
                # Pick random positions
                ui = rng.randint(0, nm)
                vj = rng.randint(0, mp)
                wk = rng.randint(0, pn)
                U[ui, c] = rng.choice([1, -1])
                V[vj, c] = rng.choice([1, -1])
                W[wk, c] = rng.choice([1, -1])
            
            # Iterative refinement: for each (i,j,k) where residual is large,
            # try to adjust a column
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute residuals for all (i,j,k)
                residuals = np.zeros((n, m, p))
                for c in range(r):
                    uc = U[:, c]
                    vc = V[:, c]
                    wc = W[:, c]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residuals[i,j,k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
                
                # Check if close enough
                max_err = np.max(np.abs(residuals - 1.0))
                if max_err < 1e-4:
                    # Try to convert to exact
                    Uf = to_fraction_matrix(U)
                    Vf = to_fraction_matrix(V)
                    Wf = to_fraction_matrix(W)
                    # Verify with fractions
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = Fraction(0)
                                for c in range(r):
                                    val += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                                if val != Fraction(1):
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        return (r, U, V, W)
                
                # Find worst residual
                idx = np.unravel_index(np.argmax(np.abs(residuals - 1.0)), (n,m,p))
                i, j, k = idx
                err = residuals[i,j,k] - 1.0
                if abs(err) < 1e-6:
                    break
                    
                # Try to fix by modifying one column
                # Find column c that maximizes |U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]|
                contribs = np.zeros(r)
                for c in range(r):
                    contribs[c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                c_best = int(np.argmax(np.abs(contribs)))
                # Flip sign of one factor in this column to reduce error
                # We want to change the contribution by -err
                # Simple heuristic: flip the sign of W[k*n+i, c_best]
                U[i*m+j, c_best] *= -1
                # This changes contribution by -2*old_contrib
                # Not perfect but may help
                
                # Alternative: try flipping U or V
                # Let's try a different approach: zero out and reassign
                # Actually, let's use a simpler approach: random restarts with structured init
            
        return None
    
    # Try structured approaches first
    # Approach 1: Use known small-rank constructions
    # For 2x5x5, try ranks from 49 down
    
    # Better approach: Use the fact that we can combine terms
    # Start with schoolbook and try to merge columns
    
    # Approach: Randomized search with constraint satisfaction
    for target_rank in range(49, max(20, 49 - 30), -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(target_rank, max_iters=500, seed=target_rank*7+1)
        if result is not None:
            r, U, V, W = result
            if r < best[0]:
                best = (r, U, V, W)
                break
    
    # Also try a different strategy: build from scratch with more structure
    if time.time() < deadline - 15:
        # Try using the structure: for 2x5x5, we can think of it as
        # C = A @ B where A is 2x5, B is 5x5
        # C[i,k] = sum_j A[i,j] B[j,k]
        # We need T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        
        # Try a more systematic search
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(seed * 1000 + 7)
            r = 45 + seed  # Try various ranks
            if r >= best[0]:
                continue
                
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Initialize with random ±1 patterns
            for c in range(r):
                # Each column: random sparse vector
                nz_u = rng.randint(1, 4)
                nz_v = rng.randint(1, 4)
                nz_w = rng.randint(1, 3)
                idx_u = rng.choice(nm, size=nz_u, replace=False)
                idx_v = rng.choice(mp, size=nz_v, replace=False)
                idx_w = rng.choice(pn, size=nz_w, replace=False)
                for idx in idx_u:
                    U[idx, c] = rng.choice([1, -1])
                for idx in idx_v:
                    V[idx, c] = rng.choice([1, -1])
                for idx in idx_w:
                    W[idx, c] = rng.choice([1, -1])
            
            # Verify
            ok, _ = verify_decomp(U, V, W, tol=1e-3)
            if ok:
                best = (r, U, V, W)
                break
    
    r, U, V, W = best
    
    # Convert to lists
    U_list = [[float(U[i,j]) for j in range(r)] for i in range(nm)]
    V_list = [[float(V[i,j]) for j in range(r)] for i in range(mp)]
    W_list = [[float(W[i,j]) for j in range(r)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
