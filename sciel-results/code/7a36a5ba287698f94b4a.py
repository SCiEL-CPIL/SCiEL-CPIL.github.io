"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try to find a low-rank exact decomposition
    # The (2,3,3) tensor should have rank 12
    for target_rank in [12]:
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, target_rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank):
    """Try to find an exact rank decomposition."""
    if rank == 12:
        result = _construct_rank12(n, m, p)
        if result is not None:
            return result
    
    # For other ranks, use schoolbook if rank >= n*m*p
    if rank >= n * m * p:
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_rank12(n, m, p):
    """Try to construct a rank-12 decomposition for (2,3,3) using ALS."""
    import numpy as np
    import time
    from fractions import Fraction
    
    start = time.time()
    deadline = start + 130
    R = 12
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need U (n*m x R), V (m*p x R), W (p*n x R) such that:
    # For all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # Use structured initialization based on schoolbook with merging
    # For (2,3,3), schoolbook has 18 terms. We need to merge 6 pairs.
    # 
    # Key insight: if two terms (t1, t2) have the same W factor,
    # we can merge them if the U and V factors are proportional.
    # 
    # Let's try a different approach: use a known rank-12 construction
    # based on the fact that for (n,m,p) with n=2, we can use
    # rank = m*p + n*(m-1)*(p-1) or similar formulas.
    
    # Actually, let's try a direct construction:
    # For (2,3,3), one known rank-12 decomposition uses the following idea:
    # Split the 18 schoolbook terms into 12 groups where each group
    # corresponds to a rank-1 tensor.
    
    # Let's use a more systematic approach:
    # The matrix multiplication tensor T_{ijk} = delta_{ik} (for A_{ij} * B_{jk})
    # Wait, no. T_{i,j,k} = 1 for all i,j,k in the standard basis.
    # Actually, the multiplication tensor is: T_{(i,j),(j,k),(k,i)} = 1
    # In our indexing: U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Let me try a specific construction:
    # For each (j,k) pair, we can have a term that handles all i values
    # if we use a specific structure.
    
    # Actually, let's just try harder with ALS using better initialization.
    
    np.random.seed(42)
    
    # Try multiple random initializations with different strategies
    best_result = None
    best_res = float('inf')
    
    for seed in range(50):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        
        # Initialize with a mix of structured and random
        if seed % 3 == 0:
            # Start close to schoolbook
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            # First 12 terms: schoolbook-like
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = i * m * p + j * p + k
                        if col < R:
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
            # Add random perturbation to escape local minima
            U += np.random.randn(n*m, R) * 0.1
            V += np.random.randn(m*p, R) * 0.1
            W += np.random.randn(p*n, R) * 0.1
        elif seed % 3 == 1:
            # Random initialization
            U = np.random.randn(n*m, R) * 0.5 + 0.5
            V = np.random.randn(m*p, R) * 0.5 + 0.5
            W = np.random.randn(p*n, R) * 0.5 + 0.5
        else:
            # Structured initialization based on Kronecker products
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            # Try to use a block structure
            for t in range(R):
                i = t // (m*p)
                rem = t % (m*p)
                j = rem // p
                k = rem % p
                if i < n and j < m and k < p:
                    U[i*m+j, t] = 1.0
                    V[j*p+k, t] = 1.0
                    W[k*n+i, t] = 1.0
            # Perturb
            U += np.random.randn(n*m, R) * 0.2
            V += np.random.randn(m*p, R) * 0.2
            W += np.random.randn(p*n, R) * 0.2
        
        max_res = float('inf')
        for iteration in range(300):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    ATA = A.T @ A
                    ATb = A.T @ np.ones(p)
                    ATA += 1e-10 * np.eye(R)
                    try:
                        U[i*m+j, :] = np.linalg.solve(ATA, ATb)
                    except np.linalg.LinAlgError:
                        U[i*m+j, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    BTA = B.T @ B
                    BTb = B.T @ np.ones(n)
                    BTA += 1e-10 * np.eye(R)
                    try:
                        V[j*p+k, :] = np.linalg.solve(BTA, BTb)
                    except np.linalg.LinAlgError:
                        V[j*p+k, :] = np.linalg.lstsq(BTA, BTb, rcond=None)[0]
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    CTA = C.T @ C
                    CTb = C.T @ np.ones(m)
                    CTA += 1e-10 * np.eye(R)
                    try:
                        W[k*n+i, :] = np.linalg.solve(CTA, CTb)
                    except np.linalg.LinAlgError:
                        W[k*n+i, :] = np.linalg.lstsq(CTA, CTb, rcond=None)[0]
            
            # Check convergence
            max_res = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_res = max(max_res, abs(s - 1))
            
            if max_res < 1e-12:
                break
        
        if max_res < best_res:
            best_res = max_res
            if max_res < 1e-8:
                # Try to round to rationals
                def to_fraction_array(arr, max_denom=10000):
                    frac_arr = []
                    for val in arr.flatten():
                        best_frac = Fraction(val).limit_denominator(max_denom)
                        frac_arr.append(best_frac)
                    return frac_arr
                
                U_frac = to_fraction_array(U)
                V_frac = to_fraction_array(V)
                W_frac = to_fraction_array(W)
                
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for t in range(R):
                                s += U_frac[i*m*R + j*R + t] * V_frac[j*p*R + k*R + t] * W_frac[k*n*R + i*R + t]
                            if s != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    U_list = [[int(f) if f.denominator == 1 else float(f) for f in U_frac[i*R:(i+1)*R]] for i in range(n*m)]
                    V_list = [[int(f) if f.denominator == 1 else float(f) for f in V_frac[j*R:(j+1)*R]] for j in range(m*p)]
                    W_list = [[int(f) if f.denominator == 1 else float(f) for f in W_frac[k*R:(k+1)*R]] for k in range(p*n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
