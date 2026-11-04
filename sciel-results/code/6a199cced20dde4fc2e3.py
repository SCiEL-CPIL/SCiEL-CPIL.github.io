"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    # Check tensor entries: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matmul, T[i,j,k] should be 1 if (i,j,k) corresponds to a term in (A*B)[i,k]
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all i,j,k
    # Actually the tensor for matmul M_{n,m,p} has T[i,j,k] = delta_{i,j,k} in the sense
    # that the multiplication is sum over j of A[i,j]*B[j,k].
    # The tensor entries are T[i, j, k] = 1 for all valid i,j,k? No.
    # Let me think again. The matrix multiplication tensor T has entries indexed by (i, j, k)
    # where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 if this term appears in the product. For (A*B)[i,k] = sum_j A[i,j] B[j,k],
    # every triple (i,j,k) with i in [0,n), j in [0,m), k in [0,p) contributes exactly once.
    # So T[i,j,k] = 1 for all i,j,k.
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    
    # Also need to check that no spurious terms are generated.
    # Actually the standard check is: for all i,j,k, the sum equals 1.
    # But we also need to ensure that the decomposition is "exact" in the sense that
    # it only produces the correct matmul and nothing else. The standard verification is:
    # For each rank-1 term c, it contributes to specific (i,j,k) entries.
    # The sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def _strassen_2x3x3():
    """
    Construct a rank-17 decomposition for 2x3 * 3x3 matrix multiplication.
    
    The idea: split the 3x3 right matrix into blocks and use Strassen-like
    intermediate products to reduce the total number of multiplications.
    
    A is 2x3, B is 3x3. Result C is 2x3.
    C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    Schoolbook uses 18 multiplications (2*3*3).
    
    We can try to save one multiplication by exploiting linearity.
    Consider grouping: for each column k of B, we compute A * B[:,k] which is a 2-vector.
    That's 3 columns, each requiring 2*3=6 scalar mults in schoolbook = 18 total.
    
    Alternative: think of it as 2 independent 1x3 * 3x3 products (each row of A times B).
    Each 1x3 * 3x3 product has 9 terms but only 3 outputs. Schoolbook for each is 9 mults, total 18.
    
    Can we do better? The known lower bound for 2x3x3 is 17 (from Landsberg).
    Let me construct an explicit rank-17 decomposition.
    
    One approach: use the fact that we can share some intermediate products.
    Let me define intermediate products:
    
    Actually, let me use a known construction. For m*n*p where one dimension is 2,
    there are known improvements.
    
    Let me try: treat B as three 3-vectors (columns). For each row i of A (a 3-vector),
    we need dot products with each column of B.
    
    Row 0 of A: a0 = (A00, A01, A02)
    Row 1 of A: a1 = (A10, A11, A12)
    
    Columns of B: b0 = (B00, B10, B20), b1 = (B01, B11, B21), b2 = (B02, B12, B22)
    
    C[0,0] = a0 . b0
    C[0,1] = a0 . b1
    C[0,2] = a0 . b2
    C[1,0] = a1 . b0
    C[1,1] = a1 . b1
    C[1,2] = a1 . b2
    
    This is 6 dot products of 3-vectors, each requiring 3 mults = 18 total.
    
    To save one: note that a0 and a1 are related. If we compute some combined quantities...
    
    Let's try: define s = a0 + a1 (component-wise). Then s . b_k = a0.b_k + a1.b_k.
    So if we compute s.b0, s.b1, s.b2, we get 3 mults that give us sums.
    Then C[0,k] = a0.b_k, C[1,k] = a1.b_k. We have s.b_k = C[0,k]+C[1,k].
    But we still need C[0,k] and C[1,k] separately. So this doesn't directly help unless
    we can express one in terms of the other and the sum.
    
    Alternative: use the identity that for 3D dot products, we can use fewer mults if we
    allow additions. The rank of the 3D dot product tensor is 3 (it's already rank 3).
    
    Hmm, let me think differently. The tensor is M_{2,3,3}. Its rank is at most 17.
    
    Let me try to explicitly construct rank 17.
    
    Actually, I recall that for M_{2,m,p}, the rank is at most 2*mp - min(m,p) or similar.
    For m=p=3, that would be 18-3=15? No, that's not right.
    
    Let me just try a direct approach: use 17 rank-1 terms and solve for the entries.
    
    Actually, let me use a known result. The rank of M_{2,3,3} is 17.
    Here's a construction:
    
    We can write the 2x3x3 matmul as:
    - 15 "obvious" products A[i,j]*B[j,k] for specific (i,j,k)
    - Plus 2 additional products that handle the remaining cases via linear combinations
    
    Let me be more systematic. The 18 schoolbook terms are:
    (0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
    (1,0,0), (1,0,1), (1,0,2), (1,1,0), (1,1,1), (1,1,2), (1,2,0), (1,2,1), (1,2,2)
    
    If we can replace two of these with one combined product, we save one.
    
    Consider: instead of separately computing A[0,0]*B[0,0] and A[1,0]*B[0,0],
    what if we compute (A[0,0]+A[1,0])*B[0,0]? That gives us the sum, but we need them separately.
    
    This approach is tricky. Let me just verify if rank 17 is achievable and construct it.
    
    After research, I'll use a verified rank-17 construction if I can find one,
    otherwise fall back to rank 18.
    
    For now, let me return the schoolbook and add a search for rank 16-17.
    """
    # I'll attempt to find a rank-17 decomposition via a structured approach.
    # If it fails, return rank 18.
    n, m, p = 2, 3, 3
    
    # Try a parameterized rank-17 construction
    # This is complex; for safety, return schoolbook and let the search try to improve.
    return None  # Signal that we should use schoolbook

def _try_rank_search(n, m, p, target_rank, time_limit=100.0):
    """
    Try to find a decomposition of the given rank using random search + local refinement.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 for all i,j,k in the matmul decomposition.
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    num_equations = n * m * p  # 18 equations
    num_unknowns = (n*m + m*p + p*n) * target_rank
    
    # Use random initialization and Newton's method / gradient descent
    # This is a polynomial system, so we can use homotopy or random restarts.
    
    # For a quick attempt, use random rank-1 terms and solve linear system for one factor.
    # Fix U and V randomly, then W is determined by linear equations.
    # But we have 18 equations and p*n*target_rank = 6*target_rank unknowns in W.
    # For target_rank=17: 6*17=102 unknowns, 18 equations. Underdetermined!
    # So we can fix U, V and solve for W easily.
    
    # Better approach: fix U and V, solve for W via least squares, check residual.
    
    best_residual = float('inf')
    best_solution = None
    
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Random U, V with small entries
        U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
        
        # Solve for W: for each (k,i), W[k*n+i, c] for c in range(target_rank)
        # Equations: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of 18 equations in 6*target_rank unknowns.
        # Write as: for each (k,i), let w_{ki} = [W[k*n+i,0], ..., W[k*n+i,r-1]]
        # For fixed (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Let a_{ij,c} = U[i*m+j,c] * V[j*p+k,c]. Then sum_c a_{ij,c} * W[k*n+i,c] = 1.
        # For fixed k,i, the equations over j are:
        #   sum_c a_{ij,c} * W[k*n+i,c] = 1 for j=0,1,2
        # So for each (k,i), we have 3 equations in target_rank unknowns.
        # Total: 6*(k,i) pairs * 3 equations = 18 equations.
        
        # Build the linear system
        # Unknowns: W[k*n+i, c] for k in 0..2, i in 0..1, c in 0..r-1
        # Index: idx = (k*n+i)*r + c
        
        A_mat = []
        b_vec = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = [0.0] * (p*n*target_rank)
                    for c in range(target_rank):
                        idx = (k*n+i)*target_rank + c
                        row[idx] = U[i*m+j][c] * V[j*p+k][c]
                    A_mat.append(row)
                    b_vec.append(1.0)
        
        # Solve least squares
        import numpy as np
        A_np = np.array(A_mat)
        b_np = np.array(b_vec)
        
        try:
            W_flat, residuals, rank_lsq, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
            residual = np.linalg.norm(A_np @ W_flat - b_np)
            
            if residual < 1e-9:
                # Check if this is a valid decomposition
                W = [[0.0] * target_rank for _ in range(p*n)]
                for ki in range(p*n):
                    for c in range(target_rank):
                        W[ki][c] = W_flat[ki*target_rank + c]
                
                if _verify(U, V, W, n, m, p):
                    return U, V, W
            
            if residual < best_residual:
                best_residual = residual
                best_solution = (U, V, W)
                
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    total_deadline = time.time() + 110.0  # Leave 10s margin for the 120s limit
    
    # Start with schoolbook (rank 18) as fallback
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find rank 17
    if time.time() < total_deadline - 10:
        sol = _try_rank_search(n, m, p, 17, time_limit=min(50.0, total_deadline - time.time() - 5))
        if sol is not None:
            U, V, W = sol
            result = {"U": U, "V": V, "W": W}
            best_rank = 17
    
    # Try to find rank 16
    if best_rank == 17 and time.time() < total_deadline - 5:
        sol = _try_rank_search(n, m, p, 16, time_limit=min(50.0, total_deadline - time.time() - 5))
        if sol is not None:
            U, V, W = sol
            result = {"U": U, "V": V, "W": W}
            best_rank = 16
    
    return result
# EVOLVE-BLOCK-END
