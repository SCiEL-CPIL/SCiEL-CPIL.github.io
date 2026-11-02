"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

def _build_tensor_matrix(n, m, p):
    """
    Constructs the matrix M such that a rank-r decomposition exists iff 
    the flattened tensor t is in the column space of M.
    M has shape (n*m*p, (n*m*r) * (m*p*r) * (p*n*r)) -- too big to form explicitly.
    Instead, we use the property that the tensor T is a 3-way array.
    
    Actually, a better way:
    We are looking for U (nm x r), V (mp x r), W (pn x r).
    The condition is T = sum_{l=1}^r u_l v_l^T w_l^T (outer product).
    This is equivalent to saying that for all i,j,k:
    T[i,j,k] = sum_l U[i,l] V[j,l] W[k,l]
    
    This is a system of nonlinear equations.
    However, we can fix U and V, and solve for W linearly? No, it's still coupled.
    
    Alternative strategy:
    For a fixed rank r, the set of decomposable tensors is an algebraic variety.
    We can use the "border rank" or numerical rank determination.
    
    Let's try a different structural approach:
    The tensor for 3x5x5 multiplication.
    n=3, m=5, p=5.
    
    Is there a known decomposition better than 75?
    For 2x2x2, Strassen is 7 (vs 8).
    For 3x3x3, best known is 23 (vs 27).
    For 3x5x5, it's less standard.
    
    Let's try a "randomized linear algebra" approach to find a valid decomposition of rank r.
    We can formulate this as:
    Find U, V, W such that for all (i,j,k), <u_i, v_j, w_k> = T[i,j,k].
    
    If we fix U and V, the equations for W are linear?
    T[i,j,k] = sum_l U[i,l] V[j,l] W[k,l]
    Let A_{ij, l} = U[i,l] V[j,l].
    Then T[i,j,k] = sum_l A_{ij,l} W[k,l].
    This is a linear system in the entries of W, provided U and V are fixed.
    
    So the algorithm is:
    1. Iterate r from lower bound to 75.
    2. Generate random U (nm x r) and V (mp x r) with rational/integer entries? 
       Random floats are fine for finding existence, but we need exact integer/half-integer output.
       However, if a decomposition exists with rank r, it might require algebraic numbers.
       The problem asks for "integer/half-integer entries". This is a strong constraint.
       Wait, "exactly decompose... with integer/half-integer entries".
       This implies we must find a decomposition with those specific entry types.
       
    If we find a real-valued decomposition, we can't just round it.
    However, the schoolbook is integer.
    If a lower rank decomposition exists, does it necessarily have integer/half-integer entries?
    Not necessarily. But often in these small cases, if the rank is low, the structure might be simple.
    
    Let's refine the search:
    We will search for U, V, W with entries in {0, 1, -1, 0.5, -0.5, 2, -2} etc.
    This is a discrete search, which is hard.
    
    Alternative:
    Use the fact that the tensor is sparse and structured.
    T[i,j,k] is 1 if indices match, 0 otherwise.
    
    Let's try the "fix U and V, solve for W" approach with random rational U, V.
    If we pick U, V with small integers, the resulting W might be rational.
    If W turns out to be half-integers, we have a winner.
    
    Lower bound for rank:
    For n x m x p, rank >= max(n*m, m*p, p*n) is not quite right.
    Rank >= max(n, m, p) * something?
    Actually, rank >= max(n*m, m*p, p*n) is false.
    Rank >= max(n, m, p) is true.
    For 3x5x5, rank >= 5.
    A better lower bound is the "slice rank" or similar.
    But let's just iterate r from 1 up to 75.
    
    Optimization:
    For a given r, we can try many random U, V.
    For each (U, V), we solve for W.
    The system for W:
    For each (i, j), we have a vector equation for k:
    T[i,j,:] = sum_l (U[i,l] V[j,l]) W[:,l]
    Let C_{ij} be the vector T[i,j,:] (length p).
    Let D_{ij,l} = U[i,l] V[j,l].
    Then C_{ij} = sum_l D_{ij,l} W_l, where W_l is the l-th column of W (length p*n? No, W is p*n x r).
    Wait, W[k,l] is the entry.
    The equation is: T[i,j,k] = sum_l U[i,l] V[j,l] W[k,l].
    Here k ranges over p*n? No, k in T[i,j,k] corresponds to the output index.
    In the flattened tensor T is of size (n*m) x (m*p) x (p*n).
    Let indices be a in [0, nm), b in [0, mp), c in [0, pn).
    T[a,b,c] = 1 if a,b,c are consistent, else 0.
    
    Consistency:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    Note that k in b is the same as k in c?
    Let's check:
    b = j*p + k_b.
    c = k_c*n + i_c.
    Consistency requires:
    i_a = i_c
    j_a = j_b
    k_b = k_c
    
    So T[a,b,c] = 1 iff i_a=i_c, j_a=j_b, k_b=k_c.
    
    So for fixed a, b, the vector T[a,b,:] has a 1 at position c = k_b*n + i_a, and 0 elsewhere.
    Wait, c is the index into W.
    W has shape (p*n, r).
    So for fixed a, b, we have:
    T[a,b,c] = 1 for c = c*(a,b), 0 otherwise.
    Equation:
    sum_l U[a,l] V[b,l] W[c,l] = 1  (for c = c*(a,b))
    sum_l U[a,l] V[b,l] W[c,l] = 0  (for c != c*(a,b))
    
    This is a linear system in W.
    Number of unknowns: p*n*r.
    Number of equations: n*m * (p*n) ? No, for each a,b we have p*n equations.
    Total equations: n*m * p*n.
    For 3,5,5: 15 * 25 = 375 equations.
    Unknowns: 25 * r.
    
    If r is small, the system is overdetermined.
    We can solve it using least squares and check if the residual is 0.
    
    Algorithm:
    1. For r in range(1, 76):
       2. For attempt in range(100):
          3. Generate random U (15 x r) and V (25 x r) with entries from a small set, e.g. [-1, 0, 1].
          4. Construct the linear system A W_flat = b.
             A is (n*m*p*n) x (p*n*r).
             b is (n*m*p*n).
          5. Solve for W_flat.
          6. Check if residual is small.
          7. Check if W entries are "nice" (close to integers or half-integers).
          8. If yes, verify exact decomposition with these rounded values.
             If verified, return.
    
    This is a structural change: instead of pruning a known solution, we are solving for a solution from scratch in a lower-dimensional space, using the linearity in W given U,V.
    """
    pass # Placeholder, logic in construct

def _is_nice_number(x, tol=1e-6):
    """Check if x is close to an integer or half-integer."""
    nearest_int = round(x)
    nearest_half = round(x * 2) / 2.0
    return abs(x - nearest_int) < tol or abs(x - nearest_half) < tol

def _find_decomposition_for_rank(n, m, p, r, max_attempts=50, seed=0):
    """
    Try to find a rank-r decomposition by fixing U, V randomly and solving for W.
    """
    np.random.seed(seed)
    
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Precompute the tensor structure
    # T[a, b, c] = 1 if consistent, 0 else.
    # We need to build the linear system for W.
    # Equation for each (a, b, c):
    # sum_l U[a,l] V[b,l] W[c,l] = T[a,b,c]
    
    # Let's vectorize the construction of A and b.
    # A is a big matrix.
    # Rows correspond to (a, b, c).
    # Cols correspond to (c, l).
    
    # Instead of building the full A (which might be large), we can iterate.
    # But for 3x5x5, size is manageable.
    # Rows: 15 * 25 * 25 = 9375.
    # Cols: 25 * r.
    # If r=10, cols=250.
    # 9375 x 250 matrix is fine.
    
    A = np.zeros((dim_u * dim_v * dim_w, dim_w * r))
    b = np.zeros(dim_u * dim_v * dim_w)
    
    idx = 0
    for a in range(dim_u):
        i = a // m
        j = a % m
        for b_idx in range(dim_v):
            j_b = b_idx // p
            k_b = b_idx % p
            # Consistency check for a, b
            # a determines i, j. b determines j_b, k_b.
            # We need j == j_b for any non-zero T.
            # Actually, T[a,b,c] is 0 unless j == j_b.
            # If j != j_b, then T[a,b,c] = 0 for all c.
            # So we can skip adding equations or add zero equations.
            # Adding zero equations: sum_l U[a,l]V[b,l]W[c,l] = 0.
            
            for c in range(dim_w):
                k_c = c // n
                i_c = c % n
                
                # Determine T[a,b,c]
                if j == j_b and i == i_c and k_b == k_c:
                    t_val = 1.0
                else:
                    t_val = 0.0
                
                b[idx] = t_val
                
                # Fill A[idx, :]
                # A[idx, c*r + l] = U[a, l] * V[b_idx, l]
                for l in range(r):
                    A[idx, c * r + l] = 0.0 # Will be filled when U,V are known?
                    # Wait, U and V are random and change per attempt.
                    # So we cannot precompute A fully independent of U,V.
                    # We must compute A for each U,V pair.
                    # This is expensive.
                
                idx += 1
    
    # Optimization:
    # A depends on U and V.
    # A[idx, c*r+l] = U[a,l] V[b,l]
    # We can compute this on the fly or precompute the structure.
    # Let's restructure the loop to be inside the attempt loop.
    
    for attempt in range(max_attempts):
        # Generate random U, V with small integer entries
        # Using {-1, 0, 1} to keep numbers small
        U = np.random.choice([-1, 0, 1], size=(dim_u, r))
        V = np.random.choice([-1, 0, 1], size=(dim_v, r))
        
        # Build A and b
        # b is constant, precomputed?
        # Yes, b depends only on tensor structure.
        if not hasattr(_find_decomposition_for_rank, 'b_cache'):
             _find_decomposition_for_rank.b_cache = {}
        if (n,m,p) not in _find_decomposition_for_rank.b_cache:
             b_full = np.zeros(dim_u * dim_v * dim_w)
             idx = 0
             for a in range(dim_u):
                 i = a // m
                 j = a % m
                 for b_idx in range(dim_v):
                     j_b = b_idx // p
                     k_b = b_idx % p
                     for c in range(dim_w):
                         k_c = c // n
                         i_c = c % n
                         if j == j_b and i == i_c and k_b == k_c:
                             b_full[idx] = 1.0
                         idx += 1
             _find_decomposition_for_rank.b_cache[(n,m,p)] = b_full
        b = _find_decomposition_for_rank.b_cache[(n,m,p)]
        
        # Construct A
        # A[idx, c*r+l] = U[a,l] * V[b_idx,l]
        # We can do this efficiently using broadcasting if we reshape.
        # But the mapping (a, b_idx, c) -> idx is sequential.
        
        # Let's just build it in a loop, it's small enough (9375 * 250 ops ~ 2M ops per attempt)
        # 50 attempts * 2M = 100M ops. Might be slow in Python.
        # Let's try to vectorize A construction.
        
        # A is shape (N_eq, dim_w * r)
        # N_eq = dim_u * dim_v * dim_w
        
        # We can create indices for a, b_idx, c
        # a_idx, b_idx, c_idx
        # Then U[a_idx, l] * V[b_idx, l]
        
        # Let's try a faster way:
        # For each l, A[:, l::r] corresponds to U[a,l]*V[b,l]
        # The column index for (c, l) is c*r + l.
        # So A[:, l::r] is not contiguous.
        
        # Alternative:
        # A is a block matrix?
        # Let's just use numpy operations on the full arrays.
        
        # Create index arrays
        a_rep = np.repeat(np.arange(dim_u), dim_v * dim_w)
        b_rep = np.tile(np.repeat(np.arange(dim_v), dim_w), dim_u)
        c_rep = np.tile(np.arange(dim_w), dim_u * dim_v)
        
        # We need U[a_rep, l] * V[b_rep, l] for each l.
        # And place it at column c_rep * r + l.
        
        A = np.zeros((dim_u * dim_v * dim_w, dim_w * r))
        
        for l in range(r):
            vals = U[a_rep, l] * V[b_rep, l]
            cols = c_rep * r + l
            A[np.arange(len(a_rep)), cols] = vals
            
        # Solve A W_flat = b
        # Use lstsq
        try:
            W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check residual
        if residuals.size > 0 and residuals[0] > 1e-6:
            continue
            
        # Reshape W_flat to W (dim_w, r)
        W = W_flat.reshape(dim_w, r).T # Wait, W_flat index is c*r+l, so W[c,l]
        W = W_flat.reshape(dim_w, r)
        
        # Check if W entries are nice
        nice = True
        for val in W:
            if not _is_nice_number(val):
                nice = False
                break
                
        if not nice:
            continue
            
        # Round W to nearest nice number
        W_rounded = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                v = W[i,j]
                ni = round(v)
                nh = round(v*2)/2.0
                if abs(v-ni) < abs(v-nh):
                    W_rounded[i,j] = ni
                else:
                    W_rounded[i,j] = nh
                    
        # Verify exact decomposition with U, V, W_rounded
        # T[a,b,c] == sum_l U[a,l] V[b,l] W[c,l]
        valid = True
        for a in range(dim_u):
            for b_idx in range(dim_v):
                for c in range(dim_w):
                    s = 0.0
                    for l in range(r):
                        s += U[a,l] * V[b_idx,l] * W_rounded[c,l]
                    # Compare with b
                    if abs(s - b[a*dim_v*dim_w + b_idx*dim_w + c]) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return U, V, W_rounded
            
    return None

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Lower bound for rank
    # Simple bound: max(n, m, p)
    # Better bound: ceil(n*m*p / max(n,m,p)) ? No.
    # For 3x5x5, let's start from a reasonable guess.
    # Schoolbook is 75.
    # Let's try from 1 up to 75.
    # But we can skip small ranks if they are impossible.
    # Rank of 3x3x3 is 23.
    # Rank of 3x5x5 is likely > 23.
    # Let's start from 30 to save time?
    # No, let's be systematic but fast.
    
    # If we find one, return it.
    # If not, return schoolbook.
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Build schoolbook first as fallback
    U_s = [[0] * best_rank for _ in range(n*m)]
    V_s = [[0] * best_rank for _ in range(m*p)]
    W_s = [[0] * best_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
                
    # Search for lower rank
    # Start from a lower bound.
    # The rank of the multiplication tensor is at least the "slice rank".
    # For n x m x p, slice rank >= max(n, m, p).
    # Let's try ranks from 1 to 75.
    # To speed up, we can try larger r first if we think it's hard?
    # No, we want minimal rank.
    
    # However, solving for small r is less likely to succeed with random U,V.
    # Let's try r from 1 to 75.
    
    # Optimization: Use a time budget.
    # If we spend too much time, return schoolbook.
    
    # Let's try a range of r.
    # If we assume the answer is close to schoolbook, we might miss it.
    # But the prompt asks for a DIFFERENT strategy.
    # The linear algebra approach is different.
    
    # Let's try r from 1 to 40 first (more likely to be "interesting" if it exists)
    # then 40 to 75.
    
    for r in range(1, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Increase attempts for smaller r?
        # Small r means underdetermined or just determined.
        # Equations: 375. Unknowns: 25*r.
        # If 25*r < 375, i.e., r < 15, the system is overdetermined.
        # Random U,V will almost certainly fail.
        # So for r < 15, we need many attempts or smarter U,V.
        # For r >= 15, it might be easier.
        
        # Let's use 100 attempts for r < 20, and 50 for r >= 20.
        attempts = 100 if r < 20 else 50
        
        # Add a small seed offset
        res = _find_decomposition_for_rank(n, m, p, r, max_attempts=attempts, seed=r)
        
        if res is not None:
            U, V, W = res
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break # Found minimal rank? 
            # Note: We are iterating r upwards, so the first found is minimal.
            # But is it guaranteed minimal?
            # If a rank r-1 decomposition exists, we would have found it earlier.
            # Unless our random search failed to find it.
            # So "first found" is not necessarily "true minimal".
            # But it's a valid decomposition of rank r.
            # We can continue to see if we can find lower?
            # No, we are iterating 1, 2, 3...
            # So if we find r=10, we return.
            # If we miss r=5, we might find r=10.
            # To be safe, we could check r=1..r_found-1 again?
            # Given time constraints, we accept the first found.
            
    if best_U is None:
        # Return schoolbook
        return {"U": U_s, "V": V_s, "W": W_s}
    else:
        # Convert to lists
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
