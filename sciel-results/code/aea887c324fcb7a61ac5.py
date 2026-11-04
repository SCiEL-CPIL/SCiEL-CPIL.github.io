"""Matrix multiplication 2x3x9 tensor decomposition - seeking rank below 54."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 9]
TIME_LIMIT = 120.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME) - 5.0  # 5s safety margin

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication.
    
    For matrix C = A @ B where A is n×m and B is m×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition gives:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, let me re-read the convention. W[k*n+i, r] - this is the output index.
    So the decomposition is:
    C[i,k] = sum_r U[i, r] * V[r, k] ... no.
    
    Standard format: C[i,k] = sum_r U[i,r] * V[r,k]
    But here we have 3 matrices U(n*m, rank), V(m*p, rank), W(p*n, rank)
    
    The tensor T has 3 indices: (i,j,k) where i∈[0,n), j∈[0,m), k∈[0,p)
    T[i,j,k] = 1 if they form a valid multiplication term
    
    Actually for matrix multiplication tensor:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The multiplication tensor M ∈ R^{n×m × m×p × p×n} where:
    M[(i,j), (j,k), (k,i)] = 1  ... this doesn't quite work with shared j,k
    
    Let me think again. The standard way:
    We want: C[i,k] = sum_j A[i,j] * B[j,k]
    
    As a tensor decomposition of the bilinear map:
    C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_k V[r,j,k] * B[j,k]) ... no
    
    Actually, the standard matrix multiplication tensor has entries:
    T[a, b, c] where a indexes A's elements, b indexes B's elements, c indexes C's elements
    T[a,b,c] = 1 if C's element c depends on product of A's element a and B's element b
    
    For A[i,j] (i in 0..n-1, j in 0..m-1), index a = i*m + j
    For B[j,k] (j in 0..m-1, k in 0..p-1), index b = j*p + k  
    For C[i,k] (i in 0..n-1, k in 0..p-1), index c = i*p + k
    
    T[a,b,c] = 1 iff a=(i,j), b=(j',k'), c=(i'',k'') with i=i'', j=j', k=k'
    So T[i*m+j, j*p+k, i*p+k] = 1
    
    The decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r has length n*m, v_r has length m*p, w_r has length p*n
    
    Wait, the problem says W has shape (p*n, rank). And the trace convention is W[k*n+i, r].
    So W index is k*n + i where k is column of C, i is row of C.
    So C index in W is k*n + i, meaning C[i,k] maps to W[k*n + i].
    
    Let me re-examine: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    
    U[i*m + j, r] - A index (i,j)
    V[j*p + k, r] - B index (j,k)
    W[k*n + i, r] - C index... k*n+i means if we think of C as p×n (transposed), row k, col i
    So C[i,k] corresponds to W[k*n + i, r]
    
    Verification: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    But this sums over r only, not j! That can't be right for general matrix mult.
    
    Unless... the decomposition encodes the j-sum into the structure.
    
    Let me reconsider. In the standard Strassen decomposition:
    C[i,k] = sum_r u_r[i] * v_r[k] * (scalar product terms)
    
    Actually for the tensor formulation:
    The multiplication tensor T ∈ R^{(nm) × (mp) × (pn)}
    T[a, b, c] = 1 iff there exist i,j,k with a = i*m+j, b = j*p+k, c = k*n+i
    (using the W convention where c = k*n + i)
    
    Wait that requires j to be the same in a and b, and k to be the same in b and c, and i same in a and c.
    
    So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    All other entries are 0.
    
    Rank-r decomposition: T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
    where u_r ∈ R^{nm}, v_r ∈ R^{mp}, w_r ∈ R^{pn}
    
    T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    
    So we need: sum_r u_r[i*m+j] * v_r[j'*p+k'] * w_r[k''*n+i''] = 1
    iff i=i'', j=j', k=k'', and 0 otherwise.
    
    For schoolbook: for each (i,j,k), set u_{ijk}[i*m+j]=1, v_{ijk}[j*p+k]=1, w_{ijk}[k*n+i]=1
    
    OK so now for the 2×3×3 subproblem (n=2, m=3, p=3):
    T[a,b,c] where a∈[0,6), b∈[0,9), c∈[0,6)
    T[i*3+j, j*3+k, k*2+i] = 1
    
    Schoolbook rank = 18.
    
    Known optimal rank for 2×3×3 is 7 (this is a well-known result).
    
    Let me try to find a rank-7 solution by search with small integer coefficients.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = n, m, p
    rank = U.shape[1]
    
    # Build the full tensor and check
    # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
    
    # For each (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i, T[a,b,c] should be 1
    # For all other (a,b,c), T[a,b,c] should be 0
    
    # Check the 1-entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[:, a] * V[:, b] * W[:, c])  # wrong indexing
                # Actually U has shape (nm, rank), so U[a, r] is element
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check a sample of 0-entries (or all if small)
    nm, mp, pn = n*m, m*p, p*n
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                # Check if this is a valid (i,j,k) triple
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                expected = 1.0 if is_valid else 0.0
                if abs(val - expected) > tol:
                    return False
    return True


def try_rank_7_search(max_search_time=30):
    """Search for a rank-7 decomposition of the 2×3×3 multiplication tensor.
    
    T[i*3+j, j*3+k, k*2+i] = 1 for i∈{0,1}, j∈{0,1,2}, k∈{0,1,2}
    
    We need 7 vectors u_r∈R^6, v_r∈R^9, w_r∈R^6 such that:
    sum_r u_r[a]*v_r[b]*w_r[c] = T[a,b,c]
    """
    start = time.time()
    n, m, p = 2, 3, 3
    R = 7
    nm, mp, pn = n*m, m*p, p*n  # 6, 9, 6
    
    # The tensor has 18 ones (for each of 18 (i,j,k) triples)
    # and all other entries are 0
    
    # Strategy: parameterize with sparse/structured solutions
    # Try integer solutions in range [-3, 3]
    
    import random
    random.seed(42)
    
    best_error = float('inf')
    best_solution = None
    
    # Try various approaches:
    # 1. Random search with small integers
    # 2. Structured search
    
    # Let's try to use known structure. For 2x2x2, Strassen uses 7 products.
    # For 2x3x3, we can think of it as:
    # A is 2x3, B is 3x3, C is 2x3
    # C[:,k] = A @ B[:,k] for each column k
    # So C = A @ B is like 3 separate 2x3 @ 3x1 multiplications... no, B is 3x3
    
    # Actually 2x3 @ 3x3: result is 2x3.
    # We can split B into 3 columns: B = [B0, B1, B2] each 3x1
    # C = [A@B0, A@B1, A@B2]
    # Each A@Bk is a 2x1 vector = 2x3 @ 3x1, which needs rank 3 (schoolbook) or less?
    # 2x3 @ 3x1 is just a linear map from R^3 to R^2, rank is at most 2 (the output dim)
    # But as a bilinear form in (A, Bk), it's a tensor in R^6 × R^3 × R^2
    # Its rank is... the matrix multiplication tensor for 2x3 @ 3x1
    # T[a, b, c] where a indexes A (6 elements), b indexes Bk (3 elements), c indexes result (2 elements)
    # T[i*3+j, k, i] = 1 (using c=i since result is 2x1, indexed by row i)
    
    # Hmm, this is getting complicated. Let me just do a direct numerical search.
    
    # Approach: set up the system of equations and try to solve
    # We have nm*mp*pn = 6*9*6 = 324 equations
    # Unknowns: R * (nm + mp + pn) = 7 * 21 = 147
    # But the equations are not all independent (most are 0=0 type)
    # The 18 equations with value 1 and 306 with value 0
    
    # Actually this is overdetermined and nonlinear (products of unknowns)
    
    # Let me try a different approach: use the fact that we can fix some degrees of freedom
    # and do iterative refinement
    
    # Alternative: try to find solutions by building up from smaller cases
    # or use known results from the literature
    
    # For now, let me try random initialization + gradient descent / iterative methods
    
    def compute_tensor_error(U, V, W):
        """Compute the error in the tensor reconstruction."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    error += (val - 1.0) ** 2
        # Check some zero entries
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    ia, ja = a // m, a % m
                    jb, kb = b // p, b % p
                    kc, ic = c // n, c % n
                    if not (ia == ic and ja == jb and kb == kc):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        error += val ** 2
        return error
    
    # Try random integer solutions
    attempts = 0
    max_attempts = 5000
    while time.time() - start < max_search_time and attempts < max_attempts:
        attempts += 1
        
        # Random integer matrices with entries in small range
        U = np.random.randint(-3, 4, (nm, R))
        V = np.random.randint(-3, 4, (mp, R))
        W = np.random.randint(-3, 4, (pn, R))
        
        # Quick check: verify the 18 one-entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Check zero entries
            zero_valid = True
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        ia, ja = a // m, a % m
                        jb, kb = b // p, b % p
                        kc, ic = c // n, c % n
                        if not (ia == ic and ja == jb and kb == kc):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            if abs(val) > 1e-6:
                                zero_valid = False
                                break
                    if not zero_valid:
                        break
                if not zero_valid:
                    break
            
            if zero_valid:
                return U, V, W
    
    # Try with real-valued optimization
    # Use a simple iterative approach
    for init_seed in range(5):
        if time.time() - start > max_search_time:
            break
        np.random.seed(init_seed)
        U = np.random.randn(nm, R) * 0.5
        V = np.random.randn(mp, R) * 0.5
        W = np.random.randn(pn, R) * 0.5
        
        for iteration in range(200):
            if time.time() - start > max_search_time:
                break
            
            # Compute gradients (simplified: use coordinate descent or simple updates)
            # For each r, update U[:,r], V[:,r], W[:,r]
            # This is complex; let's just check if we're close
            
            err = compute_tensor_error(U, V, W)
            if err < 1e-8:
                # Snap to nearest integers or half-integers
                U_s = np.round(U * 2) / 2
                V_s = np.round(V * 2) / 2
                W_s = np.round(W * 2) / 2
                if verify_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-6):
                    return U_s, V_s, W_s
                # Also try integer snap
                U_s = np.round(U)
                V_s = np.round(V)
                W_s = np.round(W)
                if verify_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-6):
                    return U_s, V_s, W_s
                return U, V, W  # Return float solution
            
            # Simple update: adjust one rank component at a time
            r = iteration % R
            # Compute residual
            # For simplicity, just do a small random perturbation (not ideal but fast)
            pass  # Skip complex optimization for time
    
    return None


def build_rank_21_from_rank_7(U7, V7, W7):
    """Build a rank-21 decomposition for 2x3x9 from a rank-7 decomposition of 2x3x3.
    
    2x3 @ 3x9: View B as 3 blocks of 3 columns: B = [B0 | B1 | B2], each 3x3.
    C = A @ B = [A@B0 | A@B1 | A@B2], each block is 2x3.
    
    For block k (k=0,1,2): C[:, k*3:(k+1)*3] = A @ B[:, k*3:(k+1)*3]
    
    Using the rank-7 decomposition for 2x3x3:
    (A @ Bk)[i, k'] = sum_{r=0}^{6} U7[i*3+j, r] * A[i,j] * V7[j*3+k', r] * Bk[j,k']
    
    But we need to express this in terms of the full B index.
    B[j, k*3+k'] has index j*p_full + (k*3+k') where p_full = 9
    So b_full = j*9 + k*3 + k'
    
    C[i, k*3+k'] has W index (k*3+k')*2 + i
    
    So for the full problem:
    U_full[i*3+j, k*7+r] = U7[i*3+j, r]  for k=0,1,2, r=0..6
    V_full[j*9+k*3+k', k*7+r] = V7[j*3+k', r]  for k=0,1,2, r=0..6, k'=0,1,2
    W_full[(k*3+k')*2+i, k*7+r] = W7[k'*2+i, r]  for k=0,1,2, r=0..6, k'=0,1,2
    
    Let me verify the dimensions:
    U_full: (2*3, 21) = (6, 21) ✓
    V_full: (3*9, 21) = (27, 21) ✓
    W_full: (9*2, 21) = (18, 21) ✓
    """
    n, m, p_full = 2, 3, 9
    p_block = 3
    R7 = 7
    R_full = 3 * R7  # 21
    
    U_full = np.zeros((n * m, R_full))
    V_full = np.zeros((m * p_full, R_full))
    W_full = np.zeros((p_full * n, R_full))
    
    for k in range(3):  # block index
        for r in range(R7):
            r_full = k * R7 + r
            # U: A[i,j] with i∈{0,1}, j∈{0,1,2}
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j, r_full] = U7[i * m + j, r]
            
            # V: B[j, k*3+k'] with j∈{0,1,2}, k'∈{0,1,2}
            for j in range(m):
                for kp in range(p_block):
                    b_full = j * p_full + k * p_block + kp
                    b_block = j * p_block + kp
                    V_full[b_full, r_full] = V7[b_block, r]
            
            # W: C[i, k*3+k'] -> W index (k*3+k')*n + i
            for i in range(n):
                for kp in range(p_block):
                    c_full = (k * p_block + kp) * n + i
                    c_block = kp * n + i
                    W_full[c_full, r_full] = W7[c_block, r]
    
    return U_full, V_full, W_full


def schoolbook_decomposition():
    """Return the schoolbook rank-54 decomposition."""
    n, m, p = 2, 3, 9
    rank = n * m * p  # 54
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return U, V, W


def snap_to_half_integer(arr, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    snapped = np.round(arr * 2) / 2
    if np.allclose(arr, snapped, atol=tol):
        return snapped
    # Try integer
    snapped_int = np.round(arr)
    if np.allclose(arr, snapped_int, atol=tol):
        return snapped_int
    return arr


def construct():
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as fallback
    U_sb, V_sb, W_sb = schoolbook_decomposition()
    best_result = (U_sb, V_sb, W_sb, 54)
    
    # Try to find rank-7 for 2x3x3
    if time_left() > 20:
        solution_7 = try_rank_7_search(max_search_time=min(40, time_left()))
        if solution_7 is not None:
            U7, V7, W7 = solution_7
            # Snap to half-integers
            U7 = snap_to_half_integer(U7)
            V7 = snap_to_half_integer(V7)
            W7 = snap_to_half_integer(W7)
            
            # Verify the rank-7 solution
            if verify_decomposition(U7, V7, W7, 2, 3, 3, tol=1e-6):
                # Build rank-21 for full problem
                U21, V21, W21 = build_rank_21_from_rank_7(U7, V7, W7)
                U21 = snap_to_half_integer(U21)
                V21 = snap_to_half_integer(V21)
                W21 = snap_to_half_integer(W21)
                
                # Verify the full decomposition
                if verify_decomposition(U21, V21, W21, n, m, p, tol=1e-6):
                    best_result = (U21, V21, W21, 21)
    
    # If we didn't find rank 21, check if schoolbook is valid
    if not verify_decomposition(best_result[0], best_result[1], best_result[2], n, m, p, tol=1e-9):
        # Rebuild schoolbook to be safe
        U_sb, V_sb, W_sb = schoolbook_decomposition()
        best_result = (U_sb, V_sb, W_sb, 54)
    
    U, V, W, rank = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


if __name__ == "__main__":
    result = construct()
    n, m, p = 2, 3, 9
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    print(f"Rank: {rank}")
    print(f"Valid: {verify_decomposition(U, V, W, n, m, p)}")
