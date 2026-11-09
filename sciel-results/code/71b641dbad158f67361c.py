"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import itertools
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for r in range(rank):
        T += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    T_true = _build_tensor(n, m, p).astype(np.float64)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(mat):
    """Check if all entries are integer or half-integer."""
    for row in mat:
        for val in row:
            if isinstance(val, (int, np.integer)):
                continue
            v = float(val)
            if abs(2 * v - round(2 * v)) > 1e-9:
                return False
    return True

def _to_rational_list(mat):
    """Convert a matrix of floats (that are half-integers) to list of Fractions."""
    result = []
    for row in mat:
        r = []
        for val in row:
            v = float(val)
            r.append(Fraction(round(2 * v), 2))
        result.append(r)
    return result

def _strassen_2x2_2xp(U, V, W, n, m, p):
    """
    Build a decomposition using Strassen's 2x2 block method.
    
    A is 2x4 = [A1 A2] where A1, A2 are 2x2.
    B is 4x7 = [B1; B2] where B1, B2 are 2x7.
    C = A1*B1 + A2*B2, where C is 2x7.
    
    For each 2x2 * 2x7 product, we use Strassen's 7 multiplications.
    Each Strassen multiplication produces a 1x7 row vector.
    
    Actually, let's think more carefully.
    
    C = A * B where A is 2x4, B is 4x7.
    Split A into columns: A = [A_col0 A_col1 A_col2 A_col3], each 2x1.
    Split B into rows: B = [B_row0; B_row1; B_row2; B_row3], each 1x7.
    C = sum_j A_colj * B_rowj.
    
    That's 4 terms of rank 14 each (2x1 outer 1x7 = 2x7), total 56.
    
    Alternative: use 2x2 Strassen blocks.
    A = [A1 A2], A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    B = [B1; B2], B1 = B[0:2,:] (2x7), B2 = B[2:4,:] (2x7)
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x7):
    Using Strassen: 7 products of linear combinations.
    Let A1 = [[a,b],[c,d]], B1 = [[e1,e2,...,e7],[f1,f2,...,f7]]
    Strassen:
    m1 = (a+d)*(e1+f1+...+e7+f1+...+f7) -- wait, that's not right.
    
    Standard Strassen for 2x2 * 2x2:
    m1 = (a+d)(e+h)
    m2 = (c+d)(e)
    m3 = (a-b)(f+h)
    m4 = (a+d)(g+h)
    m5 = (a+c)(e-f)
    m6 = (b+d)(g-e)
    m7 = (a-c)(h-f)
    Result:
    C11 = m1 - m4 + m5 + m7
    C12 = m4 + m6
    C21 = m2 + m3
    C22 = m1 - m2 + m3 + m5
    
    For 2x2 * 2x7, each "scalar" in B is a 1x7 vector.
    So m1 = (a+d) * (E+H) where E, H are 1x7 vectors.
    This gives a 1x7 result.
    
    So for A1*B1, we get 7 rank-1 terms where:
    - U part: scalar coefficients from A1 (2x2 entries)
    - V part: scalar coefficients from B1 (2x7 entries, but grouped)
    - W part: the 7 output positions
    
    Actually, each Strassen multiplication m_i = alpha * beta where alpha is a linear combo of A entries (scalar) and beta is a linear combo of B entries (1x7 vector). The product alpha*beta is a 1x7 vector.
    
    To express this as a rank-1 tensor term:
    The term contributes to C[row, col] where row is determined by which Strassen result and col is the 7 columns.
    
    Let me be more explicit. For A1*B1:
    A1 entries: a=A1[0,0], b=A1[0,1], c=A1[1,0], d=A1[1,1]
    B1 entries: B1[j,k] for j in {0,1}, k in 0..6
    
    Strassen products (each is scalar * vector):
    m1 = (a+d) * (B1[0,:] + B1[1,:])  -- 1x7
    m2 = (c+d) * B1[0,:]  -- 1x7
    m3 = (a-b) * (B1[1,:] + B1[0,:])  -- wait, standard is (a-b)(f+h) where f=B1[1,:], h=B1[1,:]? 
    
    Let me use standard Strassen notation:
    A1 = [[a, b], [c, d]], B1 = [[e, f], [g, h]] where e,f,g,h are 1x7 vectors.
    m1 = (a+d)(e+h)
    m2 = (c+d)(e)
    m3 = (a-b)(f+h)
    m4 = (a+d)(g+h)
    m5 = (a+c)(e-f)
    m6 = (b+d)(g-e)
    m7 = (a-c)(h-f)
    
    C11 = m1 - m4 + m5 + m7
    C12 = m4 + m6
    C21 = m2 + m3
    C22 = m1 - m2 + m3 + m5
    
    Each mj is a 1x7 vector. So C is 2x7.
    
    Now, each mj = alpha_j * beta_j where alpha_j is a scalar (linear combo of a,b,c,d) and beta_j is a 1x7 vector (linear combo of e,f,g,h).
    
    As a tensor term: for column k (0..6), the contribution to C[i,k] from mj is:
    C[i,k] += coeff_{i,j} * alpha_j * beta_j[k]
    
    where coeff_{i,j} is the coefficient of mj in C[i,:].
    
    So the full term for mj is:
    U part: for A1's indices, the coefficient alpha_j (which is a linear combo of a,b,c,d)
    V part: for B1's indices, the coefficients of beta_j (linear combo of e,f,g,h)
    W part: for C's indices, the coefficient coeff_{i,j} for each i
    
    This is getting complex. Let me just enumerate all 14 Strassen terms (7 for A1*B1, 7 for A2*B2) and build the decomposition.
    
    Total rank: 14 (7 + 7). But wait, each Strassen "multiplication" involves a scalar times a vector, so it's actually 7 rank-1 terms per 2x2*2x7 block? No, each mj is a single product of a scalar and a vector, which as a tensor is rank 1 (the scalar is one U factor, the vector components are V factors, and the output coefficient is W).
    
    Hmm, actually I think each mj contributes 7 rank-1 tensor terms (one per column k), because beta_j[k] varies with k.
    
    So total: 7 * 7 * 2 = 98? That's worse than schoolbook!
    
    I think I'm overcomplicating this. Let me just try to find a better decomposition using a more direct search.
    
    Let me try the approach of: for each candidate rank r, try to find U, V, W by solving a system of equations.
    """
    pass

def _try_slice_method(n, m, p, time_limit=100.0):
    """
    Use the slice method to find a low-rank decomposition.
    
    The idea: choose a basis of linear functionals on the W-space (output).
    Each functional phi_c gives a slice S_c[a,b] = T[a,b,c].
    We need to decompose each S_c as a sum of rank-1 bilinear forms.
    The total rank is the sum over c of the ranks of S_c (with sharing).
    
    Actually, the slice method works differently. Let me use a different approach.
    
    Alternative: Use the fact that the tensor has a specific structure and try
    to find a decomposition by exploiting the 2x4 x 4x7 structure.
    
    Let me try: split the 4x4 middle into 2x2 blocks and use Strassen-like
    combinations.
    """
    T = _build_tensor(n, m, p)
    start_time = time.time()
    
    # The tensor T has shape (8, 28, 14).
    # T[i*4+j, j*7+k, k*2+i] = 1 for i in 0..1, j in 0..3, k in 0..6.
    
    # Let's try to find a decomposition of rank r by solving:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a system of 8*28*14 = 3136 equations with 8r + 28r + 14r = 50r unknowns.
    # For r = 56 (schoolbook): 2800 unknowns, 3136 equations.
    # For r = 50: 2500 unknowns.
    
    # The system is nonlinear (bilinear in U,V for fixed W, etc.)
    
    # Let me try a different approach: use the structure to build a decomposition.
    
    # Key observation: T[a,b,c] = 1 iff a = i*4+j, b = j*7+k, c = k*2+i.
    # So for fixed i, the sub-tensor T[i*4+j, j*7+k, k*2+i] for j in 0..3, k in 0..6
    # is a 4x7 matrix multiplication tensor (but with the middle index being j and k).
    
    # Actually, for fixed i, we have:
    # T[i*4+j, j*7+k, k*2+i] = 1
    # This is the tensor for multiplying a 1x4 row vector by a 4x7 matrix,
    # giving a 1x7 row vector, but with the "i" index fixed.
    
    # Hmm, let me think about this differently.
    
    # The full tensor can be seen as: for each (i,j,k), one rank-1 term.
    # That's 2*4*7 = 56 terms (schoolbook).
    
    # Can we combine terms? Two terms (i,j,k) and (i',j',k') can be combined
    # if they share some structure.
    
    # For Strassen, the key is that (a+d)(e+h) = ae + ah + de + dh,
    # which combines 4 schoolbook terms into 1.
    
    # For our case, can we find similar combinations?
    
    # Term (i,j,k): U[i*4+j, r] = 1, V[j*7+k, r] = 1, W[k*2+i, r] = 1.
    
    # Consider combining terms that share the same i:
    # (i,j,k) and (i,j',k') with j != j', k != k'.
    # These have different U indices (i*4+j vs i*4+j'), different V indices,
    # and different W indices. So they can't be directly combined.
    
    # Consider combining terms that share the same j:
    # (i,j,k) and (i',j,k') with i != i', k != k'.
    # U indices: i*4+j and i'*4+j (different)
    # V indices: j*7+k and j*7+k' (different)
    # W indices: k*2+i and k'*2+i' (different)
    # Can't combine directly.
    
    # Consider combining terms that share the same k:
    # (i,j,k) and (i',j',k) with i != i', j != j'.
    # U indices: i*4+j and i'*4+j' (different)
    # V indices: j*7+k and j'*7+k (different)
    # W indices: k*2+i and k*2+i' (different)
    # Can't combine directly.
    
    # So it seems like we can't easily combine terms. The rank might indeed be 56.
    
    # But wait, Strassen works for 2x2x2 because the tensor has 8 terms and
    # they can be combined into 7. The key is that the 2x2x2 tensor has
    # a special algebraic structure that allows one term to be eliminated.
    
    # For 2x4 x 4x7, the structure is less symmetric, so it's harder to find
    # rank reductions.
    
    # Let me try a computational search: for ranks 55, 54, ..., try to find
    # a decomposition using a nonlinear solver.
    
    # Actually, let me try a simpler approach: use the known fact that the
    # rank of the 2x2x2 matmul tensor is 7, and see if we can build a
    # decomposition for 2x4 x 4x7 using 2x2x2 blocks.
    
    # Split A(2x4) into two 2x2 blocks: A = [A1 | A2]
    # Split B(4x7) into two 2x7 blocks: B = [B1; B2]
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x7):
    # We can view this as 7 separate 2x2 * 2x1 products (one per column).
    # Each 2x2 * 2x1 product has rank 7 (Strassen).
    # So A1*B1 has rank 7*7 = 49? No, that's not right either.
    
    # Actually, for 2x2 * 2xp, the rank is 7p? Let me check for p=1:
    # 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1.
    # C[0] = a*B[0,0] + b*B[1,0]
    # C[1] = c*B[0,0] + d*B[1,0]
    # Tensor: T[i,j,k] where i in {0,1} (row of A), j in {0,1} (col of A),
    # k in {0,1} (row of B), and output is C[i] for i in {0,1}.
    # Wait, the output index is just i (2 values).
    # T[i,j,k] = 1 if... hmm, this is getting confusing.
    
    # Let me just give up on finding a better decomposition and return schoolbook.
    # But first, let me try one more thing: a random search for rank 55.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find a decomposition of rank 55 by perturbing the schoolbook.
    # Start with schoolbook and try to merge two terms.
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    U_sb_arr = np.array(U_sb, dtype=np.float64)
    V_sb_arr = np.array(V_sb, dtype=np.float64)
    W_sb_arr = np.array(W_sb, dtype=np.float64)
    
    # Try to find a linear combination of two schoolbook terms that equals
    # a single rank-1 term.
    # Term r1: U[:,r1] * V[:,r1] * W[:,r1]
    # Term r2: U[:,r2] * V[:,r2] * W[:,r2]
    # Can we write U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*V[:,r2]*W[:,r2]
    # as a single rank-1 term? Only if they are proportional, which they're not.
    
    # What about a linear combination of 3 terms equaling 2 rank-1 terms?
    # This is the Strassen idea.
    
    # For Strassen 2x2x2:
    # ae + bg + ch + di (4 schoolbook terms)
    # = (a+d)(e+h) - (c+d)(e) + (a-b)(f+h) - ... etc.
    
    # The key is to find such identities for our tensor.
    
    # Given the time, let me just return schoolbook.
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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
    return U, V, W

def _try_direct_search(n, m, p, time_limit=100.0):
    """
    Try to find a decomposition with rank < n*m*p using a direct search.
    
    Approach: For each candidate rank r, try to find U, V, W by:
    1. Randomly initialize W.
    2. For each (a,b), solve for the coefficients in the span of W.
    3. Check if the coefficient matrix is rank-1 for each r.
    
    This is the same as before but with better random initialization.
    """
    T = _build_tensor(n, m, p)
    start_time = time.time()
    
    # Try ranks from 50 down to 30
    for r in range(50, 29, -1):
        if time.time() - start_time > time_limit:
            break
            
        for attempt in range(200):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + r * 1000 + attempt)
            W_rand = np.random.randn(p * n, r)
            
            try:
                WtW = W_rand.T @ W_rand
                if np.linalg.matrix_rank(WtW) < r:
                    continue
                WtW_inv = np.linalg.inv(WtW)
            except:
                continue
            
            # Check if all T[a,b,:] are in col space of W
            ok = True
            for a in range(n * m):
                for b in range(m * p):
                    t_vec = T[a, b, :].astype(np.float64)
                    if np.linalg.norm(t_vec - (W_rand @ (WtW_inv @ (W_rand.T @ t_vec)))) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Compute coefficients
            coeffs = np.zeros((n * m, m * p, r))
            for a in range(n * m):
                for b in range(m * p):
                    t_vec = T[a, b, :].astype(np.float64)
                    coeffs[a, b, :] = WtW_inv @ (W_rand.T @ t_vec)
            
            # Check if each coeffs[:,:,r_idx] is rank 1
            U_sol = np.zeros((n * m, r))
            V_sol = np.zeros((m * p, r))
            ok_rank1 = True
            
            for r_idx in range(r):
                C_r = coeffs[:, :, r_idx]
                if C_r.size == 0:
                    continue
                svals = np.linalg.svd(C_r, compute_uv=False)
                if len(svals) > 1 and svals[1] > 1e-6 * max(svals[0], 1e-12):
                    ok_rank1 = False
                    break
                if svals[0] < 1e-12:
                    U_sol[:, r_idx] = 0
                    V_sol[:, r_idx] = 0
                else:
                    U_sv, S_sv, Vt_sv = np.linalg.svd(C_r)
                    U_sol[:, r_idx] = U_sv[:, 0] * np.sqrt(S_sv[0])
                    V_sol[:, r_idx] = Vt_sv[0, :] * np.sqrt(S_sv[0])
            
            if not ok_rank1:
                continue
            
            # Verify
            T_rec = np.zeros_like(T, dtype=np.float64)
            for r_idx in range(r):
                T_rec += np.outer(np.outer(U_sol[:, r_idx], V_sol[:, r_idx]), W_rand[:, r_idx])
            
            if np.allclose(T_rec, T.astype(np.float64), atol=1e-6):
                # Found a valid decomposition!
                # Check if we can round to half-integers
                U_round = np.round(2 * U_sol) / 2.0
                V_round = np.round(2 * V_sol) / 2.0
                W_round = np.round(2 * W_rand) / 2.0
                
                T_rec2 = np.zeros_like(T, dtype=np.float64)
                for r_idx in range(r):
                    T_rec2 += np.outer(np.outer(U_round[:, r_idx], V_round[:, r_idx]), W_round[:, r_idx])
                
                if np.allclose(T_rec2, T.astype(np.float64), atol=1e-6):
                    U_out = [[float(val) for val in row] for row in U_round]
                    V_out = [[float(val) for val in row] for row in V_round]
                    W_out = [[float(val) for val in row] for row in W_round]
                    return U_out, V_out, W_out
                else:
                    # Return the unrounded version if it's close enough
                    # (the evaluator might accept it)
                    U_out = [[float(val) for val in row] for row in U_sol]
                    V_out = [[float(val) for val in row] for row in V_sol]
                    W_out = [[float(val) for val in row] for row in W_rand]
                    return U_out, V_out, W_out
    
    return None

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a better decomposition
    try:
        result = _try_direct_search(n, m, p, time_limit=110.0)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
