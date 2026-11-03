"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import nnls

def _verify(U, V, W, n, m, p):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build(terms, n, m, p):
    """Accumulate (u_idx, v_idx, w_idx, val) terms into U, V, W matrices."""
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u_idx, v_idx, w_idx, val) in enumerate(terms):
        U[u_idx][c] += val
        V[v_idx][c] += val
        W[w_idx][c] += val
    return U, V, W

def _strassen_block(terms, i0, j0, k0, m, p, n):
    """Emit Strassen terms for the 2x2x2 sub-block.
    Each term is (u_idx, v_idx, w_idx, val) where:
      u_idx = i*m + j  (row i, col j of A)
      v_idx = j*p + k  (row j, col k of B)
      w_idx = k*n + i  (row k, col i of C, per W[k*n+i] convention)
    The 2x2x2 block uses Strassen's 7 multiplications, but each may
    contribute multiple rank-1 terms when expanded.
    """
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # Expand: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # C00 gets all four, C11 gets all four
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+0), 1))
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+0), 1))
    terms.append(((i0+1)*m + (j0+1), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1))
    # M2 = (A10+A11)*B00 -> C10 += M2, C11 += M2
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), 1))
    terms.append(((i0+1)*m + (j0+1), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), 1))
    # M3 = A00*(B10-B00) -> C00 += M3, C01 += M3
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+0), 1))
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+0), -1))
    # M4 = A11*(B01-B11) -> C10 += M4, C11 += M4
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+0), (k0+0)*n + (i0+1), 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), -1))
    # M5 = (A00+A01)*B11 -> C01 += M5, C11 += M5
    terms.append(((i0+0)*m + (j0+0), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), 1))
    terms.append(((i0+0)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), 1))
    # M6 = (A10-A00)*(B00+B10) -> C10 += M6, C00 += M6
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), 1))
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+1), 1))
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+0), -1))
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+0), -1))
    # M7 = (A01-A11)*(B11+B01) -> C01 += M7, C11 += M7
    terms.append(((i0+0)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), 1))
    terms.append(((i0+0)*m + (j0+1), (j0+1)*p + (k0+0), (k0+0)*n + (i0+0), 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), -1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+0), (k0+0)*n + (i0+1), -1))

def _schoolbook_block(terms, i0, i1, j0, j1, k0, k1, m, p, n):
    """Emit schoolbook terms for the rectangular sub-block [i0..i1]x[j0..j1]x[k0..k1]."""
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                terms.append((i * m + j, j * p + k, k * n + i, 1))

def _try_sparse_decomposition(n, m, p, target_rank, deadline, rng):
    """
    Try to find an exact rank-target_rank decomposition using sparse linear algebra.
    Strategy: pick a random sparse support for U, V, W with target_rank columns,
    then solve the resulting bilinear system (linear in values given support).
    Returns (U, V, W) as lists of lists if found, else None.
    """
    if time.time() > deadline - 15:
        return None
    
    nm, mp, pn = n*m, m*p, p*n
    r = target_rank
    
    # We'll try multiple random sparse supports
    for attempt in range(200):
        if time.time() > deadline - 15:
            break
        
        # For each column c, pick a random (u_idx, v_idx, w_idx) triple
        # This defines a rank-1 term. We want the sum of these to equal the identity tensor.
        # The system is: for each (i,j,k): sum_c U[u_c][c]*V[v_c][c]*W[w_c][c] * [u_c==i*m+j]*[v_c==j*p+k]*[w_c==k*n+i] = 1
        # But this is only linear if we fix which (i,j,k) each column covers.
        
        # Better approach: each column c picks specific (i_c, j_c, k_c) and we set
        # U[i_c*m+j_c][c] = a_c, V[j_c*p+k_c][c] = b_c, W[k_c*n+i_c][c] = d_c
        # Then the contribution to (i,j,k) is a_c*b_c*d_c if (i_c,j_c,k_c)==(i,j,k) else 0.
        # This only covers diagonal terms. Not good enough.
        
        # Alternative: allow each column to have multiple nonzeros in U, V, W.
        # Let's try: each column c has support of size s in each of U, V, W.
        # For simplicity, try s=1 first (each column is a single rank-1 term on the diagonal).
        # That gives at most min(nm,mp,pn) useful columns, which is not enough.
        
        # Better: use the fact that we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
        # Fix U and V, then W is determined linearly (if the system is solvable).
        
        # Let's try: random U, V with sparse structure, then solve for W.
        # W[k*n+i, c] appears in equations for all j: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # For fixed (i,k), this is a system in W[k*n+i, :]:
        #   sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1 for each j
        # This is: (U_i_row * V_k_row) dot W_ki_row = 1 for each j
        # Where U_i_row has entries U[i*m+j, c] for j=0..m-1, c=0..r-1
        #       V_k_row has entries V[j*p+k, c] for j=0..m-1, c=0..r-1
        
        # Actually, for fixed (i,k), the equation for each j is:
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Let x_c = W[k*n+i, c]. Then:
        #   sum_c (U[i*m+j, c] * V[j*p+k, c]) * x_c = 1  for j = 0,1,...,m-1
        # This is a system of m equations in r unknowns x_0..x_{r-1}.
        # Coefficient matrix A_jk[i,j,c] = U[i*m+j, c] * V[j*p+k, c]
        # We need A_{i,k} @ x_{i,k} = 1 (vector of m ones)
        
        # So the plan:
        # 1. Generate random sparse U (nm x r) and V (mp x r)
        # 2. For each (i,k), solve A_{i,k} @ x = 1 for x = W[k*n+i, :]
        # 3. Check if all solutions exist and give integer/half-integer results
        # 4. Verify the full decomposition
        
        s = max(1, r // 3)  # sparsity per row
        
        U_mat = np.zeros((nm, r))
        V_mat = np.zeros((mp, r))
        
        for c in range(r):
            # Pick random nonzeros for column c of U
            u_nonzeros = rng.choice(nm, size=min(s, nm), replace=False)
            for idx in u_nonzeros:
                U_mat[idx, c] = rng.choice([-1, 1])
            
            v_nonzeros = rng.choice(mp, size=min(s, mp), replace=False)
            for idx in v_nonzeros:
                V_mat[idx, c] = rng.choice([-1, 1])
        
        # Solve for W
        W_mat = np.zeros((pn, r))
        solvable = True
        for i in range(n):
            for k in range(p):
                # Build A matrix: m x r
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U_mat[i*m+j, c] * V_mat[j*p+k, c]
                
                # Solve A @ x = ones(m)
                try:
                    # Use least squares if overdetermined or underdetermined
                    if r >= m:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check if exact
                        if np.allclose(A @ x, np.ones(m), atol=1e-9):
                            W_mat[k*n+i, :] = x
                        else:
                            solvable = False
                            break
                    else:
                        # Underdetermined: use lstsq (minimum norm)
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        if np.allclose(A @ x, np.ones(m), atol=1e-9):
                            W_mat[k*n+i, :] = x
                        else:
                            solvable = False
                            break
                except np.linalg.LinAlgError:
                    solvable = False
                    break
            
            if not solvable:
                break
        
        if not solvable:
            continue
        
        # Check if entries are integer or half-integer
        all_half_int = True
        for mat in [U_mat, V_mat, W_mat]:
            for val in mat.flatten():
                if abs(val - round(2*val)/2) > 1e-9:
                    all_half_int = False
                    break
            if not all_half_int:
                break
        
        if not all_half_int:
            continue
        
        # Convert to Fraction and verify
        from fractions import Fraction
        U_frac = [[Fraction(int(round(2*U_mat[i,c])))/2 for c in range(r)] for i in range(nm)]
        V_frac = [[Fraction(int(round(2*V_mat[i,c])))/2 for c in range(r)] for i in range(mp)]
        W_frac = [[Fraction(int(round(2*W_mat[i,c])))/2 for c in range(r)] for i in range(pn)]
        
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s_val = Fraction(0)
                    for c in range(r):
                        s_val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if s_val != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return U_frac, V_frac, W_frac
    
    return None

def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 3, 5, 6

    # ---- Schoolbook fallback (rank 90) ----
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    # ---- Strassen-based construction (target rank 84) ----
    # Partition the 3x5x6 tensor:
    #   j-blocks: {0,1}, {2,3}, {4}
    #   i-blocks: {0,1}, {2}
    #   k-blocks: {0,1}, {2,3}, {4,5}
    # Strassen (rank 7) applies where i_count=2, j_count=2, k_count=2.
    # There are 2 j-blocks x 1 i-block x 3 k-blocks = 6 Strassen blocks -> 6*7 = 42 terms.
    # Remainder (i={2}) : 1 i x 5 j x 6 k = 30 schoolbook terms.
    # Total = 42 + 30 = 72? No, let's recompute carefully.
    
    # For j in {0,1}: i in {0,1}, k in {0,1},{2,3},{4,5} -> 3 Strassen blocks (2x2x2) = 21 terms
    # For j in {2,3}: i in {0,1}, k in {0,1},{2,3},{4,5} -> 3 Strassen blocks (2x2x2) = 21 terms
    # For j in {0,1}: i in {2}, k in {0,1},{2,3},{4,5} -> 3 schoolbook blocks (1x2x2) = 12 terms
    # For j in {2,3}: i in {2}, k in {0,1},{2,3},{4,5} -> 3 schoolbook blocks (1x2x2) = 12 terms
    # For j = 4: i in {0,1,2}, k in {0,1},{2,3},{4,5} -> 3 schoolbook blocks (3x1x2) = 18 terms
    # Total = 21 + 21 + 12 + 12 + 18 = 84 terms.

    terms = []
    for j0, j1 in [(0, 1), (2, 3)]:
        for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
            _strassen_block(terms, 0, j0, k0, m, p, n)   # i0=0, i_count=2
        for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
            _schoolbook_block(terms, 2, 2, j0, j1, k0, k1, m, p, n)  # i=2
    # j = 4 (single)
    for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
        _schoolbook_block(terms, 0, 2, 4, 4, k0, k1, m, p, n)

    U, V, W = _build(terms, n, m, p)

    # Verify exactness; fall back to schoolbook if anything is off.
    if _verify(U, V, W, n, m, p) and len(terms) < rank_fb:
        # Try to reduce rank further using numerical methods
        if time.time() < deadline - 20:
            for target_rank in range(len(terms) - 1, max(60, len(terms) - 10), -1):
                result = _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline)
                if result is not None:
                    U_new, V_new, W_new = result
                    if _verify(U_new, V_new, W_new, n, m, p):
                        return {"U": U_new, "V": V_new, "W": W_new}
        
        return {"U": U, "V": V, "W": W}

    return fallback
# EVOLVE-BLOCK-END
