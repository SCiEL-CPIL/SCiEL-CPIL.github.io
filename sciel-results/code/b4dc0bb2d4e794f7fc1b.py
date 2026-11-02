"""Rewritten program: tensor rank reduction via trimming/peeling on schoolbook 3x4x6."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Build the tensor T[i,j,k,r] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check against identity tensor T[i,j,k,r] = 1 if (i,j,k) matches the (i,j,k) index
    # Actually the tensor for matrix multiplication M[n,m] x N[m,p] -> P[n,p]
    # T[i, j, k, r] should equal delta(i, r) for the (j,k) contraction... 
    # Let me think more carefully.
    # The multiplication tensor T has indices (i, j, k) -> output (i, k) where:
    # P[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = A[i,j]*B[j,k]*C[i,k] ... no.
    # The tensor is T[i,j,k] such that <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j]B[j,k]
    # In factor form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And this should equal 1 when the indices are consistent, 0 otherwise.
    # Actually: T[i,j,k] = 1 for all (i,j,k) since each term A[i,j]*B[j,k] contributes to C[i,k].
    # Wait no. The tensor T satisfies: for all A, B: C = T(A,B) where C[i,k] = sum_j A[i,j]B[j,k]
    # In the trilinear form: <T, A ⊗ B ⊗ C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k]
    # For this to equal <C, C'> where C'[i,k] = sum_j A[i,j]B[j,k], we need:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C'[i,k] = sum_{i,k} C[i,k] C'[i,k] = sum_{i,j,k} A[i,j]B[j,k]C'[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    # Let me verify with the factorization:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should = 1 for all i,j,k.
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    u = U[i*m+j][r] if isinstance(U, list) else U[i*m+j, r]
                    v = V[j*p+k][r] if isinstance(V, list) else V[j*p+k, r]
                    w = W[k*n+i][r] if isinstance(W, list) else W[k*n+i, r]
                    val += Fraction(u) * Fraction(v) * Fraction(w)
                if val != Fraction(1):
                    return False
    return True

def _trim_decomposition(U, V, W, n, m, p, max_iterations=100):
    """
    Apply trimming/peeling to reduce rank.
    At each step, try to find a linear dependency among the rank-1 terms.
    A term r can be removed if its contribution can be expressed as a linear combination
    of other terms. We look for a direction in the factor spaces.
    
    The key idea: if there exists a vector (a, b, c) such that for all (i,j,k):
    sum_s a_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 0
    and a_r != 0 for some r, then we can eliminate term r by shifting the other factors.
    
    More practically: we look for a non-trivial linear relation among the "slices".
    """
    n, m, p = n, m, p
    rank = len(U[0])
    
    # Convert to numpy for linear algebra
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want to find if any rank-1 term is redundant.
    
    # Strategy: for each term r, check if the "slice" defined by that term is in the
    # span of the other terms' slices.
    # 
    # A cleaner approach: build the full tensor and check if rank decreases.
    # But that's expensive. Instead, use the following:
    #
    # If we can find vectors u (n*m dim), v (m*p dim), w (p*n dim) such that:
    # u^T U[:, r] = 0, v^T V[:, r] = 0, w^T W[:, r] = 0 for some r
    # No, that's not quite right either.
    #
    # Let me use a different approach: the "peeling" method.
    # We look for a linear combination of terms that gives zero tensor.
    # i.e., find alpha_r such that sum_r alpha_r * U[:,r] ⊗ V[:,r] ⊗ W[:,r] = 0
    #
    # This is equivalent to: for all (i,j,k), sum_r alpha_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
    #
    # This is a system of n*m*p equations in rank unknowns. If rank > n*m*p, there's always
    # a solution, but here rank = n*m*p so we need a different approach.
    #
    # Actually, the standard trimming approach for matrix multiplication:
    # We look for a term r and coefficients such that we can "absorb" term r into others.
    #
    # Let me try a more direct approach: use the fact that for matrix multiplication,
    # we can apply Strassen-like tricks on 2x2 blocks.
    
    # First, let's try Strassen on 2x2 blocks of the A matrix (rows)
    # A is n x m = 3 x 4. We can group rows into pairs.
    # For rows (i1, i2) and columns j in 0..m-1:
    # Standard: 2*m products
    # Strassen: 2*m + m = 3m products (for the 2x2x2 block per column)
    # Wait, for a 2xm block of A times mxp block of B:
    # Standard: 2*m*p products
    # With Strassen on the 2x2 (row x col) substructure per output column...
    
    # Actually, let me just implement the known Strassen reduction for the 2x2x2 case
    # applied to pairs of rows of A.
    
    # For rows i1, i2 of A (each of length m) and the full B (m x p):
    # Standard terms: for each j in 0..m-1, k in 0..p-1:
    #   A[i1,j]*B[j,k] contributes to C[i1,k]
    #   A[i2,j]*B[j,k] contributes to C[i2,k]
    # That's 2*m*p terms.
    #
    # Strassen-like: define
    #   M1 = (A[i1,:] + A[i2,:]) * B[:,k] for each k  -- m*p terms
    #   M2 = (A[i1,:] - A[i2,:]) * B[:,k] for each k  -- m*p terms  
    # But that's still 2*m*p. The saving comes from sharing.
    #
    # Actually for 2x1x1 (two inputs, one intermediate, one output per pair):
    # Standard: 2 terms: a1*b + a2*b... no that's not right.
    #
    # Let me think about this differently. The 2x2x2 Strassen algorithm:
    # C11 = (A11+A22)(B11+B22) - M3 - M4
    # C12 = A22(B12-B22) + M2 + M4  -- wait, let me get this right.
    #
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    #
    # This replaces 8 scalar multiplications with 7.
    #
    # For our 3x4x6 problem, we can apply Strassen to 2x2 blocks of the "row x col" 
    # structure. Let's apply it to pairs of rows of A.
    
    # Let me build the decomposition by applying Strassen to row pairs.
    # Rows of A: 0, 1, 2 (n=3). Pair (0,1), row 2 separate.
    # Columns of A: 0,1,2,3 (m=4).
    # Rows of B: 0,1,2,3 (m=4). Cols of B: 0,1,2,3,4,5 (p=6).
    
    # For row pair (0,1) of A and all columns/rows:
    # A[0,:] and A[1,:] are both length 4.
    # B is 4x6.
    # C[0,:] and C[1,:] are both length 6.
    #
    # Standard: 2*4*6 = 48 terms.
    # Strassen on 2x2 (row x first-col-group) ... this gets complicated.
    
    # Let me use a simpler approach: just apply the 2x2x2 Strassen block
    # to each (row-pair, col, output-col) triple.
    #
    # For rows (i1, i2), column j of A/B, output column k:
    # Standard: 2 terms: A[i1,j]*B[j,k] -> C[i1,k], A[i2,j]*B[j,k] -> C[i2,k]
    # That's already minimal (2 terms for 2 outputs).
    #
    # The Strassen saving comes when we have a 2x2 block in both A and B.
    # So we need 2 rows of A, 2 cols of A (= 2 rows of B), and 2 cols of B (= 2 rows of C).
    #
    # For rows (i1,i2) of A, cols (j1,j2) of A/B, output cols (k1,k2) of B/C:
    # Standard: 2*2*2 = 8 terms.
    # Strassen: 7 terms.
    #
    # n=3, m=4, p=6.
    # Row pairs: (0,1), (2,?) - only 1 full pair, row 2 is leftover.
    # Col pairs: (0,1), (2,3) - 2 pairs.
    # Output col pairs: (0,1), (2,3), (4,5) - 3 pairs.
    #
    # Full Strassen blocks: 1 * 2 * 3 = 6 blocks, each saving 1 term: save 6.
    # Remaining terms:
    #   Row 2 with all cols and output cols: 1 * 4 * 6 = 24
    #   Total: 72 - 6 = 66.
    #
    # Let me implement this.
    
    terms = []  # Each term is (U_entry, V_entry, W_entry, value)
    # U is indexed by (i, j) where i in 0..n-1, j in 0..m-1
    # V is indexed by (j, k) where j in 0..m-1, k in 0..p-1
    # W is indexed by (k, i) where k in 0..p-1, i in 0..n-1
    
    def add_term(i, j, k, val):
        """Add a term with U[i,j]=val, V[j,k]=val, W[k,i]=val (all same val for simplicity)."""
        terms.append(((i, j), (j, k), (k, i), val))
    
    # Process Strassen blocks for row pair (0,1)
    i1, i2 = 0, 1
    for j1 in range(0, m, 2):
        j2 = j1 + 1
        if j2 >= m:
            # Leftover column
            for k in range(p):
                add_term(i1, j1, k, 1)
                add_term(i2, j1, k, 1)
            continue
        for k1 in range(0, p, 2):
            k2 = k1 + 1
            if k2 >= p:
                # Leftover output column
                add_term(i1, j1, k1, 1)
                add_term(i1, j2, k1, 1)
                add_term(i2, j1, k1, 1)
                add_term(i2, j2, k1, 1)
                continue
            
            # Strassen block for A[i1,i2; j1,j2] * B[j1,j2; k1,k2]
            # M1 = (A[i1,j1]+A[i1,j2]+A[i2,j1]+A[i2,j2])... 
            # No, Strassen operates on 2x2 submatrices.
            # A_sub = [[A[i1,j1], A[i1,j2]], [A[i2,j1], A[i2,j2]]]
            # B_sub = [[B[j1,k1], B[j1,k2]], [B[j2,k1], B[j2,k2]]]
            # C_sub = A_sub * B_sub
            #
            # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
            # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
            # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
            # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
            # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
            # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
            # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
            #
            # C[i1,k1] = M1 + M4 - M5 - M7
            # C[i1,k2] = M3 + M5
            # C[i2,k1] = M2 + M4
            # C[i2,k2] = M1 - M2 + M3 + M7
            #
            # Each M is a product of a linear combo of A entries and a linear combo of B entries.
            # In tensor decomposition, each M corresponds to a rank-1 term where:
            # U part: the linear combo of A entries (coefficients on U[i,j] for various (i,j))
            # V part: the linear combo of B entries (coefficients on V[j,k] for various (j,k))
            # W part: distributes to C entries.
            
            # M1: U has +1 at (i1,j1) and +1 at (i2,j2); V has +1 at (j1,k1) and +1 at (j2,k2)
            #     W has +1 at (k1,i1) and +1 at (k2,i2)
            terms.append({
                'U': {(i1,j1): 1, (i2,j2): 1},
                'V': {(j1,k1): 1, (j2,k2): 1},
                'W': {(k1,i1): 1, (k2,i2): 1}
            })
            
            # M2: U has +1 at (i2,j1) and +1 at (i2,j2); V has +1 at (j1,k1)
            #     W has +1 at (k1,i2)
            terms.append({
                'U': {(i2,j1): 1, (i2,j2): 1},
                'V': {(j1,k1): 1},
                'W': {(k1,i2): 1}
            })
            
            # M3: U has +1 at (i1,j1); V has +1 at (j1,k2) and -1 at (j2,k2)
            #     W has +1 at (k2,i1)
            terms.append({
                'U': {(i1,j1): 1},
                'V': {(j1,k2): 1, (j2,k2): -1},
                'W': {(k2,i1): 1}
            })
            
            # M4: U has +1 at (i2,j2); V has +1 at (j2,k1) and -1 at (j1,k1)
            #     W has +1 at (k1,i2)
            terms.append({
                'U': {(i2,j2): 1},
                'V': {(j2,k1): 1, (j1,k1): -1},
                'W': {(k1,i2): 1}
            })
            
            # M5: U has +1 at (i1,j1) and +1 at (i1,j2); V has +1 at (j2,k2)
            #     W has -1 at (k1,i1) and +1 at (k2,i1)
            terms.append({
                'U': {(i1,j1): 1, (i1,j2): 1},
                'V': {(j2,k2): 1},
                'W': {(k1,i1): -1, (k2,i1): 1}
            })
            
            # M6: U has +1 at (i2,j1) and -1 at (i1,j1); V has +1 at (j1,k1) and +1 at (j1,k2)
            #     W has +1 at (k1,i2)
            terms.append({
                'U': {(i2,j1): 1, (i1,j1): -1},
                'V': {(j1,k1): 1, (j1,k2): 1},
                'W': {(k1,i2): 1}
            })
            
            # M7: U has +1 at (i1,j2) and -1 at (i2,j2); V has +1 at (j2,k1) and +1 at (j2,k2)
            #     W has -1 at (k1,i1) and +1 at (k2,i2)
            terms.append({
                'U': {(i1,j2): 1, (i2,j2): -1},
                'V': {(j2,k1): 1, (j2,k2): 1},
                'W': {(k1,i1): -1, (k2,i2): 1}
            })
    
    # Process remaining row (row 2)
    i_rem = 2
    for j in range(m):
        for k in range(p):
            add_term(i_rem, j, k, 1)
    
    # Now build U, V, W matrices from terms
    # Collect all unique (i,j), (j,k), (k,i) indices
    all_U = set()
    all_V = set()
    all_W = set()
    for t in terms:
        if isinstance(t, dict):
            all_U.update(t['U'].keys())
            all_V.update(t['V'].keys())
            all_W.update(t['W'].keys())
        else:
            # (i,j), (j,k), (k,i), val
            all_U.add(t[0])
            all_V.add(t[1])
            all_W.add(t[2])
    
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, t in enumerate(terms):
        if isinstance(t, dict):
            for (i, j), val in t['U'].items():
                U[i*m + j, r] = val
            for (j, k), val in t['V'].items():
                V[j*p + k, r] = val
            for (k, i), val in t['W'].items():
                W[k*n + i, r] = val
        else:
            (i, j), (j2, k), (k2, i2), val = t
            U[i*m + j, r] = val
            V[j*p + k, r] = val
            W[k*n + i, r] = val
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    else:
        # Try to trim further
        U, V, W = _attempt_trimming(U, V, W, n, m, p)
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _attempt_trimming(U, V, W, n, m, p, max_steps=50):
    """
    Attempt to reduce rank by finding and removing dependent terms.
    Uses the fact that if a term's contribution is in the span of others, we can eliminate it.
    """
    rank = U.shape[1]
    
    # Try to find a term that can be eliminated.
    # For each term r, check if there exist vectors a, b, c such that:
    # a^T U[:,s] * b^T V[:,s] * c^T W[:,s] = 0 for all s != r, and a^T U[:,r] etc. 
    # This is complex. Let's try a simpler heuristic:
    # Check if any two terms have proportional U, V, or W columns.
    
    for step in range(max_steps):
        # Check for duplicate or proportional terms
        removed = False
        for r in range(rank):
            for s in range(r+1, rank):
                # Check if term r and s are "the same" (proportional in all three factors)
                # If U[:,r] = a*U[:,s], V[:,r] = b*V[:,s], W[:,r] = c*W[:,s]
                # and a*b*c = 1, then they're redundant (can combine).
                # More generally, if the rank-1 tensors are proportional.
                
                Ur = U[:, r]
                Us = U[:, s]
                Vr = V[:, r]
                Vs = V[:, s]
                Wr = W[:, r]
                Ws = W[:, s]
                
                # Check if Ur is proportional to Us
                if np.allclose(Us, 0) or np.allclose(Ur, 0):
                    continue
                # Find ratio
                mask = np.abs(Us) > 1e-10
                if not np.any(mask):
                    continue
                ratios = Ur[mask] / Us[mask]
                if not np.allclose(ratios, ratios[0], atol=1e-10):
                    continue
                a = ratios[0]
                
                mask = np.abs(Vs) > 1e-10
                if not np.any(mask):
                    continue
                ratios = Vr[mask] / Vs[mask]
                if not np.allclose(ratios, ratios[0], atol=1e-10):
                    continue
                b = ratios[0]
                
                mask = np.abs(Ws) > 1e-10
                if not np.any(mask):
                    continue
                ratios = Wr[mask] / Ws[mask]
                if not np.allclose(ratios, ratios[0], atol=1e-10):
                    continue
                c = ratios[0]
                
                # If a*b*c == 0, one of them is zero, meaning the term is already zero
                # If a*b*c == 1, they're identical, remove one
                # If a*b*c != 1 but nonzero, we can still combine
                if abs(a*b*c - 1.0) < 1e-10:
                    # Identical terms, remove s
                    U = np.delete(U, s, axis=1)
                    V = np.delete(V, s, axis=1)
                    W = np.delete(W, s, axis=1)
                    rank -= 1
                    removed = True
                    break
                elif abs(a*b*c) > 1e-10:
                    # Proportional but not identical: term_r = (a*b*c) * term_s
                    # We can absorb: new_term_s = term_s + term_r / (a*b*c)... 
                    # Actually, term_r contributes (a*b*c) times what term_s contributes.
                    # We can just remove term_r and scale term_s... no, that changes the sum.
                    # term_r + term_s = (1 + a*b*c) * term_s (in tensor sense, if proportional)
                    # Wait, they're proportional rank-1 tensors, so their sum is also rank-1.
                    # term_r = (a*b*c) * term_s as tensors.
                    # So term_r + term_s = (1 + a*b*c) * term_s.
                    # We can replace both with a single term: (1+a*b*c) * term_s.
                    if abs(1 + a*b*c) > 1e-10:
                        scale = 1 + a*b*c
                        U[:, s] = scale * Us
                        V[:, s] = scale * Vs
                        W[:, s] = scale * Ws
                        U = np.delete(U, r, axis=1)
                        V = np.delete(V, r, axis=1)
                        W = np.delete(W, r, axis=1)
                        rank -= 1
                        removed = True
                        break
        
        if removed:
            continue
        else:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try the Strassen-based construction
    result = _trim_decomposition(None, None, None, n, m, p)
    
    # Verify the result
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        result = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return result
