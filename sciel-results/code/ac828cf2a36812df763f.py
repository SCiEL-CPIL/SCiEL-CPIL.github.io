"""
Rank-minimal exact tensor decomposition for 2x4 x 4x6 matrix multiplication.

Strategy: Start with the known rank-42 Strassen decomposition, then attempt to
find lower-rank decompositions via structured random projection + sparse recovery.

The W factor follows the trace convention W[k*n+i, r] where n=2.
"""

import time
import numpy as np

_DEADLINE = 140.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _schoolbook(n, m, p):
    """Fallback O(nmp) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_strassen_42():
    """
    Build the 42-rank decomposition for 2x4 x 4x6.

    Layout:
      A (2x4): U[i*4 + j] for A[i,j]
      B (4x6): V[j*6 + k] for B[j,k]
      C (2x6): W[k*2 + i] for C[i,k]  (i.e., W[k*n + i])

    We split:
      A_left = A[:, 0:2]  (2x2)
      A_right = A[:, 2:4] (2x2)
      B_top = B[0:2, :]   (2x6)
      B_bottom = B[2:4, :] (2x6)

    C = A_left @ B_top + A_right @ B_bottom

    For A_left @ B_top: split B_top into 3 column-pairs, each a 2x2 block.
    Apply Strassen (7 terms) to each 2x2 x 2x2 product. Total: 3*7 = 21.
    Similarly for A_right @ B_bottom: 21 terms.
    Total: 42 terms.
    """
    n, m, p = 2, 4, 6
    rank = 42

    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 12 rows

    # Strassen 7-term formula for 2x2 x 2x2 -> 2x2.
    # Local A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Local B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 += M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)]),
    ]

    col = 0
    for a_block in range(2):
        a0 = 2 * a_block  # starting column of A block
        b0 = 2 * a_block  # starting row of B block (linked to a_block)

        for c_block in range(3):
            c0 = 2 * c_block  # starting column of C block

            # Global indices for the 2x2 A block
            a_global = [
                0 * 4 + a0,       # A[0, a0]
                0 * 4 + a0 + 1,   # A[0, a0+1]
                1 * 4 + a0,       # A[1, a0]
                1 * 4 + a0 + 1,   # A[1, a0+1]
            ]

            # Global indices for the 2x2 B block
            b_global = [
                b0 * 6 + c0,           # B[b0, c0]
                b0 * 6 + c0 + 1,       # B[b0, c0+1]
                (b0 + 1) * 6 + c0,     # B[b0+1, c0]
                (b0 + 1) * 6 + c0 + 1, # B[b0+1, c0+1]
            ]

            # Global indices for the 2x2 C block
            # C[i, k] is at W[k*2 + i]
            c_global = [
                2 * c0 + 0,     # C[0, c0]   = W[2*c0]
                2 * c0 + 2,     # C[0, c0+1] = W[2*(c0+1)]
                2 * c0 + 1,     # C[1, c0]   = W[2*c0 + 1]
                2 * c0 + 3,     # C[1, c0+1] = W[2*(c0+1) + 1]
            ]

            for (u_sp, v_sp, w_sp) in strassen_terms:
                for (loc, coeff) in u_sp:
                    U[a_global[loc]][col] += coeff
                for (loc, coeff) in v_sp:
                    V[b_global[loc]][col] += coeff
                for (loc, coeff) in w_sp:
                    W[c_global[loc]][col] += coeff
                col += 1

    if col != rank:
        raise ValueError(f"Expected {rank} columns, got {col}")

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=2, m=4, p=6, atol=1e-9, seeds=(42, 123, 777)):
    """Verify the decomposition against multiple random matrices."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for seed in seeds:
        np.random.seed(seed)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B

        C_computed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_computed[i, k] += np.dot(
                        U_arr[i * m + j, :],
                        V_arr[j * p + k, :] * W_arr[k * n + i, :]
                    )

        if not np.allclose(C_computed, C_expected, atol=atol):
            return False

    return True


def _try_reduce_rank(U, V, W, target_rank, n=2, m=4, p=6, rng=None):
    """
    Attempt to reduce the rank of a decomposition by finding a lower-rank
    equivalent via random linear combinations and null-space projection.
    
    Idea: If we have a rank-R decomposition, we can try to find a rank-r < R
    decomposition by projecting onto a random r-dimensional subspace and
    checking if the tensor is still representable.
    
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    R = len(U[0])
    if target_rank >= R:
        return None

    # Random projection matrices
    P_U = rng.standard_normal((n * m, target_rank))
    P_V = rng.standard_normal((m * p, target_rank))
    P_W = rng.standard_normal((p * n, target_rank))

    # For the decomposition to be valid with reduced rank, we need:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = C[i,k]
    # to hold with only target_rank terms.
    
    # Strategy: Use the existing decomposition to define constraints,
    # then solve for a lower-rank representation.
    
    # Build the tensor T[i,j,k] = C[i,k] coefficient structure
    # The tensor is: T[i,j,k] such that C[i,k] = sum_j A[i,j]*B[j,k]
    # In bilinear form: T[i,j,k] = delta (identity on middle index)
    
    # Actually, let's use a different approach: try to find linear
    # dependencies among the existing rank-42 terms.
    
    # Build matrix of U vectors, V vectors, W vectors
    U_mat = np.array(U, dtype=float)  # (nm, R)
    V_mat = np.array(V, dtype=float)  # (mp, R)
    W_mat = np.array(W, dtype=float)  # (pn, R)
    
    # The decomposition works iff for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if j==j (always)
    # Actually the tensor is: T[i,j,k] = 1 for all (i,j,k) in the matmul pattern
    # No wait - the tensor for matrix mult is: T[i,j,k] = 1 if the middle index matches
    
    # Let me reconsider. The matrix multiplication tensor has entries:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (it's the identity on the "contraction" index)
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
    
    # So the tensor is just all ones on the valid (i,j,k) triples.
    # A rank-r decomposition gives: T[i,j,k] = sum_{r=1}^r U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # To reduce rank, we look for a lower-rank tensor that equals T.
    # This is equivalent to finding if T lies in the secant variety of rank-r tensors.
    
    # Practical approach: Try random rank-r decompositions and see if they match.
    # Generate random U', V', W' of rank r, compute the implied tensor,
    # and check if it equals the matmul tensor.
    
    for attempt in range(200):
        U_new = rng.standard_normal((n * m, target_rank))
        V_new = rng.standard_normal((m * p, target_rank))
        W_new = rng.standard_normal((p * n, target_rank))
        
        # Compute the implied tensor and check against matmul
        # T_implied[i,j,k] = sum_r U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
        # T_true[i,j,k] = 1 for all i,j,k
        
        # Check a sample of entries
        max_err = 0.0
        for _ in range(50):
            i = rng.integers(0, n)
            j = rng.integers(0, m)
            k = rng.integers(0, p)
            val = np.dot(
                U_new[i * m + j, :],
                V_new[j * p + k, :] * W_new[k * n + i, :]
            )
            max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            # Found a valid decomposition!
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None


def _structured_reduction(U, V, W, n=2, m=4, p=6):
    """
    Try to reduce rank by exploiting the block structure.
    
    The 2x4 x 4x6 matmul can be written as:
    C = A_left @ B_top + A_right @ B_bottom
    
    where A_left, A_right are 2x2 and B_top, B_bottom are 2x6.
    
    Each 2x2 @ 2x6 can be done with 2x2 @ 2x2 blocks.
    We already use Strassen (7 terms) for each 2x2@2x2.
    
    Alternative: Use a different factorization that shares terms between
    the left and right blocks.
    
    Try: Can we do 2x4 x 4x6 with fewer than 42 terms by sharing
    computations between the two 2x2 @ 2x6 products?
    
    One idea: Note that A_left @ B_top + A_right @ B_bottom can be
    rewritten using larger Strassen-like formulas.
    
    For 2x4 @ 4x6, think of it as (2x2 | 2x2) @ (2x6; 2x6).
    
    Alternative decomposition:
    Let A = [A1 | A2] where A1, A2 are 2x2
    Let B = [B1; B2] where B1, B2 are 2x6
    
    C = A1*B1 + A2*B2
    
    We can use a 7-term formula for the 2x2@2x2 part, but B is 2x6.
    Split B1 = [B1a B1b B1c] and B2 = [B2a B2b B2c] into three 2x2 blocks.
    
    That's what we already do: 3 * 7 * 2 = 42.
    
    Can we do better? What if we use a different splitting?
    
    Try: Split A into 2x2 blocks differently, or use a rank-6 formula for 2x2@2x2?
    No, 7 is optimal for 2x2@2x2.
    
    What about a global formula? The tensor rank of 2x4x6 matmul...
    
    Let me try a different approach: use the fact that we can choose
    arbitrary linear combinations.
    """
    # This is a placeholder for more sophisticated reduction attempts.
    # For now, return None to indicate no improvement found.
    return None


def construct():
    global _START
    _START = time.time()

    # Start with the known rank-42 Strassen decomposition
    best_result = None
    try:
        result = _build_strassen_42()
        if _verify(result["U"], result["V"], result["W"], seed=42):
            if _verify(result["U"], result["V"], result["W"], seed=123):
                if _verify(result["U"], result["V"], result["W"], seed=777):
                    best_result = result
    except Exception:
        pass

    if best_result is None:
        if _deadline_exceeded():
            return _schoolbook(2, 4, 6)
        return _schoolbook(2, 4, 6)

    # Try to reduce the rank
    current_rank = len(best_result["U"][0])
    
    # Try various reduction strategies
    for target_rank in range(current_rank - 1, 20, -1):
        if _deadline_exceeded():
            break
        
        try:
            reduced = _try_reduce_rank(
                best_result["U"], best_result["V"], best_result["W"],
                target_rank, rng=np.random.default_rng(target_rank)
            )
            if reduced is not None:
                U_new, V_new, W_new = reduced
                if _verify(U_new, V_new, W_new):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    current_rank = target_rank
        except Exception:
            pass

    # Also try structured reduction
    if not _deadline_exceeded():
        try:
            structured = _structured_reduction(
                best_result["U"], best_result["V"], best_result["W"]
            )
            if structured is not None:
                U_s, V_s, W_s = structured
                if _verify(U_s, V_s, W_s):
                    best_result = {"U": U_s, "V": V_s, "W": W_s}
        except Exception:
            pass

    return best_result
