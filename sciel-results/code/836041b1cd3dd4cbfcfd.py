"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin below 150s

    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18

    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W exactly decompose T."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=atol)

    def is_half_integer(A, tol=1e-9):
        """Check if all entries are half-integers (k/2 for integer k)."""
        doubled = A * 2.0
        rounded = np.round(doubled)
        return np.all(np.abs(doubled - rounded) < tol)

    def snap_to_half_integer(A):
        """Snap entries to nearest half-integer."""
        return np.round(A * 2.0) / 2.0

    # Schoolbook decomposition (rank = n*m*p = 90)
    rank_school = n * m * p
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Strategy: Try to find lower-rank decompositions using structured approaches
    # Key insight: For small matrices, we can try various rank values and use
    # a combination of random initialization + local optimization + snapping

    rng = np.random.RandomState(12345)

    # Try ranks from 45 (half of schoolbook) up to schoolbook
    # Focus on promising ranges first
    candidate_ranks = list(range(45, 61)) + list(range(61, rank_school + 1))
    
    # Also try some very low ranks to see if anything works
    candidate_ranks = list(range(30, 45)) + candidate_ranks
    candidate_ranks = sorted(set(candidate_ranks))

    for r in candidate_ranks:
        if time.time() > deadline - 30:
            break
        if r >= best_rank:
            continue
            
        # Multiple attempts per rank
        max_attempts = 5 if r < 50 else 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                break
                
            # Initialize with random values
            U = rng.randn(nm, r) * 0.3
            V = rng.randn(mp, r) * 0.3
            W = rng.randn(pn, r) * 0.3
            
            # Local search: perturb columns
            obj_prev = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
            
            # Adaptive learning rate
            lr = 0.1
            stuck_count = 0
            
            max_iters = 500
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                    
                # Pick a random matrix and column to perturb
                which = rng.randint(0, 3)
                col_idx = rng.randint(0, r)
                
                if which == 0:
                    delta = rng.randn(nm) * lr
                    U_new = U.copy()
                    U_new[:, col_idx] += delta
                    cand = (U_new, V, W)
                elif which == 1:
                    delta = rng.randn(mp) * lr
                    V_new = V.copy()
                    V_new[:, col_idx] += delta
                    cand = (U, V_new, W)
                else:
                    delta = rng.randn(pn) * lr
                    W_new = W.copy()
                    W_new[:, col_idx] += delta
                    cand = (U, V, W_new)

                obj_new = np.linalg.norm(np.einsum('ar,br,cr->abc', *cand) - T)
                if obj_new < obj_prev:
                    U, V, W = cand
                    obj_prev = obj_new
                    stuck_count = 0
                    lr = min(lr * 1.05, 0.5)
                else:
                    stuck_count += 1
                    if stuck_count > 50:
                        lr *= 0.8
                        stuck_count = 0
                        if lr < 0.001:
                            break

            # Try to snap to half-integers and verify
            if obj_prev < 1e-6:
                U_h = snap_to_half_integer(U)
                V_h = snap_to_half_integer(V)
                W_h = snap_to_half_integer(W)
                
                if is_half_integer(U_h) and is_half_integer(V_h) and is_half_integer(W_h):
                    if verify_decomposition(U_h, V_h, W_h, atol=1e-8):
                        if r < best_rank:
                            best_rank = r
                            best_U, best_V, best_W = U_h.copy(), V_h.copy(), W_h.copy()
                        break  # found exact at this rank

            # Also try a more aggressive snap: project onto nearest half-integer grid
            # and then do a few refinement steps
            if obj_prev < 0.1:
                U_h = snap_to_half_integer(U)
                V_h = snap_to_half_integer(V)
                W_h = snap_to_half_integer(W)
                
                # Check if the snapped version is close
                T_rec_h = np.einsum('ar,br,cr->abc', U_h, V_h, W_h)
                err_h = np.linalg.norm(T_rec_h - T)
                
                if err_h < 1e-6:
                    if is_half_integer(U_h) and is_half_integer(V_h) and is_half_integer(W_h):
                        if verify_decomposition(U_h, V_h, W_h, atol=1e-8):
                            if r < best_rank:
                                best_rank = r
                                best_U, best_V, best_W = U_h.copy(), V_h.copy(), W_h.copy()
                            break

    # Final cleanup: ensure entries are exact half-integers
    def clean_half_int(lst):
        out = []
        for row in lst:
            new_row = []
            for v in row:
                v2 = round(float(v) * 2) / 2.0
                if v2 == int(v2):
                    new_row.append(int(v2))
                else:
                    new_row.append(float(v2))
            out.append(new_row)
        return out

    U_out = clean_half_int(best_U.tolist())
    V_out = clean_half_int(best_V.tolist())
    W_out = clean_half_int(best_W.tolist())

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
