"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 27, 18
    schoolbook_rank = n * m * p  # 54

    # Precompute the list of (a, b, c) indices for non-zero tensor entries
    # a = i*m+j, b = j*p+k, c = k*n+i
    nonzero_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_triples.append((a, b, c))

    # Also precompute zero-constraint triples (sample a subset for speed)
    # For verification we'll check all, but for optimization we use nonzero + sampled zeros
    all_abc = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    nonzero_set = set(nonzero_triples)
    zero_triples = [t for t in all_abc if t not in nonzero_set]

    # Sample zero triples for optimization (use a subset to keep system manageable)
    rng_zero = np.random.default_rng(0)
    n_zero_sample = min(200, len(zero_triples))
    zero_sample_indices = rng_zero.choice(len(zero_triples), size=n_zero_sample, replace=False)
    zero_sample = [zero_triples[i] for i in zero_sample_indices]

    def build_U_system(V, W, rank):
        """Build the linear system A @ U_flat = b for U given V, W.
        For each nonzero triple (a,b,c): sum_d U[a,d]*V[b,d]*W[c,d] = 1
        For each zero triple (a,b,c): sum_d U[a,d]*V[b,d]*W[c,d] = 0
        """
        n_eq = len(nonzero_triples) + len(zero_sample)
        A = np.zeros((n_eq, nm * rank))
        b = np.zeros(n_eq)
        row = 0
        for (a, b_idx, c_idx) in nonzero_triples:
            for d in range(rank):
                A[row, a * rank + d] = V[b_idx, d] * W[c_idx, d]
            b[row] = 1.0
            row += 1
        for (a, b_idx, c_idx) in zero_sample:
            for d in range(rank):
                A[row, a * rank + d] = V[b_idx, d] * W[c_idx, d]
            b[row] = 0.0
            row += 1
        return A, b

    def build_V_system(U, W, rank):
        """Build the linear system for V given U, W."""
        n_eq = len(nonzero_triples) + len(zero_sample)
        A = np.zeros((n_eq, mp * rank))
        b = np.zeros(n_eq)
        row = 0
        for (a, b_idx, c_idx) in nonzero_triples:
            for d in range(rank):
                A[row, b_idx * rank + d] = U[a, d] * W[c_idx, d]
            b[row] = 1.0
            row += 1
        for (a, b_idx, c_idx) in zero_sample:
            for d in range(rank):
                A[row, b_idx * rank + d] = U[a, d] * W[c_idx, d]
            b[row] = 0.0
            row += 1
        return A, b

    def build_W_system(U, V, rank):
        """Build the linear system for W given U, V."""
        n_eq = len(nonzero_triples) + len(zero_sample)
        A = np.zeros((n_eq, pn * rank))
        b = np.zeros(n_eq)
        row = 0
        for (a, b_idx, c_idx) in nonzero_triples:
            for d in range(rank):
                A[row, c_idx * rank + d] = U[a, d] * V[b_idx, d]
            b[row] = 1.0
            row += 1
        for (a, b_idx, c_idx) in zero_sample:
            for d in range(rank):
                A[row, c_idx * rank + d] = U[a, d] * V[b_idx, d]
            b[row] = 0.0
            row += 1
        return A, b

    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        # Check all nonzero entries
        for (a, b, c) in nonzero_triples:
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(val - 1.0) > tol:
                return False
        # Check all zero entries
        for (a, b, c) in zero_triples:
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(val) > tol:
                return False
        return True

    def snap_and_verify(U, V, W):
        """Snap to half-integers and verify."""
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        if verify_decomposition(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    def try_rank_als(rank, n_restarts=10, max_iters=50):
        """Try to find rank-r decomposition using linear-system ALS."""
        rng = np.random.default_rng(rank * 1000 + 42)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = rng.standard_normal((nm, rank)) * 0.5
            V = rng.standard_normal((mp, rank)) * 0.5
            W = rng.standard_normal((pn, rank)) * 0.5
            
            prev_err = float('inf')
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                A_U, b_U = build_U_system(V, W, rank)
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                    U = U_flat.reshape(nm, rank)
                except:
                    U = rng.standard_normal((nm, rank)) * 0.5
                
                # Update V
                A_V, b_V = build_V_system(U, W, rank)
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                    V = V_flat.reshape(mp, rank)
                except:
                    V = rng.standard_normal((mp, rank)) * 0.5
                
                # Update W
                A_W, b_W = build_W_system(U, V, rank)
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                    W = W_flat.reshape(pn, rank)
                except:
                    W = rng.standard_normal((pn, rank)) * 0.5
                
                # Check convergence
                if verify_decomposition(U, V, W, tol=1e-8):
                    snapped = snap_and_verify(U, V, W)
                    if snapped:
                        return snapped
                    # If it works in float but doesn't snap, try to polish
                    # Do a few more iterations with tighter tolerance
                    for polish in range(20):
                        if time.time() > deadline:
                            break
                        A_U, b_U = build_U_system(V, W, rank)
                        U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                        U = U_flat.reshape(nm, rank)
                        A_V, b_V = build_V_system(U, W, rank)
                        V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                        V = V_flat.reshape(mp, rank)
                        A_W, b_W = build_W_system(U, V, rank)
                        W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                        W = W_flat.reshape(pn, rank)
                        if verify_decomposition(U, V, W, tol=1e-10):
                            snapped = snap_and_verify(U, V, W)
                            if snapped:
                                return snapped
                    # Even if it doesn't snap to half-integers, if it's exact in float
                    # the evaluator might accept it. But spec says integer/half-integer.
                    # Let's try rounding more aggressively
                    U_r = np.round(U * 2) / 2.0
                    V_r = np.round(V * 2) / 2.0
                    W_r = np.round(W * 2) / 2.0
                    if verify_decomposition(U_r, V_r, W_r, tol=1e-9):
                        return U_r, V_r, W_r
            
            # Final check after all iterations
            if verify_decomposition(U, V, W, tol=1e-9):
                snapped = snap_and_verify(U, V, W)
                if snapped:
                    return snapped
    
        return None

    # Schoolbook fallback
    def schoolbook():
        U = np.zeros((nm, schoolbook_rank))
        V = np.zeros((mp, schoolbook_rank))
        W = np.zeros((pn, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_best, V_best, W_best = schoolbook()
    best_rank = schoolbook_rank

    # Try to find lower rank decompositions
    # Start from rank 53 down to a reasonable minimum
    # The theoretical minimum for 2x3x9 is not well-known, but let's search
    for rank in range(schoolbook_rank - 1, 25, -1):
        if time.time() > deadline - 10:
            break
        
        # More restarts for lower ranks
        n_restarts = 15 if rank < 45 else 10
        result = try_rank_als(rank, n_restarts=n_restarts, max_iters=40)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = rank
            # Continue searching for even lower rank if time permits
            if time.time() > deadline - 30:
                break

    # Convert to lists
    U_out = U_best.tolist()
    V_out = V_best.tolist()
    W_out = W_best.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
