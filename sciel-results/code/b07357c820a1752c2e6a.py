import time
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    def check_time():
        return time.time() - t0 < DEADLINE

    # Target tensor: T[i,j,k] = 1 for all i,j,k
    target = np.ones((n, m, p), dtype=np.int64)

    # Initialize with schoolbook (rank 144)
    r0 = n * m * p
    U0 = np.zeros((nm, r0), dtype=np.int64)
    V0 = np.zeros((mp, r0), dtype=np.int64)
    W0 = np.zeros((pn, r0), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, c] = 1
                V0[j*p+k, c] = 1
                W0[k*n+i, c] = 1
                c += 1

    def compute_tensor(U, V, W, r):
        """Compute T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        T = np.zeros((n, m, p), dtype=np.int64)
        for c in range(r):
            ur = U[:, c].reshape(n, m)
            vr = V[:, c].reshape(m, p)
            wr = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', ur, vr, wr)
        return T

    def is_valid(U, V, W, r):
        T = compute_tensor(U, V, W, r)
        return np.all(T == target)

    def try_remove_column(U, V, W, r, col):
        """Try removing column col; return new (U,V,W,r-1) if still valid."""
        if r <= 1:
            return None
        U_new = np.delete(U, col, axis=1)
        V_new = np.delete(V, col, axis=1)
        W_new = np.delete(W, col, axis=1)
        if is_valid(U_new, V_new, W_new, r - 1):
            return U_new, V_new, W_new, r - 1
        return None

    def random_integer_mix(U, V, W, r, seed_val):
        """Apply a random invertible integer transformation to columns.
        This preserves the tensor if done correctly (mixing columns of U,V,W
        with the same transformation). We use a random unimodular matrix.
        """
        rng = np.random.RandomState(seed_val)
        # Generate a random unimodular matrix (det = ±1)
        # Simple approach: random permutation + random sign flips + random row additions
        r_small = min(r, 64)  # limit to keep it fast
        if r <= 2:
            return U, V, W, r

        # Build a random unimodular matrix of size r x r
        M = np.eye(r, dtype=np.int64)
        # Apply random elementary operations
        for _ in range(r * 2):
            row1 = rng.randint(r)
            row2 = rng.randint(r)
            if row1 == row2:
                continue
            op = rng.randint(3)
            if op == 0:
                # Swap rows
                M[[row1, row2], :] = M[[row2, row1], :]
            elif op == 1:
                # Negate a row
                M[row1, :] *= -1
            else:
                # Add k * row2 to row1 (k = -1, 0, 1)
                k = rng.randint(-1, 2)
                if k != 0:
                    M[row1, :] += k * M[row2, :]

        # Apply: new_U = U @ M, etc.
        # But M must be applied as column mixing: new_col_c = sum_d M[d,c] * old_col_d
        # So U_new = U @ M
        U_new = U @ M
        V_new = V @ M
        W_new = W @ M

        # Verify validity (should be preserved by unimodular mixing)
        if not is_valid(U_new, V_new, W_new, r):
            return U, V, W, r
        return U_new, V_new, W_new, r

    def prune_all(U, V, W, r):
        """Greedily remove any column that can be removed, repeatedly."""
        improved = True
        while improved and check_time():
            improved = False
            for col in range(r):
                if not check_time():
                    break
                result = try_remove_column(U, V, W, r, col)
                if result is not None:
                    U, V, W, r = result
                    improved = True
                    break
        return U, V, W, r

    # Start
    U, V, W, r = U0, V0, W0, r0

    # First, do initial pruning (should remove nothing from schoolbook, but just in case)
    U, V, W, r = prune_all(U, V, W, r)
    best_r = r

    # Randomized search: mix columns, then prune
    seed_counter = 0
    while check_time() and r > 50:
        # Apply random unimodular mixing
        U_mix, V_mix, W_mix, r_mix = random_integer_mix(U, V, W, r, seed_counter)
        seed_counter += 1

        # Prune
        U_pruned, V_pruned, W_pruned, r_pruned = prune_all(U_mix, V_mix, W_mix, r_mix)

        if r_pruned < r:
            U, V, W, r = U_pruned, V_pruned, W_pruned, r_pruned
            best_r = min(best_r, r)
        else:
            # Keep the mixed version if it didn't get worse (it's the same rank)
            # Actually keep original to avoid drift; only accept improvements
            pass

        # Also try: from the current best, do a single random mix and prune
        if check_time():
            U_mix2, V_mix2, W_mix2, _ = random_integer_mix(U, V, W, r, seed_counter + 1000000)
            seed_counter += 1
            U_p2, V_p2, W_p2, r_p2 = prune_all(U_mix2, V_mix2, W_mix2, r)
            if r_p2 < r:
                U, V, W, r = U_p2, V_p2, W_p2, r_p2
                best_r = min(best_r, r)

    # Final: try removing columns one more time
    U, V, W, r = prune_all(U, V, W, r)
    best_r = min(best_r, r)

    # Validate final result
    if not is_valid(U, V, W, r):
        # Fallback to schoolbook
        U = U0.tolist()
        V = V0.tolist()
        W = W0.tolist()
    else:
        U = U.tolist()
        V = V.tolist()
        W = W.tolist()

    return {"U": U, "V": V, "W": W}
