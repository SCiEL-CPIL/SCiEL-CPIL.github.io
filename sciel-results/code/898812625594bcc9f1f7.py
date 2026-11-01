"""Program that searches for a rank-minimal exact tensor decomposition
for multiplying a 2x4 matrix by a 4x8 matrix."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=200, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_l V[l*p+k,c]*B[l,k]) * W[k*n+i,c]
    """
    rank = len(U[0]) if U else 0
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # factor_A[i] = sum_j A[i,j] * U[i*m+j, c]
            factor_A = np.zeros(n)
            for i in range(n):
                s = 0.0
                for j in range(m):
                    s += A[i, j] * U[i * m + j][c]
                factor_A[i] = s
            # factor_B[k] = sum_l V[l*p+k, c] * B[l,k]
            factor_B = np.zeros(p)
            for k in range(p):
                s = 0.0
                for l in range(m):
                    s += V[l * p + k][c] * B[l, k]
                factor_B[k] = s
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True


def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """Faster vectorized verification."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Compute C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_l V[l*p+k,c]*B[l,k]) * W[k*n+i,c]
        # Reshape U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
        U_r = U_arr.reshape(n, m, rank)
        V_r = V_arr.reshape(m, p, rank)
        W_r = W_arr.reshape(p, n, rank)
        # factor_A[i,c] = sum_j A[i,j]*U[i,j,c]
        factor_A = np.einsum('ij,ijc->ic', A, U_r)
        # factor_B[k,c] = sum_l V[l,k,c]*B[l,k]
        factor_B = np.einsum('lk,lc->kc', B, V_r)
        # C[i,k] = sum_c factor_A[i,c] * factor_B[k,c] * W[k,i,c]
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_r)
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True


def _try_rank_reduction(U, V, W, n=2, m=4, p=8, max_attempts=200):
    """Try to reduce rank by removing columns that are redundant."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    rank = len(U[0]) if U else 0

    for _ in range(max_attempts):
        removed = False
        for c in range(rank):
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            if _verify_fast(U_test, V_test, W_test, n, m, p, trials=30, tol=1e-9):
                U, V, W = U_test, V_test, W_test
                rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 64

    # Build schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = schoolbook_rank

    # Strategy 1: Try to find a valid decomposition at rank 63, 62, ...
    # using structured modifications of schoolbook.
    # Key insight: the tensor T_{i,j,k} = e_i e_j^T e_k (the multiplication tensor)
    # has rank exactly n*m*p in general, BUT there may be specific decompositions
    # that exploit structure.

    # Actually, for generic matrix multiplication n x m x p, the rank is known to be
    # at most n*m*p (schoolbook) and the Strassen-type improvements apply mainly
    # to square or specific cases. For 2x4x8, let's see if we can do better.

    # Strategy: Try to merge some schoolbook terms using linearity.
    # For each pair of terms that share structure, try to combine them.

    # Let's try a different approach: use the fact that for the 2x4x8 case,
    # we can view it as blocks and apply Strassen-like ideas.

    # The matrix multiplication C = A * B where A is 2x4, B is 4x8.
    # Split A into columns a1, a2, a3, a4 (each 2x1)
    # Split B into rows b1, b2, b3, b4 (each 1x8)
    # C = a1*b1 + a2*b2 + a3*b3 + a4*b4 (outer product sum)
    # Each outer product of a 2x1 and 1x8 gives a 2x8 matrix = 16 scalar products
    # Total = 4 * 16 = 64 (schoolbook)

    # Can we do better? The rank of the 2x4x8 multiplication tensor is an open
    # question in general, but let's try computational search.

    # Approach: Start from schoolbook and try to find linear dependencies.
    # If we can express one term as a linear combination of others (in the tensor sense),
    # we can remove it.

    # Let's try: for each column c, check if removing it still gives correct results.
    # This is what _try_rank_reduction does. Let's run it on schoolbook.

    if time.time() < deadline - 60:
        U_r, V_r, W_r, rank_r = _try_rank_reduction(
            [row[:] for row in U], [row[:] for row in V], [row[:] for row in W],
            n, m, p, max_attempts=100
        )
        if rank_r < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank_r
            print(f"Rank reduction from schoolbook: {best_rank}")

    # Strategy 2: Try random perturbations of schoolbook + rank reduction
    random.seed(42)
    np.random.seed(42)

    for attempt in range(300):
        if time.time() > deadline - 30:
            break

        # Create perturbed schoolbook
        U_p = [row[:] for row in U]
        V_p = [row[:] for row in V]
        W_p = [row[:] for row in W]

        # Perturb a few random entries
        for _ in range(3):
            ri = random.randint(0, len(U_p) - 1)
            ci = random.randint(0, schoolbook_rank - 1)
            U_p[ri][ci] += random.gauss(0, 0.05)
            V_p[ri % len(V_p)][ci] += random.gauss(0, 0.05)
            W_p[ri % len(W_p)][ci] += random.gauss(0, 0.05)

        # Verify
        if _verify_fast(U_p, V_p, W_p, n, m, p, trials=50, tol=1e-8):
            U_r, V_r, W_r, rank_r = _try_rank_reduction(U_p, V_p, W_p, n, m, p, max_attempts=50)
            if rank_r < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank_r
                print(f"Perturbation improved to rank {best_rank}")

    # Strategy 3: Try to build a decomposition at lower rank directly.
    # For rank r, we need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all A (n x m), B (m x p): C = A@B satisfies
    # C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_l V[l*p+k,c]*B[l,k]) * W[k*n+i,c]

    # This is equivalent to: the bilinear map (A,B) -> C is factored through
    # r rank-1 bilinear forms.

    # Let's try to use the structure: since n=2 is small, we can try to
    # find decompositions where U has special structure.

    # Try rank from 60 down to 40, with random initialization and optimization.
    # Actually, random initialization won't work because the solution space is
    # very constrained. Let's stick with perturbation of schoolbook.

    # Strategy 4: More aggressive perturbation - scale entire columns
    for attempt in range(200):
        if time.time() > deadline - 15:
            break

        U_p = [row[:] for row in U]
        V_p = [row[:] for row in V]
        W_p = [row[:] for row in W]

        # Randomly scale columns
        for ci in range(schoolbook_rank):
            scale = random.gauss(1.0, 0.1)
            for ri in range(len(U_p)):
                U_p[ri][ci] *= scale
                V_p[ri % len(V_p)][ci] /= scale if abs(scale) > 1e-6 else 1.0
                W_p[ri % len(W_p)][ci] *= 1.0  # W stays same to maintain product

        # Hmm, this doesn't preserve the decomposition. Let me think differently.

        # The decomposition requires:
        # C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_l V[l*p+k,c]*B[l,k]) * W[k*n+i,c]
        #
        # If we scale U[:,c] by alpha, V[:,c] by beta, W[:,c] by gamma,
        # the product scales by alpha*beta*gamma. So we need alpha*beta*gamma = 1.
        # This doesn't help reduce rank.

        # Let's try a different perturbation: add a small multiple of one column to another.
        c1 = random.randint(0, schoolbook_rank - 1)
        c2 = random.randint(0, schoolbook_rank - 1)
        if c1 == c2:
            continue
        eps = random.gauss(0, 0.3)

        U_p = [row[:] for row in U]
        V_p = [row[:] for row in V]
        W_p = [row[:] for row in W]

        for ri in range(len(U_p)):
            U_p[ri][c2] += eps * U_p[ri][c1]
        for ri in range(len(V_p)):
            V_p[ri][c2] += eps * V_p[ri][c1]
        for ri in range(len(W_p)):
            W_p[ri][c2] += eps * W_p[ri][c1]

        if _verify_fast(U_p, V_p, W_p, n, m, p, trials=30, tol=1e-8):
            U_r, V_r, W_r, rank_r = _try_rank_reduction(U_p, V_p, W_p, n, m, p, max_attempts=30)
            if rank_r < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank_r
                print(f"Column mix improved to rank {best_rank}")

    # Final: return the best found
    if best_rank <= schoolbook_rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
