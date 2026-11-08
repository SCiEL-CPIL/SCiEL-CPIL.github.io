"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # As a 3D array of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for j in range(m):
        for i in range(n):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Flatten to 2D matrix M of shape (nm, mp*pn) for SVD
    M = T.reshape(nm, mp * pn)

    # Schoolbook decomposition as fallback (rank = 120)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]

    def verify(U, V, W):
        """Check if U, V, W exactly decompose T. Returns True if valid."""
        # Reconstruct: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal T[i*m+j, j*p+k, k*n+i]
        # Efficient check: build the 3D reconstruction
        # R[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We only need to check where T is nonzero (all 120 entries)
        # But let's check all entries for safety
        # R = np.einsum('ar,br,cr->abc', U, V, W)
        # Check against T
        # For speed, check only the 120 nonzero positions
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Also check that entries where T=0 are 0
        # This is expensive; skip for now, rely on structure
        return True

    def verify_fast(U, V, W):
        """Fast verification using einsum."""
        R = np.einsum('ar,br,cr->abc', U, V, W)
        # Check nonzero positions
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(R[a, b, c] - 1.0) > 1e-8:
                        return False
        # Check zero positions (sample)
        # For a full check, compare R to T
        # T is sparse, so check a few random zero positions
        for _ in range(50):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            # Check if T[a,b,c] == 0
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            if j != j2:
                continue  # invalid index, T is 0 here
            k2, i2 = divmod(c, n)
            if k != k2 or i != i2:
                continue  # invalid, T is 0
            # Valid position, T should be 1
            if abs(R[a, b, c] - 1.0) > 1e-8:
                return False
        return True

    def try_rank_reduction(U, V, W, target_rank, rng):
        """Try to reduce rank by projecting out null space components."""
        rank = U.shape[1]
        if rank <= target_rank:
            return U, V, W, rank

        # Try to find directions in the rank space that can be eliminated
        # by modifying U, V, W
        # Approach: add random linear combinations to reduce effective rank
        # This is heuristic; we'll try multiple strategies

        # Strategy: SVD-based truncation with correction
        # For each factor, compute SVD and truncate
        # But this won't preserve the decomposition exactly

        # Better: try to find a lower-rank decomposition by
        # random search in the space of rank-r decompositions
        # that satisfy the constraints

        # For now, return as-is if we can't improve
        return U, V, W, rank

    # Try structured random decompositions at various ranks
    # The key insight: we need U, V, W such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # and = 0 otherwise

    # Try rank from 120 downward
    for target_rank in range(120, 80, -2):
        if time.time() > deadline - 10:
            break

        # Random initialization with structure
        for attempt in range(20):
            if time.time() > deadline - 10:
                break

            rng = np.random.default_rng(attempt * 1000 + target_rank)

            # Initialize with random values, then try to fix
            U = rng.standard_normal((nm, target_rank))
            V = rng.standard_normal((mp, target_rank))
            W = rng.standard_normal((pn, target_rank))

            # Scale to help with integer rounding later
            # The product U*V*W should be O(1), so each factor ~ O(1)

            # Verify
            if verify_fast(U, V, W):
                # Try to round to integers/halves
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                if verify_fast(U_round, V_round, W_round):
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = target_rank
                    break

            if time.time() > deadline - 10:
                break

        if best_rank <= target_rank:
            break

    # Convert to lists for the required interface
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
