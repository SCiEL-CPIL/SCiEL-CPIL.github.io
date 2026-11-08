"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def check_time():
        return time.time() > deadline

    def schoolbook():
        """Guaranteed valid schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, n, m, p):
        """Exact verification: T[(i,j),(j,k),(k,i)] = 1 for all valid, 0 otherwise."""
        rank = len(U[0])
        # Check the 60 "on" entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check "off" entries (sampled + full if fast)
        # Full check: for all (a,b,c) with a=(i,j),b=(j',k),c=(k',i'),
        # T=1 iff i==i' and j==j' and k==k'
        # We check all off-diagonal to be safe but it's expensive; do a thorough check.
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                    if expected == 1.0:
                        continue  # already checked
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W, n, m, p):
        """Faster verification using numpy: check all (i,j,k) on-entries and a full tensor check."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # on-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.einsum('r,r,r->', U_np[i*m+j], V_np[j*p+k], W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        # full tensor check via einsum: T[a,b,c] = sum_r U[a,r]V[b,r]W[c,r]
        # Build full T and compare to target
        T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        # Target
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        if not np.allclose(T, target, atol=1e-9):
            return False
        return True

    # Build the M_{2,3,2} rank-11 block decomposition
    def build_block_terms():
        """Return list of 11 terms for M_{2,3,2}. Each term: U list of (i,j,val), V list of (j,k,val), W list of (k,i,val)."""
        strassen_terms = [
            {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
            {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
            {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
            {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
            {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
        ]
        outer_terms = [
            {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
            {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
            {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
            {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
        ]
        return strassen_terms + outer_terms

    def build_block_decomposition(rank_per_block):
        """Build full U,V,W for p=10 as 5 blocks of size 2, each with rank_per_block terms."""
        num_blocks = p // 2
        block_size = 2
        total_rank = num_blocks * rank_per_block
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        terms = build_block_terms()
        col_offset = 0
        for b in range(num_blocks):
            for term in terms:
                col = col_offset
                for i, j, val in term['U']:
                    U[i * m + j][col] += val
                for j, k_local, val in term['V']:
                    k_global = b * block_size + k_local
                    V[j * p + k_global][col] += val
                for k_local, i, val in term['W']:
                    k_global = b * block_size + k_local
                    W[k_global * n + i][col] += val
                col_offset += 1
        return U, V, W

    # First, build and verify the rank-55 block decomposition
    best_result = None
    try:
        U55, V55, W55 = build_block_decomposition(11)
        if verify_fast(U55, V55, W55, n, m, p):
            best_result = {"U": U55, "V": V55, "W": W55}
        else:
            # fall back to schoolbook
            best_result = schoolbook()
    except Exception:
        best_result = schoolbook()

    if check_time():
        return best_result

    # Try to find a rank-10 M_{2,3,2} decomposition to get total rank 50
    # Strategy: search over integer/half-integer entries for the block,
    # then verify the full 5-block assembly.
    def try_rank10_block():
        """Attempt to find a rank-10 decomposition of M_{2,3,2}."""
        # We need 10 terms. Each term contributes to the 12 bilinear products.
        # Use a randomized search: start from rank-11 and try to merge/remove one term,
        # then solve for the remaining entries to make it exact.
        # This is a system of equations; use least squares / exact rational solve.
        #
        # Approach: pick 10 "active" (i,j,k) triples to be the support, and solve
        # for U,V,W such that the tensor equals the M_{2,3,2} tensor.
        # Actually, a cleaner approach: the M_{2,3,2} tensor has 2*3*2=12 nonzero
        # entries (one per (i,j,k)). We want rank-10, i.e., 10 rank-1 tensors summing to it.
        #
        # Let's use a parameterized search: fix the structure where each term r has
        # support on some subset, and solve linearly.
        #
        # Simpler: use the known fact that M_{2,3,2} rank is 11 (proven minimal in lit).
        # So rank-10 likely does not exist. We'll do a quick randomized attempt to be sure.
        np.random.seed(12345)
        n2, m2, p2 = 2, 3, 2
        rank = 10
        # Target tensor
        target = np.zeros((n2*m2, m2*p2, p2*n2))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    target[i*m2+j, j*p2+k, k*n2+i] = 1.0
        for attempt in range(2000):
            if check_time():
                return None
            # Random initialization with small integers/half-integers
            U = np.random.randint(-2, 3, size=(n2*m2, rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m2*p2, rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p2*n2, rank)) * 0.5
            # Compute current tensor
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - target)
            if err < 1e-9:
                # Convert to lists
                return (U.tolist(), V.tolist(), W.tolist())
        return None

    block10 = try_rank10_block()
    if block10 is not None:
        U10, V10, W10 = block10
        # Assemble full 5-block
        num_blocks = p // 2
        block_size = 2
        total_rank = num_blocks * 10
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        for b in range(num_blocks):
            for r in range(10):
                col = b * 10 + r
                for a in range(n * m):
                    if U10[a][r] != 0:
                        U[a][col] += U10[a][r]
                for bb in range(m * p):
                    # map local (j,k_local) to global (j, b*block_size+k_local)
                    j = bb // p
                    k_local = bb % p
                    # local index in V10 is j*p2 + k_local = j*2 + k_local
                    local_idx = j * 2 + k_local
                    if V10[local_idx][r] != 0:
                        V[bb][col] += V10[local_idx][r]
                for c in range(p * n):
                    # map local (k_local, i) to global (b*block_size+k_local, i)
                    k_global = c // n
                    i = c % n
                    k_local = k_global - b * block_size
                    local_idx = k_local * n + i
                    if W10[local_idx][r] != 0:
                        W[c][col] += W10[local_idx][r]
        if verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    return best_result
# EVOLVE-BLOCK-END
