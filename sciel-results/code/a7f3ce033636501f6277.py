"""Optimal ±1 polynomials with 24 coefficients minimizing sup norm."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    N = 24
    HALF = 12
    sqrtN = math.sqrt(N)

    # --- Precompute: all 2^12 sub-polynomial coefficients ---
    total = 1 << HALF  # 4096
    S = np.zeros((total, HALF), dtype=np.float64)
    for i in range(total):
        for k in range(HALF):
            S[i, k] = 1.0 if (i >> k) & 1 else -1.0

    # --- Dense grid for the DFT ---
    G = 16384  # Increase grid size for tighter bound
    angles = 2 * np.pi * np.arange(G) / G
    zs = np.exp(1j * angles)

    # Build V1[j,k] = z_j^k  (k=0..11), V2[j,k] = z_j^(12+k) = z_j^12 * z_j^k
    V1 = np.ones((G, HALF), dtype=np.complex128)
    for k in range(1, HALF):
        V1[:, k] = V1[:, k-1] * zs

    z12 = zs ** HALF
    V2 = np.ones((G, HALF), dtype=np.complex128)
    V2[:, 0] = z12
    for k in range(1, HALF):
        V2[:, k] = V2[:, k-1] * zs

    # M_A[j,i] = A_i(z_j); M_B[j,i] = B_i(z_j)
    M_A = V1 @ S.T   # (G, 4096) complex
    M_B = V2 @ S.T   # (G, 4096) complex

    # Precompute magnitudes for bounding
    abs_M_A = np.abs(M_A)  # (G, 4096)
    abs_M_B = np.abs(M_B)  # (G, 4096)

    # --- Score function for a pair (i,j) ---
    def score_pair(i, j):
        val = M_A[:, i] + M_B[:, j]
        return np.max(np.abs(val)) / sqrtN

    def time_ok():
        return time.time() < deadline - 5.0

    # --- Phase 1: Find a good initial upper bound using random restarts ---
    best_score = np.inf
    best_i = -1
    best_j = -1

    rng = np.random.RandomState(12345)

    # Quick random search to get a decent initial bound
    for _ in range(1000):
        if not time_ok():
            break
        i = rng.randint(0, total)
        j = rng.randint(0, total)
        cur = score_pair(i, j)
        if cur < best_score:
            best_score = cur
            best_i = i
            best_j = j

    # --- Phase 2: Branch and Bound ---
    # We search over the 24 bits. We use a priority queue (min-heap) ordered by lower bound.
    # State: (lower_bound, bit_index, prefix_value, i_mask, j_mask)
    # i_mask: bits set in the first 12 coefficients
    # j_mask: bits set in the last 12 coefficients
    
    # To make B&B efficient, we need to evaluate the partial sum.
    # P(z) = sum_{k=0}^{23} c_k z^k
    # Let's define the prefix sum for the first 'depth' bits.
    # However, the decomposition P = A + z^12 B makes B&B slightly tricky because
    # the "prefix" of the 24-bit string doesn't map cleanly to A and B magnitudes
    # unless we interleave or handle the two halves separately.
    
    # Alternative: B&B on the 24-bit string directly.
    # Precompute z^k for k=0..23.
    Z = np.ones((G, N), dtype=np.complex128)
    for k in range(1, N):
        Z[:, k] = Z[:, k-1] * zs
    
    # Precompute contributions of each bit?
    # If bit k is 1, contribution is +z^k. If 0, -z^k.
    # Let c_k be in {1, -1}.
    # P(z) = sum c_k z^k.
    # Let's fix the first 'd' bits.
    # The remaining N-d bits can be anything.
    # The maximum possible value of |P(z)| given the prefix is:
    # max_j | sum_{k<d} c_k z_j^k + sum_{k>=d} c_k z_j^k |
    # <= max_j | sum_{k<d} c_k z_j^k | + max_j | sum_{k>=d} z_j^k |
    # This bound is loose.
    
    # A tighter bound:
    # For a fixed prefix, the remaining sum is a vector R_j.
    # We want to minimize max_j |Prefix_j + R_j|.
    # The term R_j depends on the choices of the remaining bits.
    # The set of possible R_j is the set of all sums of the remaining basis vectors.
    # This is a large set.
    
    # Let's stick to the pair (i,j) B&B.
    # We can search i and j independently? No, the score is coupled.
    
    # Let's try a different structural approach: **Iterated Local Search with Simulated Annealing**
    # but with a much better cooling schedule and a "kick" mechanism that flips a cluster of bits.
    # The current program uses simple greedy. Simulated Annealing can escape local minima.
    
    # Actually, the directive asks for a MATERIALLY DIFFERENT strategy.
    # The current one is Greedy Bit-Flip.
    # Let's implement **Simulated Annealing** with a proper cooling schedule.
    
    T_start = 0.5
    T_end = 0.001
    cooling_rate = 0.995
    
    current_i = best_i
    current_j = best_j
    current_score = best_score
    
    best_i_sa = current_i
    best_j_sa = current_j
    best_score_sa = current_score
    
    T = T_start
    
    while T > T_end and time_ok():
        # Generate a neighbor by flipping 1 or 2 random bits
        n_flips = rng.randint(1, 3)
        new_i = current_i
        new_j = current_j
        
        for _ in range(n_flips):
            if rng.random() < 0.5:
                new_i ^= (1 << rng.randint(0, HALF))
            else:
                new_j ^= (1 << rng.randint(0, HALF))
                
        new_score = score_pair(new_i, new_j)
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_i = new_i
            current_j = new_j
            current_score = new_score
            
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_i_sa = current_i
                best_j_sa = current_j
                
        T *= cooling_rate
        
    if best_score_sa < best_score:
        best_score = best_score_sa
        best_i = best_i_sa
        best_j = best_j_sa

    # --- Phase 3: Final Polish with Greedy ---
    cur_i, cur_j = best_i, best_j
    cur_score = best_score
    improved = True
    while improved and time_ok():
        improved = False
        for k in range(HALF):
            ni = cur_i ^ (1 << k)
            s = score_pair(ni, cur_j)
            if s < cur_score:
                cur_score = s
                cur_i = ni
                improved = True
        for k in range(HALF):
            nj = cur_j ^ (1 << k)
            s = score_pair(cur_i, nj)
            if s < cur_score:
                cur_score = s
                cur_j = nj
                improved = True

    # Build final coefficients
    coeffs = np.zeros(N, dtype=np.float64)
    coeffs[:HALF] = S[cur_i, :]
    coeffs[HALF:] = S[cur_j, :]

    return coeffs.astype(int).tolist()

# EVOLVE-BLOCK-END
