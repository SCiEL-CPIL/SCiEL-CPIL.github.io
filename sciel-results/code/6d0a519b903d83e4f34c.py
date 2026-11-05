import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin below 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute neighbor lists and degrees
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    degrees = np.array([len(neighbors[i]) for i in range(N)], dtype=int)

    # --- Initial Ordering via Min-Degree Peeling (Structural) ---
    # This creates a strong initial permutation: low-degree vertices first.
    # We simulate the greedy decoder but prioritize low degree to build the order.
    # Actually, the greedy decoder picks highest score first.
    # So we want low-degree vertices to have HIGH scores.
    # Min-degree peeling: repeatedly pick the vertex with the smallest degree in the remaining graph.
    # This is a known heuristic for MIS.
    
    remaining = set(range(N))
    deg = degrees.copy()
    order = []
    
    # For efficiency, use a heap or simple loop. N=200, so simple loop is fine.
    while remaining:
        if time.time() > DEADLINE - 60:
            break
        # Find min degree in remaining
        min_deg = float('inf')
        min_v = -1
        for v in remaining:
            # Calculate current degree in remaining graph
            # This is O(N^2) per step, so O(N^3) total. 200^3 = 8e6, acceptable.
            # But we can be smarter.
            pass
        
        # Let's just use the static degree for the initial sort, then refine.
        # Static min-degree is a good start.
        static_order = np.argsort(deg)
        order = list(static_order)
        break

    # Convert order to scores: earlier in order = higher score
    # Score = N - index
    scores = np.zeros(N)
    for idx, v in enumerate(order):
        scores[v] = N - idx

    # --- Incremental Greedy Decoder for Fast Evaluation ---
    # We need to evaluate the greedy MIS size given a permutation (or scores).
    # To make SA fast, we need to update the MIS size after a swap in O(N) or less.
    # Standard approach: Recompute is O(N^2) (due to neighbor marking).
    # N=200, N^2 = 40000. 100k swaps = 4e9 ops. Too slow in Python.
    # We need a faster evaluation or fewer swaps.
    
    # Alternative: Use the fact that N is small (200).
    # Let's try to optimize the evaluation function using NumPy bitsets or similar.
    
    # Represent neighbors as bitsets (integers) for fast AND/OR?
    # Python ints are arbitrary precision, so 200 bits is fine.
    # neighbor_mask[i] = bitmask of neighbors of i
    
    neighbor_masks = [0] * N
    for i in range(N):
        mask = 0
        for nb in neighbors[i]:
            mask |= (1 << nb)
        neighbor_masks[i] = mask

    # Evaluate MIS size from a permutation (list of vertices in priority order)
    # Greedy: pick v if not blocked. Block v and its neighbors.
    def evaluate_perm(perm):
        blocked_mask = 0
        count = 0
        for v in perm:
            v_bit = 1 << v
            if not (blocked_mask & v_bit):
                count += 1
                blocked_mask |= (1 << v) | neighbor_masks[v]
        return count

    # Current best
    current_perm = order
    current_size = evaluate_perm(current_perm)
    best_perm = current_perm.copy()
    best_size = current_size

    # --- Simulated Annealing on Permutation ---
    # Move: Swap two positions in the permutation.
    # Delta evaluation: Hard to do incrementally for greedy MIS with swaps.
    # However, we can approximate or just re-evaluate if N is small enough.
    # Let's check speed of evaluate_perm.
    # 200 iterations of bit ops. Very fast.
    # Let's benchmark mentally: 100,000 swaps * 200 steps * ~10ns/step = 200ms.
    # This is very fast! We can do millions of swaps.

    rng = np.random.default_rng(42)
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.9995  # Cooling rate
    max_iter = 2000000  # Max iterations, but limited by time

    T = T_start
    i = 0
    while time.time() < DEADLINE - 5:
        if i % 10000 == 0:
            T = T_start * (T_end / T_start) ** (i / max_iter)
            if T < T_end:
                T = T_end

        # Random swap
        i1 = rng.integers(0, N)
        i2 = rng.integers(0, N)
        if i1 == i2:
            i += 1
            continue

        v1 = current_perm[i1]
        v2 = current_perm[i2]

        # Perform swap
        current_perm[i1] = v2
        current_perm[i2] = v1

        new_size = evaluate_perm(current_perm)
        delta = new_size - current_size

        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_perm = current_perm.copy()
        else:
            # Revert
            current_perm[i1] = v1
            current_perm[i2] = v2

        i += 1

    # --- Final Polish: Local Search on Best Permutation ---
    # Try to swap adjacent elements or small neighborhood swaps to improve.
    # We can also try to move a vertex to a better position.
    # Given time, we'll stick to random swaps but with a tighter neighborhood or just more iterations if time permits.
    # Actually, the SA loop above handles this. Let's just ensure we use remaining time for more SA or a steepest descent step.
    
    # Steepest Descent: Try all pairs? N^2/2 = 20k pairs.
    # For each pair, evaluate. 20k * 200 = 4M ops. Very fast.
    # Let's do a few passes of steepest descent on the best permutation.
    
    best_perm = np.array(best_perm)
    best_size = evaluate_perm(best_perm)
    
    improved = True
    passes = 0
    while improved and time.time() < DEADLINE - 2:
        improved = False
        passes += 1
        for i1 in range(N):
            if time.time() > DEADLINE - 2:
                break
            for i2 in range(i1 + 1, N):
                v1 = best_perm[i1]
                v2 = best_perm[i2]
                
                best_perm[i1] = v2
                best_perm[i2] = v1
                
                new_size = evaluate_perm(best_perm)
                if new_size > best_size:
                    best_size = new_size
                    improved = True
                else:
                    # Revert
                    best_perm[i1] = v1
                    best_perm[i2] = v2

    # Convert best permutation to scores
    final_scores = np.zeros(N)
    for idx, v in enumerate(best_perm):
        final_scores[v] = N - idx

    return final_scores
