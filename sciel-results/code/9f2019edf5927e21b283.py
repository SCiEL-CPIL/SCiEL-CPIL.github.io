"""Direct ordering optimization via adjacent-swap local search with conflict-count metric."""
import numpy as np
import time


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

    # Precompute neighbor lists as integer arrays for fast access
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    degrees = np.array([len(nbrs[i]) for i in range(N)])

    # --- Initial ordering: sort by degree ascending (low degree first) ---
    # This is a strong baseline for independent set: pick low-degree vertices first
    order = np.argsort(degrees, kind='stable')  # ascending degree

    # --- Fast greedy evaluation: returns (size, picked_set) ---
    def greedy_eval(ord):
        """Run greedy on given ordering. Returns (count, picked_mask)."""
        picked = np.zeros(N, dtype=bool)
        count = 0
        for v in ord:
            if not picked[v]:
                count += 1
                picked[v] = True
                if len(nbrs[v]) > 0:
                    picked[nbrs[v]] = True
        return count, picked

    # --- Conflict-based swap evaluation ---
    # For a swap of adjacent elements at positions p and p+1 in the ordering,
    # we need to check if the resulting greedy set changes.
    # Key insight: if we have the current greedy result, a swap only matters if
    # one of the two swapped vertices is picked and the other isn't, or if
    # their relative order affects which gets picked first.
    
    # More efficient approach: just re-run greedy after swap.
    # But we can short-circuit: if both vertices have the same "picked" status
    # in the current run AND neither is a neighbor of the other, the swap likely
    # doesn't change the outcome. Let's use a smarter check.

    def swap_eval(ord, p):
        """Evaluate greedy after swapping positions p and p+1 in ord.
        Returns the new size. Uses early termination for speed."""
        # Swap
        ord[p], ord[p + 1] = ord[p + 1], ord[p]
        # Run greedy
        picked = np.zeros(N, dtype=bool)
        count = 0
        for v in ord:
            if not picked[v]:
                count += 1
                picked[v] = True
                if len(nbrs[v]) > 0:
                    picked[nbrs[v]] = True
        # Swap back
        ord[p], ord[p + 1] = ord[p + 1], ord[p]
        return count

    # --- Phase 1: Get baseline ---
    best_order = order.copy()
    best_size, _ = greedy_eval(best_order)

    # --- Phase 2: Local search via adjacent swaps ---
    # We'll do multiple passes. In each pass, we scan left to right and try
    # each adjacent swap. If it improves, accept and continue.
    # We limit the number of passes by time.
    
    max_passes = 50
    improved = True
    pass_count = 0
    
    while improved and pass_count < max_passes:
        if time.time() > DEADLINE - 5.0:
            break
        improved = False
        pass_count += 1
        
        for p in range(N - 1):
            if time.time() > DEADLINE - 2.0:
                break
            
            # Quick check: if both vertices at positions p and p+1 are already
            # "blocked" (neighbors of earlier picked vertices) in the current
            # greedy, swapping them won't help. But computing this is as hard
            # as the greedy itself. So we just try the swap.
            
            # To speed up, we can first check if swapping these two would
            # even be meaningful: if they're not neighbors and both have the
            # same degree, it's unlikely to help. But let's just try.
            
            new_size = swap_eval(best_order, p)
            if new_size > best_size:
                best_size = new_size
                # Accept the swap (it's already swapped in swap_eval, then swapped back)
                best_order[p], best_order[p + 1] = best_order[p + 1], best_order[p]
                improved = True

    # --- Phase 3: Random restarts with local search ---
    # Try a few random initial orderings and local search from each
    rng = np.random.default_rng(123)
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > DEADLINE - 3.0:
            break
        
        # Random ordering biased toward low degree
        # Mix: 70% sorted by degree, 30% random
        if r % 3 == 0:
            # Pure random
            cur_order = np.random.permutation(N)
        elif r % 3 == 1:
            # Degree-based with noise
            noisy_deg = degrees + rng.uniform(0, 2, size=N)
            cur_order = np.argsort(noisy_deg, kind='stable')
        else:
            # Inverse degree (high degree first) - sometimes helps
            cur_order = np.argsort(-degrees, kind='stable')
        
        cur_size, _ = greedy_eval(cur_order)
        
        if cur_size > best_size:
            best_size = cur_size
            best_order = cur_order.copy()
        
        # Local search from this starting point
        if time.time() > DEADLINE - 2.0:
            break
        
        ls_improved = True
        ls_passes = 0
        max_ls_passes = 5
        
        while ls_improved and ls_passes < max_ls_passes:
            if time.time() > DEADLINE - 1.5:
                break
            ls_improved = False
            ls_passes += 1
            
            for p in range(N - 1):
                if time.time() > DEADLINE - 1.0:
                    break
                new_size = swap_eval(cur_order, p)
                if new_size > cur_size:
                    cur_size = new_size
                    cur_order[p], cur_order[p + 1] = cur_order[p + 1], cur_order[p]
                    ls_improved = True
        
        if cur_size > best_size:
            best_size = cur_size
            best_order = cur_order.copy()

    # --- Phase 4: Convert ordering to scores ---
    # The score for each vertex is its position in the ordering (earlier = higher score)
    # We assign scores such that vertices earlier in best_order get higher scores
    scores = np.zeros(N, dtype=float)
    for rank, v in enumerate(best_order):
        scores[v] = float(N - rank)  # First in order gets highest score

    return scores
