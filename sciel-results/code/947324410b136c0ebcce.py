import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Keep 45s margin for 330s limit
    
    # Precompute squared distances between all pairs
    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    # Flatten to 1D array of points
    # point i corresponds to (i // N, i % N)
    all_points_flat = np.arange(N * N)
    
    # Precompute distance matrix (N^2 x N^2) would be 4096x4096 = 16M entries.
    # 16M * 8 bytes = 128 MB. This is acceptable.
    # d2[i, j] = (xi - xj)^2 + (yi - yj)^2
    xi = all_points_flat // N
    yi = all_points_flat % N
    
    # Vectorized distance calculation
    # xi (4096, 1) - xi (1, 4096)
    dx = xi[:, None] - xi[None, :]
    dy = yi[:, None] - yi[None, :]
    d2_matrix = dx * dx + dy * dy
    
    # Strategy:
    # 1. Greedy construction with random ordering to get a good initial set.
    # 2. Local search: try to add points or swap points to increase size.
    #    Since we start from scratch or from a greedy set, we can try "add" moves
    #    if we haven't saturated, but usually greedy gets stuck.
    #    Better: Start with a greedy set. Then perform local search by:
    #    - Removing a point
    #    - Adding a different point
    #    - Or just trying to add a point if the set isn't full (unlikely to succeed if greedy was used on all points).
    #    
    #    Actually, a standard approach for this "no isosceles" (which implies all distances from any point to others are unique, i.e., a Sidon set in distance sense, often called "weak Sidon" or similar) is:
    #    Condition: For any point P in S, all distances |P-Q|^2 for Q in S\P must be distinct.
    #    Also, for any Q in S, and any R, T in S\{P}, we cannot have |Q-R|^2 == |Q-T|^2?
    #    Wait, let's re-read the isosceles condition.
    #    Three points A, B, C form an isosceles triangle if:
    #    |A-B| == |A-C| OR |A-B| == |B-C| OR |A-C| == |B-C|.
    #    This is equivalent to saying:
    #    For every point P in S, the set of distances { |P-Q| : Q in S, Q != P } must be distinct?
    #    No. If |A-B| == |A-C|, A is the apex.
    #    If |A-B| == |B-C|, B is the apex.
    #    If |A-C| == |B-C|, C is the apex.
    #    So, for EVERY point P in S, the distances from P to all OTHER points in S must be distinct.
    #    Yes, that is the condition.
    #    "S is a set of points such that for every P in S, the distances from P to S\{P} are all distinct."
    #    This is known as a "weakly Sidon" set or "no isosceles" set.
    
    # Let's verify the current program's logic:
    # can_add checks:
    # 1. Distances from p to all points in S are distinct. (Apex at p)
    # 2. For each a in S, dist(a, p) is not in dist_from[a]. (Apex at a)
    # This covers all cases.
    
    # Optimization:
    # Using the precomputed d2_matrix, we can speed up the checks.
    # Maintain a boolean mask `in_set` of size N*N.
    # For a candidate point p, and current set S (indices):
    # Check 1: d2_matrix[p, S] must have unique values.
    # Check 2: For each a in S, d2_matrix[a, p] must not be in {d2_matrix[a, q] for q in S, q != a}.
    #          Note: d2_matrix[a, q] is the same as d2_matrix[q, a].
    #          So we need to check if d2_matrix[a, p] is already a distance from a to some other point in S.
    
    # To make this fast, we can maintain for each point a, a set of squared distances to other points in S.
    # Let dist_sets[a] = set of d2_matrix[a, q] for q in S, q != a.
    # When adding p:
    # 1. Check if len(set(d2_matrix[p, S])) == len(S). If not, reject.
    # 2. For each a in S:
    #    d_ap = d2_matrix[a, p]
    #    if d_ap in dist_sets[a]: reject.
    # 3. If accepted:
    #    For each a in S: dist_sets[a].add(d2_matrix[a, p])
    #    dist_sets[p] = set(d2_matrix[p, S])
    
    # This is what the current program does, but with lists of tuples.
    # Using numpy arrays and sets of ints might be faster if we manage it well.
    # However, Python sets are quite fast. The bottleneck in the current code might be the iteration and tuple creation.
    
    # Let's try a different strategy: Random Restart Local Search.
    # 1. Generate a random permutation.
    # 2. Greedily add points.
    # 3. Try to improve by swapping.
    
    # Even better: Simulated Annealing or Simple Hill Climbing on the set of points.
    # But since the problem is "maximize size", and we can't easily remove points to add better ones without complex bookkeeping,
    # let's stick to:
    # 1. Fast Greedy with multiple random orders.
    # 2. For each greedy result, try to "swap" a point in the set with a point outside to see if we can increase size?
    #    No, swapping keeps size constant. We want to increase size.
    #    So we just need to find a point outside that can be added.
    #    If greedy ran on ALL points, then no point outside can be added (by definition of greedy failure? No, greedy order matters).
    #    If we process all points in a random order, we might skip a point because it conflicts with earlier points.
    #    Later points might also conflict.
    #    Is it possible that a point skipped earlier could be added if we removed some other point?
    #    Yes. So local search with swaps or removals is needed.
    
    # However, given the 300s limit and N=64, a simple multi-restart greedy might be sufficient if optimized.
    # Let's optimize the greedy step using numpy for the "unique distances" check.
    
    def greedy_once(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        order = list(range(N * N))
        random.shuffle(order)
        
        S = []
        # dist_from[a] is a set of squared distances from a to other points in S
        dist_from = {}
        
        for p_idx in order:
            # Check if p_idx can be added
            # 1. Distances from p_idx to all in S are unique
            if len(S) > 0:
                # d2_matrix[p_idx, :] gives distances to all points.
                # We need distances to points in S.
                # Let S_indices be a list of indices in S.
                # d_vals = d2_matrix[p_idx, S_indices]
                # Check uniqueness: len(set(d_vals)) == len(d_vals)
                
                # To avoid creating a huge list every time, we can maintain S as a list of indices.
                # For small S, this is fine.
                
                d_vals = d2_matrix[p_idx, S]
                if len(set(d_vals)) != len(d_vals):
                    continue
                
                # 2. For each a in S, d2_matrix[a, p_idx] not in dist_from[a]
                valid = True
                for a in S:
                    d_ap = d2_matrix[a, p_idx]
                    if d_ap in dist_from[a]:
                        valid = False
                        break
                if not valid:
                    continue
            
            # Add p_idx
            S.append(p_idx)
            dist_from[p_idx] = set()
            for a in S[:-1]:
                d_ap = d2_matrix[a, p_idx]
                dist_from[a].add(d_ap)
                dist_from[p_idx].add(d_ap)
        
        return S, dist_from

    best_S = []
    best_dist_from = {}
    
    # Run multiple greedy passes
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        S, dist_from = greedy_once(random_seed=i)
        if len(S) > len(best_S):
            best_S = S
            best_dist_from = dist_from
            
    # Now, try to improve best_S via local search
    # Idea: Pick a random point in S, remove it, and try to add a random point not in S.
    # If successful, keep it.
    # This is a form of "swap" but since we want to maximize size, we should only do swaps if we can then add another point?
    # Actually, removing a point decreases size by 1. Adding a point increases by 1.
    # So a swap keeps size constant.
    # We want to INCREASE size.
    # So we should look for a point not in S that can be added to S.
    # But if we ran greedy on all points, we already tried all points.
    # However, the context of "which points are in S" changed.
    # A point that couldn't be added before might be addable now if we removed a conflicting point.
    
    # So the strategy:
    # 1. Have S.
    # 2. Identify a set of "blocked" points (points not in S that failed to add).
    # 3. Pick a point q in S to remove.
    # 4. Try to add a point p not in S (maybe one of the blocked ones, or any).
    # 5. If p can be added to (S \ {q}), accept the new set S' = (S \ {q}) U {p}.
    # 6. Now try to add another point r to S'.
    # 7. If r can be added, accept S'' = S' U {r}. Net gain +1.
    # 8. If not, revert to S or keep S'? Keeping S' (same size) might help escape local optima for future additions.
    #    But we want to maximize size. So we should prefer moves that increase size.
    #    If we can't increase size, we might still swap to change the configuration.
    
    # Let's implement a simple local search:
    # 1. Maintain S and dist_from.
    # 2. Loop:
    #    a. Try to add a random point not in S.
    #       If success, keep it. (Size increases)
    #    b. If fail, try a swap: remove random q from S, add random p not in S.
    #       If p can be added to S\{q}, perform the swap.
    #       Then immediately try to add another point to the new S.
    #       If success, size increases by 1 (net).
    #       If fail, we are at same size. We can keep the swap (exploration) or revert.
    #       Keeping the swap is usually better for SA/Hill Climbing.
    
    # To check "can p be added to S\{q}", we need to temporarily remove q.
    # Removing q from S:
    # - Remove q from list S.
    # - Remove distances to q from dist_from of all other points in S.
    # - Check if p can be added.
    # - If yes, add p.
    # - If no, revert: re-add q.
    
    # This is expensive if done naively.
    # But with N=64, S is likely small (< 100).
    # Let's estimate S size.
    # Current score 9.625.
    # If seed is 1.0, and score is 9.625, does that mean 9.625 points?
    # "Score: number of points".
    # "Score is value/seed, so 1.0 equals the seed program".
    # So the seed program returns a set of size 1? Or size 10?
    # If seed returns size 1, and we return 9, score is 9.
    # If seed returns size 10, and we return 96, score is 9.6.
    # Given N=64, max points is 4096.
    # No-isosceles sets are sparse.
    # For N=64, a good set might have ~50-100 points?
    # Let's assume S is around 50-100.
    
    # Optimized Local Search:
    
    def remove_point(S, dist_from, q):
        S.remove(q)
        # Remove distances to q from all a in S
        for a in S:
            d_aq = d2_matrix[a, q]
            dist_from[a].discard(d_aq)
        del dist_from[q]
        
    def add_point(S, dist_from, p):
        S.append(p)
        dist_from[p] = set()
        for a in S[:-1]:
            d_ap = d2_matrix[a, p]
            dist_from[a].add(d_ap)
            dist_from[p].add(d_ap)

    def can_add_to(S, dist_from, p):
        if len(S) > 0:
            d_vals = d2_matrix[p, S]
            if len(set(d_vals)) != len(d_vals):
                return False
            for a in S:
                d_ap = d2_matrix[a, p]
                if d_ap in dist_from[a]:
                    return False
        return True

    S = best_S[:]
    dist_from = {k: set(v) for k, v in best_dist_from.items()}
    
    in_S = np.zeros(N * N, dtype=bool)
    for p in S:
        in_S[p] = True
        
    not_in_S = np.where(~in_S)[0]
    
    steps = 0
    while time.time() < deadline - 5:
        steps += 1
        # 1. Try to add a random point not in S
        if len(not_in_S) > 0:
            p = not_in_S[random.randint(0, len(not_in_S) - 1)]
            if can_add_to(S, dist_from, p):
                add_point(S, dist_from, p)
                in_S[p] = True
                not_in_S = np.where(~in_S)[0]
                # Restart the "not_in_S" list since size changed?
                # np.where is O(N^2). N^2=4096. Fast enough.
                continue
        
        # 2. Try a swap
        if len(S) > 0 and len(not_in_S) > 0:
            q = S[random.randint(0, len(S) - 1)]
            p = not_in_S[random.randint(0, len(not_in_S) - 1)]
            
            # Remove q
            remove_point(S, dist_from, q)
            in_S[q] = False
            
            # Try to add p
            if can_add_to(S, dist_from, p):
                add_point(S, dist_from, p)
                in_S[p] = True
                # Swap accepted
                # Now try to add another point to increase size
                not_in_S = np.where(~in_S)[0]
                if len(not_in_S) > 0:
                    p2 = not_in_S[random.randint(0, len(not_in_S) - 1)]
                    if can_add_to(S, dist_from, p2):
                        add_point(S, dist_from, p2)
                        in_S[p2] = True
                        # Size increased by 1
                        not_in_S = np.where(~in_S)[0]
            else:
                # Revert swap
                add_point(S, dist_from, q)
                in_S[q] = True
                # not_in_S doesn't change (p was not added, q is back)
                # Actually, p was not in S, so not_in_S is unchanged.
        
        if steps % 1000 == 0 and len(S) > len(best_S):
            best_S = S[:]
            best_dist_from = {k: set(v) for k, v in dist_from.items()}

    # Final update
    if len(S) > len(best_S):
        best_S = S[:]
        
    # Convert indices to (x, y) tuples
    result = [(p // N, p % N) for p in best_S]
    return result

# Test run
if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} points")
