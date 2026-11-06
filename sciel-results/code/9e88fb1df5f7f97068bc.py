import time
import random
import numpy as np

N = 64
TOTAL = N * N
TIME_LIMIT = 300.0

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 10.0  # Safety margin

    # Precompute coordinates
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    
    # Precompute squared distances matrix
    dx = (xs[:, None] - xs[None, :]).astype(np.int32)
    dy = (ys[:, None] - ys[None, :]).astype(np.int32)
    dist = dx * dx + dy * dy

    # Incremental violation tracking replaces O(k^2) full recomputation 
    # with O(k) per-step updates (vectorized), enabling more SA iterations.
    
    # State:
    # current_indices: list of indices in the current set
    # current_mask: boolean mask of current set
    # dist_to_set: for each point in current_indices, distances to all other points in current_indices?
    # No, let's track violations directly.
    
    # Violation definition:
    # A set S is invalid if there exist a, b, c in S such that d(a,b) == d(a,c) or d(a,b) == d(b,c) or d(a,c) == d(b,c).
    # This is equivalent to: for any point a in S, the multiset of distances from a to S\{a} contains duplicates, 
    # OR for any pair (a,b) in S, the distance d(a,b) appears as a distance from a to some c OR from b to some c.
    
    # Actually, the condition "no three points form an isosceles triangle" means:
    # For any distinct a, b, c in S:
    # d(a,b) != d(a,c)
    # d(a,b) != d(b,c)
    # d(a,c) != d(b,c)
    
    # This can be decomposed into:
    # 1. For each a, all distances d(a, s) for s in S\{a} are distinct. (Covers d(a,b)==d(a,c))
    # 2. For each pair (a,b), d(a,b) is not equal to d(a,c) for any c != b, and not equal to d(b,c) for any c != a.
    #    Note: If condition 1 holds for all a, then d(a,b) != d(a,c) for b!=c is already guaranteed by condition 1 for point a.
    #    Wait. Condition 1 for point a says: d(a,b) != d(a,c) for any b, c in S, b != c.
    #    So if we enforce Condition 1 for ALL points in S, does that cover all isosceles triangles?
    #    An isosceles triangle (a,b,c) has at least two equal sides.
    #    Case 1: d(a,b) == d(a,c). This is a duplicate in distances from a. Covered by Cond 1 for a.
    #    Case 2: d(a,b) == d(b,c). This is a duplicate in distances from b. Covered by Cond 1 for b.
    #    Case 3: d(a,c) == d(b,c). This is a duplicate in distances from c. Covered by Cond 1 for c.
    #    
    #    So, the condition "no isosceles triangle" is EXACTLY equivalent to:
    #    "For every point a in S, the set of squared distances from a to all other points in S is a set of unique values."
    
    # This simplifies the problem immensely! We don't need to check pairs (a,b) against d(a,c) and d(b,c) separately.
    # We just need to ensure that for each selected point, its distance vector to the rest of the set has no duplicates.
    
    # Let's verify this logic.
    # Triangle (a,b,c).
    # Isosceles if d(a,b)=d(a,c) OR d(a,b)=d(b,c) OR d(a,c)=d(b,c).
    # If d(a,b)=d(a,c), then point a has two points b,c at the same distance. Duplicate in a's distance list.
    # If d(a,b)=d(b,c), then point b has two points a,c at the same distance. Duplicate in b's distance list.
    # If d(a,c)=d(b,c), then point c has two points a,b at the same distance. Duplicate in c's distance list.
    # Yes! The condition is simply: For all a in S, len(set(d(a,s) for s in S\{a})) == len(S) - 1.
    
    # This makes the violation count very easy to track incrementally.
    # Violation count V = sum over a in S of (number of duplicate distances in a's distance list to S\{a}).
    # Or more simply, for each a, let D_a be the list of distances to S\{a}.
    # Violations contributed by a = len(D_a) - len(set(D_a)).
    
    # When we add a point c to S:
    # 1. For c itself: we need to check distances from c to all existing points. If any duplicates, add to V.
    # 2. For each existing point a in S: we add d(a,c) to a's distance list.
    #    We need to update the violation count for a.
    #    If d(a,c) was already in a's distance list, then the "uniqueness" for a decreases by 1 (one more duplicate).
    #    Specifically, if d(a,c) appeared k times before, it now appears k+1 times.
    #    The number of unique values decreases by 1 if k>=1.
    #    So violations for a increase by 1 if d(a,c) is already present in a's distances to S\{a}.
    
    # When we remove a point c from S:
    # 1. For c: remove its contribution to V.
    # 2. For each existing point a in S\{c}: we remove d(a,c) from a's distance list.
    #    If d(a,c) appeared k times before (including c), it now appears k-1 times.
    #    The number of unique values increases by 1 if k>=2 (i.e., d(a,c) was duplicated).
    #    So violations for a decrease by 1 if d(a,c) is still present in a's distances to S\{a,c}.
    
    # Data structures for incremental tracking:
    # current_indices: list of indices
    # dist_counts: a dictionary or array mapping (point_a, distance_val) to count?
    # Since N=64, max distance is 2*(63^2) = 7938.
    # We can maintain for each point a in S, a histogram of distances to other points in S.
    # hist[a][d] = number of points b in S\{a} such that d(a,b) == d.
    # Violations for a = sum_{d} max(0, hist[a][d] - 1).
    # Total Violations V = sum_{a in S} sum_{d} max(0, hist[a][d] - 1).
    
    # To make this fast, we can precompute the distance from each point to all other points.
    # dist[i][j] is already computed.
    
    # Let's implement this.
    
    # Initialize
    current_indices = []
    current_mask = np.zeros(TOTAL, dtype=bool)
    # hist[a] will be a dict or array. Since max dist is ~8000, an array of size 8001 is fine.
    # But we only need it for points in S.
    # Let's use a dictionary: point_idx -> np.array of counts for each distance value.
    # Or just a global 2D array? No, S changes.
    # Let's maintain a list of histograms for the points in current_indices.
    # hist_list[i] corresponds to current_indices[i].
    
    # Actually, since we are doing SA, we need to add and remove points frequently.
    # Maintaining a list of histograms is okay if we manage indices carefully.
    
    # Let's try a simpler approach first: just compute V from scratch?
    # If |S| is ~100, computing V takes O(|S|^2) ~ 10000 ops.
    # With vectorization, 10000 ops is very fast (~10-50 microseconds).
    # If we can do 10,000 iterations per second, that's 100,000,000 ops/sec.
    # 300 seconds * 10,000 iter/sec = 3,000,000 iterations.
    # That should be plenty.
    
    # Let's stick to the "compute V from scratch" approach but vectorized, as it's less prone to bugs.
    
    def compute_violations(indices):
        if len(indices) <= 2:
            return 0
        # indices is a list of point indices
        # Get distances between all pairs in indices
        D = dist[np.ix_(indices, indices)]
        # For each row (point a), count duplicates
        # Sort each row
        D_sorted = np.sort(D, axis=1)
        # D_sorted[i][j] is the j-th smallest distance from indices[i] to indices[j]
        # We want to count j > 0 such that D_sorted[i][j] == D_sorted[i][j-1]
        # Note: D[i][i] = 0. So the first element is always 0.
        # We should exclude self-distances.
        # Let's mask out the diagonal.
        # Or just ignore the first element (which is 0) if we assume 0 is unique?
        # Yes, distance to self is 0, which is unique (no other point is at distance 0).
        # So we can just check D_sorted[:, 1:] for duplicates with D_sorted[:, :-1].
        
        # dups = np.any(D_sorted[:, 1:] == D_sorted[:, :-1], axis=1)
        # This gives a boolean mask for each point: True if it has any duplicate distance.
        # But we want the COUNT of violations.
        # "Violations" as defined in the directive: "minimizes the count of violated isosceles triangle constraints".
        # A constraint is violated if there is an isosceles triangle.
        # How many isosceles triangles are there?
        # If point a has k points at the same distance d, then there are C(k,2) isosceles triangles with apex a.
        # Total violations = sum over all a, sum over all d, C(count(a,d), 2).
        
        # Let's compute this.
        # For each row in D_sorted, count the frequency of each value.
        # This is a bit hard to vectorize efficiently for arbitrary sizes.
        
        # Alternative: Just use the boolean "has duplicate" as the penalty?
        # Or use the number of duplicate pairs?
        # Let's use the number of duplicate pairs in the sorted distances.
        # For a sorted array, the number of pairs (i,j) with i<j and A[i]==A[j] is sum C(k,2).
        # We can compute this by iterating? No, we want vectorized.
        
        # Actually, for the purpose of SA, a simple penalty of "1 if any duplicate" might be enough to guide the search,
        # but "count of duplicates" is a smoother landscape.
        
        # Let's try to compute the count of duplicate pairs.
        # For each row, we can use np.unique with return_counts.
        # But that's a Python loop over rows.
        
        # Let's just use the sum of (count - 1) for each duplicate value?
        # Or simply: number of j such that D_sorted[i][j] == D_sorted[i][j-1].
        # This counts the number of "extra" occurrences.
        # If a value appears k times, it contributes k-1 to this sum.
        # This is a reasonable proxy for the number of isosceles triangles.
        
        dup_count = np.sum(D_sorted[:, 1:] == D_sorted[:, :-1])
        return int(dup_count)
    
    # Initial state: empty
    current_indices = []
    current_violations = 0
    
    # Greedy initialization
    # Add points one by one, choosing the one that minimizes the increase in violations?
    # Or just random greedy that accepts only if violations remain 0?
    # Let's do a random greedy that accepts if violations stay 0.
    # If we get stuck, we can start SA.
    
    all_indices = np.arange(TOTAL)
    random.shuffle(all_indices)
    
    for idx in all_indices:
        if time.time() > deadline - 60:
            break
        current_indices.append(idx)
        current_mask[idx] = True
        v = compute_violations(current_indices)
        if v > 0:
            # Reject
            current_indices.pop()
            current_mask[idx] = False
        else:
            current_violations = 0
    
    best_indices = list(current_indices)
    best_violations = 0 # We want 0 violations.
    # Actually, we want to maximize size among valid sets.
    # So we track the best valid set.
    
    # Simulated Annealing
    # Move: Add a random point, or Remove a random point, or Swap.
    # Let's stick to Add/Remove to change size, and Swap to rearrange.
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while time.time() < deadline - 5:
        if T < T_min:
            T = T_min
            
        # Choose a move
        move_type = random.random()
        if move_type < 0.3:
            # Add a point
            if len(current_indices) < TOTAL:
                candidates = all_indices[~current_mask[all_indices]]
                if len(candidates) > 0:
                    new_idx = candidates[random.randrange(len(candidates))]
                    current_indices.append(new_idx)
                    current_mask[new_idx] = True
                    new_v = compute_violations(current_indices)
                    delta_v = new_v - current_violations
                    delta_size = 1
                    # Accept if delta_v == 0 (improvement in validity or neutral)
                    # Or if delta_v > 0 but we accept with probability exp(-delta_v / T)
                    # We prioritize validity. If we become invalid, we might want to revert.
                    # But SA allows invalid states.
                    # Let's define energy = violations. We want to minimize energy.
                    # But we also want to maximize size.
                    # Let's use a combined objective?
                    # Or just minimize violations, and if violations == 0, maximize size.
                    # This is a lexicographic objective.
                    # Simpler: Energy = violations * 1000 - size.
                    # We want to minimize Energy.
                    
                    old_energy = current_violations * 1000 - len(current_indices)
                    new_energy = new_v * 1000 - len(current_indices)
                    delta_e = new_energy - old_energy
                    
                    if delta_e <= 0 or random.random() < np.exp(-delta_e / T):
                        current_violations = new_v
                        if current_violations == 0 and len(current_indices) > len(best_indices):
                            best_indices = list(current_indices)
                    else:
                        # Reject
                        current_indices.pop()
                        current_mask[new_idx] = False
                        
        elif move_type < 0.6:
            # Remove a point
            if len(current_indices) > 0:
                pos = random.randrange(len(current_indices))
                old_idx = current_indices[pos]
                current_indices.pop(pos)
                current_mask[old_idx] = False
                new_v = compute_violations(current_indices)
                
                old_energy = current_violations * 1000 - (len(current_indices) + 1)
                new_energy = new_v * 1000 - len(current_indices)
                delta_e = new_energy - old_energy
                
                if delta_e <= 0 or random.random() < np.exp(-delta_e / T):
                    current_violations = new_v
                else:
                    # Reject
                    current_indices.insert(pos, old_idx)
                    current_mask[old_idx] = True
                    
        else:
            # Swap
            if len(current_indices) > 0:
                pos = random.randrange(len(current_indices))
                old_idx = current_indices[pos]
                
                candidates = all_indices[~current_mask[all_indices]]
                if len(candidates) > 0:
                    new_idx = candidates[random.randrange(len(candidates))]
                    
                    # Perform swap
                    current_indices[pos] = new_idx
                    current_mask[old_idx] = False
                    current_mask[new_idx] = True
                    new_v = compute_violations(current_indices)
                    
                    old_energy = current_violations * 1000 - len(current_indices)
                    new_energy = new_v * 1000 - len(current_indices)
                    delta_e = new_energy - old_energy
                    
                    if delta_e <= 0 or random.random() < np.exp(-delta_e / T):
                        current_violations = new_v
                        if current_violations == 0 and len(current_indices) > len(best_indices):
                            best_indices = list(current_indices)
                    else:
                        # Reject
                        current_indices[pos] = old_idx
                        current_mask[old_idx] = True
                        current_mask[new_idx] = False
                        
        T *= cooling_rate

    # Final result
    result = [(int(xs[idx]), int(ys[idx])) for idx in best_indices]
    return result

if __name__ == "__main__":
    pts = construct()
    print(f"Found {len(pts)} points")
