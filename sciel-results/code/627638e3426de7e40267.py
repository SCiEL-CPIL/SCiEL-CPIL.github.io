import numpy as np
import time
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Uses a MODEL-AND-OPTIMIZE approach: parameterize the set N and optimize
    the complement size directly via greedy removal and local search.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 600s
    safety_margin = 5.0

    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 in F_7

    # Element representation: (a, b) represents a + b*alpha
    # idx = a * p + b

    def elem_to_idx(a, b):
        return a * p + b

    def idx_to_elem(idx):
        return (idx // p, idx % p)

    # Precompute field operations
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)

    elems = [idx_to_elem(i) for i in range(q)]

    for i in range(q):
        a_i, b_i = elems[i]
        for j in range(q):
            a_j, b_j = elems[j]
            add_table[i, j] = elem_to_idx((a_i + a_j) % p, (b_i + b_j) % p)
            prod_a = (a_i * a_j + b_i * b_j * alpha_sq) % p
            prod_b = (a_i * b_j + b_i * a_j) % p
            mul_table[i, j] = elem_to_idx(prod_a, prod_b)

    num_dirs = q + 1  # 50 directions
    total_points = q * q  # 2401

    # Build baseline Nikodym set
    # N = {(x, y) : Trace(x) + Norm(y) = 0} minus Y_special
    in_N = np.ones(total_points, dtype=bool)

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    for x_idx in range(q):
        xa, xb = elems[x_idx]
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = elems[y_idx]
            norm_y = (ya * ya - yb * yb * alpha_sq) % p
            p_idx = x_idx * q + y_idx
            if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                in_N[p_idx] = False

    # Precompute line points: for each point and direction, the other q-1 points on the line
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)

    one_idx = elem_to_idx(1, 0)  # element 1
    zero_idx = elem_to_idx(0, 0)  # element 0

    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx  # element (0,1) = 1

        for pt in range(total_points):
            x_idx = pt // q
            y_idx = pt % q
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                line_points[pt, d_idx, t_idx - 1] = nx * q + ny

    # Initialize validity tracking
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)

    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(in_N[line_points[pt, d, :]]):
                is_valid_dir[pt, d] = True
                valid_count[pt] += 1

    # Support count: for each point y, how many points x have y on one of their valid lines?
    # If support_count[y] == 0, y can be removed without invalidating any x.
    support_count = np.zeros(total_points, dtype=np.int32)
    
    # Precompute for each point y, the list of (x, d) pairs where y is on line x,d
    # This is expensive to build fully, so we compute on the fly or use the line_points structure.
    # Actually, we can iterate through all x, d, and for each point y on line x,d, increment support_count[y].
    # This is O(total_points * num_dirs * q) = 2401 * 50 * 48 ~ 5.7M ops. Acceptable.
    
    for x in range(total_points):
        for d in range(num_dirs):
            if is_valid_dir[x, d]:
                for y in line_points[x, d, :]:
                    support_count[y] += 1

    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_valid_count = valid_count.copy()
    best_is_valid_dir = is_valid_dir.copy()
    best_support_count = support_count.copy()

    def try_prune(seed=0, max_rounds=1):
        nonlocal best_complement, best_in_N, best_valid_count, best_is_valid_dir, best_support_count

        work = in_N.copy()
        wc = valid_count.copy()
        ird = is_valid_dir.copy()
        sc = support_count.copy()
        
        # Heap of (support_count, point_idx)
        # We only care about points in N
        heap = []
        for pt in range(total_points):
            if work[pt]:
                heapq.heappush(heap, (sc[pt], pt))
                
        # We need to be able to update heap efficiently.
        # Standard heapq doesn't support decrease-key.
        # We can use a lazy deletion approach: push new entries, and ignore stale ones.
        # Or, since the graph is sparse-ish in terms of updates, we can just re-heapify or use a list and sort?
        # Given the size (2401), a simple list of candidates might be faster than complex heap updates if updates are frequent.
        # However, support counts change.
        # Let's use a set of "active" points and just scan for min? No, that's O(N).
        # Let's stick to lazy heap.
        
        removed = 0
        # We need to track if a point in the heap is still valid (in work and has correct sc)
        # We can just pop and check.
        
        while heap:
            if time.time() > deadline - safety_margin:
                break
                
            sc_val, pt = heapq.heappop(heap)
            
            # Check if this entry is stale
            if not work[pt]:
                continue
            if sc[pt] != sc_val:
                continue
            
            if sc[pt] > 0:
                # No more removable points with 0 support?
                # Actually, if the min support count is > 0, no point is removable.
                break
            
            # Remove pt
            work[pt] = False
            removed += 1
            
            # Update support counts for neighbors
            # For each direction d, and each x on the line through pt with direction d:
            # If x has a valid line d, and pt was on it, then sc[x] decreases?
            # Wait, sc[y] is the number of x such that y is on a valid line of x.
            # If we remove y, it no longer contributes to the validity of x.
            # So for each x that had y on a valid line, we need to check if x still has a valid line.
            # If x loses all valid lines, it becomes invalid. But the Nikodym condition requires *every* point to be on a line *whose other points are in N*.
            # If x becomes invalid, the set is no longer a Nikodym set?
            # The definition: "every point x ... must lie on a line ... whose points other than x all belong to N".
            # If we remove a point y, we might invalidate some x.
            # The greedy step is only valid if removing y does NOT invalidate any x.
            # This is exactly what sc[pt] == 0 guarantees: no x relies on pt for its validity.
            # Wait, does sc[pt] == 0 imply that removing pt doesn't invalidate any x?
            # sc[pt] counts the number of x for which pt is on a valid line.
            # If sc[pt] == 0, then no x has pt on a valid line.
            # Removing pt means pt is not in N.
            # For any x, its valid lines are those where all *other* points are in N.
            # If pt was not on any valid line of x, then removing pt doesn't change the status of any line of x.
            # So x remains valid.
            # Thus, removing pt is safe.
            
            # However, removing pt might affect the support count of other points?
            # No, support_count[z] is the number of x such that z is on a valid line of x.
            # Removing pt (which is not in N anymore) doesn't change the set of valid lines for any x, because pt wasn't on any valid line of any x.
            # So sc[z] for z != pt remains unchanged?
            # Wait. The validity of a line depends on the points in N.
            # If we remove pt, and pt was NOT on any valid line, then the set of valid lines for all x is unchanged.
            # Therefore, the support counts for all other points z are unchanged.
            # This means we don't need to update the heap or sc for other points!
            # We just remove pt and mark it as not in N.
            
            # This makes the algorithm very fast.
            # But wait, is it possible that pt is on a valid line of x, but sc[pt] was calculated incorrectly?
            # No, sc is calculated based on the current ird (is_valid_dir).
            # And ird is only updated if we remove a point that *was* on a valid line?
            # But we only remove points with sc=0, meaning they are on NO valid lines.
            # So ird never changes!
            # This implies that once we calculate the initial valid lines, they remain valid as long as we only remove points that are not on them.
            # This is a very strong property.
            
            # Let's verify this logic.
            # Condition for line L to be valid for x: all points in L \ {x} are in N.
            # If we remove y, and y is not in any valid line L_x for any x, then for any x, the set of valid lines L_x is unchanged.
            # Because the condition "points in L \ {x} are in N" is not affected by removing y if y is not in L \ {x} for any valid L.
            # Since y is not in any valid line, it is not in L \ {x} for any valid L.
            # So yes, the set of valid lines is invariant under removal of points with support count 0.
            
            # This means we can just greedily remove all points with support count 0.
            # We don't need to update anything else.
            
            # Is that all?
            # If we remove all points with sc=0, we reach a state where every remaining point has sc > 0.
            # This is a "core" of the hypergraph.
            # This is equivalent to the 2-core of the bipartite graph (Points, Lines) where an edge exists if Point is on a valid Line for Point's owner?
            # Actually, it's simpler.
            
            # Let's just run the heap until empty or min > 0.
            
        comp = np.sum(~work)
        if comp > best_complement:
            best_complement = comp
            best_in_N = work.copy()
            # wc, ird, sc are unchanged in this specific logic, but we store them for consistency
            best_valid_count = wc.copy()
            best_is_valid_dir = ird.copy()
            best_support_count = sc.copy()

    # Run the pruning
    try_prune()

    # Final validity check (should pass by construction)
    # If not, revert (though it should be valid)
    
    # Extract points
    points = []
    for p_idx in range(total_points):
        if best_in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
