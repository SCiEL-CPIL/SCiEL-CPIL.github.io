import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.

    Uses a greedy vertex-cover / maximum-independent-set formulation on a
    conflict graph of the complement, with multiple greedy orders, a
    simulated-annealing polish pass, and a final exact validity check.
    """
    start_time = time.time()
    deadline = start_time + 615.0
    safety_margin = 8.0

    p = 7
    q = p * p  # 49
    alpha_sq = 3

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

    # Precompute line points
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    one_idx = elem_to_idx(1, 0)
    zero_idx = elem_to_idx(0, 0)

    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx

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

    # Build conflict graph among candidate removal points.
    # Two candidate points are in conflict if their joint removal would
    # invalidate the Nikodym property for some third point / direction.
    # We precompute, for each candidate y, the set of (x, d) it "protects".
    # A conflict exists between y1 and y2 if they jointly cover all valid
    # lines of some x, i.e. there is an x where both y1 and y2 lie on every
    # valid line of x.

    candidates = np.where(in_N)[0]
    num_cand = len(candidates)

    # For each candidate y, list of (x, d) such that y lies on line (x, d)
    # and is_valid_dir[x, d] is true.
    cand_protect = []
    for y in candidates:
        prot = []
        for d in range(num_dirs):
            for x in line_points[y, d, :]:
                if is_valid_dir[x, d]:
                    prot.append((x, d))
        cand_protect.append(prot)

    # Build conflict adjacency as a list of sets for each candidate index.
    # To keep it tractable, we build it lazily via a dict of bitsets.
    # conflict[i] = set of j such that y_i and y_j conflict.
    # Two candidates conflict if there exists x such that they cover all
    # valid lines of x (i.e. for every valid direction d of x, the line
    # (x, d) contains either y_i or y_j). We approximate: conflict if there
    # exists x where both are on the same valid line AND that line is the
    # only valid line of x (wc[x]==1) OR they together cover all valid lines.
    # A safe, cheap sufficient condition: conflict if they share a protected
    # (x, d) AND wc[x] == 1. We'll use that for the MIS heuristic.

    # For speed, precompute for each candidate a bitset of x it protects.
    x_protect_bits = []
    for y in candidates:
        bits = 0
        for (x, d) in cand_protect[y]:
            bits |= (1 << x)
        x_protect_bits.append(bits)

    # Conflict: y_i and y_j conflict if there is an x with wc[x] <= 2 such
    # that both protect x and together they cover all valid lines of x.
    # We use a simpler sufficient condition for the greedy: conflict if they
    # protect the same x with wc[x] == 1.
    conflict = [set() for _ in range(num_cand)]
    # Group candidates by x they protect when wc[x]==1
    x_to_cands = {}
    for i, y in enumerate(candidates):
        for (x, d) in cand_protect[y]:
            if valid_count[x] == 1:
                x_to_cands.setdefault(x, set()).add(i)
    for x, cset in x_to_cands.items():
        clist = list(cset)
        for a in range(len(clist)):
            for b in range(a + 1, len(clist)):
                conflict[clist[a]].add(clist[b])
                conflict[clist[b]].add(clist[a])

    # Weighted greedy MIS with random restarts.
    # Weight of a candidate = how "cheap" it is to remove (fewer conflicts
    # and lower coverage). We pick the candidate with best weight/conflict
    # ratio, add to independent set, remove it and its neighbors.

    best_complement = int(np.sum(~in_N))
    best_in_N = in_N.copy()
    best_valid_count = valid_count.copy()
    best_is_valid_dir = is_valid_dir.copy()

    def evaluate_state(work):
        """Return complement size if work is a valid Nikodym set, else -1."""
        # Recompute validity from scratch.
        vc = np.zeros(total_points, dtype=np.int32)
        for pt in range(total_points):
            for d in range(num_dirs):
                if np.all(work[line_points[pt, d, :]]):
                    vc[pt] += 1
        if np.any(vc == 0):
            return -1
        return int(np.sum(~work))

    def greedy_mis(seed, max_iter=None):
        """Weighted greedy independent set on the conflict graph."""
        rng = np.random.RandomState(seed)
        # Work on a copy of in_N.
        work = in_N.copy()
        # Start with all candidates; we will remove a subset.
        # Build weight: lower coverage = cheaper.
        coverage = np.zeros(total_points, dtype=np.int32)
        for y in candidates:
            for (x, d) in cand_protect[y]:
                coverage[y] += 1
        # candidate weight = 1/(coverage+1) plus small random tiebreak
        weights = np.zeros(num_cand, dtype=np.float64)
        for i, y in enumerate(candidates):
            weights[i] = 1.0 / (coverage[y] + 1.0) + rng.random() * 0.01

        # Active candidates
        active = set(range(num_cand))
        chosen = []
        it = 0
        max_iter = max_iter or (num_cand * 3)
        while active and it < max_iter:
            it += 1
            if time.time() > deadline - safety_margin:
                break
            # Pick candidate with best weight / (degree+1)
            best_i = -1
            best_score = -1.0
            # Sample a subset for speed when active is large
            if len(active) > 200:
                sample = set(rng.choice(list(active), size=200, replace=False))
            else:
                sample = active
            for i in sample:
                deg = len(conflict[i] & active)
                score = weights[i] / (deg + 1.0)
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i < 0:
                break
            chosen.append(best_i)
            active.discard(best_i)
            for j in conflict[best_i]:
                active.discard(j)

        # Apply removals
        for i in chosen:
            work[candidates[i]] = False
        comp = evaluate_state(work)
        if comp > 0 and comp > best_complement:
            return comp, work
        return -1, None

    # Simulated annealing polish on the complement.
    def sa_polish(work, seed, n_steps=20000):
        """Anneal: try to remove a point (add to complement) or re-add a
        removed point, accepting based on validity. Objective = complement
        size (maximize). Only accept moves that keep the set valid."""
        rng = np.random.RandomState(seed)
        cur = work.copy()
        cur_comp = evaluate_state(cur)
        if cur_comp < 0:
            return -1, None
        best = cur_comp
        best_state = cur.copy()
        T = 1.0
        cooling = 0.999
        for step in range(n_steps):
            if time.time() > deadline - safety_margin:
                break
            # Random move: try to remove a random in-set point, or re-add a
            # random out-of-set point.
            if rng.random() < 0.7:
                # Remove a point currently in N
                in_set = np.where(cur)[0]
                if len(in_set) == 0:
                    break
                y = in_set[rng.randint(len(in_set))]
                cur[y] = False
                delta = 1
            else:
                # Re-add a point currently out of N
                out_set = np.where(~cur)[0]
                if len(out_set) == 0:
                    break
                y = out_set[rng.randint(len(out_set))]
                cur[y] = True
                delta = -1
            new_comp = evaluate_state(cur)
            if new_comp < 0:
                # Invalid: revert
                cur[y] = not cur[y]
            elif new_comp > cur_comp:
                cur_comp = new_comp
                if new_comp > best:
                    best = new_comp
                    best_state = cur.copy()
            else:
                # new_comp == cur_comp (delta 0 net) or worse; accept with prob
                if delta < 0:
                    # We re-added a point (complement decreased), accept rarely
                    if rng.random() > np.exp((new_comp - cur_comp) / max(T, 1e-9)):
                        cur[y] = not cur[y]
                else:
                    # Removal that didn't change complement count but kept valid
                    # (can't happen since removal increases complement by 1 if valid)
                    pass
            T *= cooling
        return best, best_state

    # Run greedy MIS with several seeds
    greedy_seeds = [42, 123, 456, 789, 101, 202, 303, 7, 13, 99]
    for s in greedy_seeds:
        if time.time() > deadline - safety_margin:
            break
        comp, state = greedy_mis(s)
        if comp > 0 and comp > best_complement:
            best_complement = comp
            best_in_N = state.copy()
            # Recompute validity for the new best
            best_valid_count = np.zeros(total_points, dtype=np.int32)
            best_is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
            for pt in range(total_points):
                for d in range(num_dirs):
                    if np.all(best_in_N[line_points[pt, d, :]]):
                        best_is_valid_dir[pt, d] = True
                        best_valid_count[pt] += 1

    # SA polish from the best state found
    if time.time() < deadline - safety_margin:
        for s in [42, 777, 1234, 555]:
            if time.time() > deadline - safety_margin:
                break
            comp, state = sa_polish(best_in_N, s, n_steps=15000)
            if comp > 0 and comp > best_complement:
                best_complement = comp
                best_in_N = state.copy()
                best_valid_count = np.zeros(total_points, dtype=np.int32)
                best_is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
                for pt in range(total_points):
                    for d in range(num_dirs):
                        if np.all(best_in_N[line_points[pt, d, :]]):
                            best_is_valid_dir[pt, d] = True
                            best_valid_count[pt] += 1

    # Final validity check; if invalid, revert to baseline.
    final_valid = np.zeros(total_points, dtype=np.int32)
    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(best_in_N[line_points[pt, d, :]]):
                final_valid[pt] += 1
    if np.any(final_valid == 0):
        in_N = np.ones(total_points, dtype=bool)
        for x_idx in range(q):
            xa, xb = elems[x_idx]
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = elems[y_idx]
                norm_y = (ya * ya - yb * yb * alpha_sq) % p
                p_idx = x_idx * q + y_idx
                if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                    in_N[p_idx] = False
        best_in_N = in_N.copy()
        best_complement = int(np.sum(~best_in_N))

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
