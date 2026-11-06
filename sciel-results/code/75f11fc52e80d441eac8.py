import time
import random

def _is_sidon(subset):
    """Check if a subset has all pair sums (including diagonals) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_subset_size(S):
    """Find the largest Sidon subset of S using backtracking with pruning."""
    S = sorted(S)
    n = len(S)
    if _is_sidon(S):
        return n
    for size in range(n - 1, 0, -1):
        def backtrack(start, current):
            if len(current) == size:
                return True
            if n - start < size - len(current):
                return False
            for i in range(start, n):
                candidate = current + [S[i]]
                if _is_sidon(candidate):
                    if backtrack(i + 1, candidate):
                        return True
            return False
        if backtrack(0, []):
            return size
    return 1

def _collision_density(S):
    """Count total sum collisions (edges in the collision graph). Higher = denser."""
    from collections import defaultdict
    sums = defaultdict(list)
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            sums[S[i] + S[j]].append((i, j))
    # Total excess pairs over unique sums = sum over sums of C(count,2)
    collisions = 0
    for c in sums.values():
        collisions += len(c) * (len(c) - 1) // 2
    return collisions

def _collision_graph_edges(S):
    """Return list of collision edges: pairs of (i,j) pairs that share the same sum."""
    from collections import defaultdict
    sums = defaultdict(list)
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            sums[S[i] + S[j]].append((i, j))
    edges = []
    for pairs in sums.values():
        if len(pairs) > 1:
            for a in range(len(pairs)):
                for b in range(a + 1, len(pairs)):
                    edges.append((pairs[a], pairs[b]))
    return edges

def _max_independent_set_size(S):
    """Approximate max Sidon subset via greedy on collision graph."""
    edges = _collision_graph_edges(S)
    if not edges:
        return len(S)
    
    # Build adjacency: vertices are (i,j) pairs, edge if they collide
    vertex_to_idx = {}
    for idx, (pair1, pair2) in enumerate(edges):
        for p in (pair1, pair2):
            if p not in vertex_to_idx:
                vertex_to_idx[p] = len(vertex_to_idx)
    
    adj = {v: set() for v in vertex_to_idx}
    for pair1, pair2 in edges:
        v1 = vertex_to_idx[pair1]
        v2 = vertex_to_idx[pair2]
        adj[v1].add(v2)
        adj[v2].add(v1)
    
    # Greedy: pick vertex with fewest neighbors, remove it and its neighbors
    remaining = set(vertex_to_idx.values())
    picked = []
    while remaining:
        # Find vertex with minimum degree in remaining
        min_deg = float('inf')
        best_v = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_v = v
        picked.append(best_v)
        remaining.discard(best_v)
        for u in adj[best_v]:
            remaining.discard(u)
    
    # Map back: each picked vertex corresponds to a (i,j) pair
    # We want max set of indices with no collisions
    # This is an approximation - use it as a heuristic
    return len(picked)

def construct():
    start = time.time()
    deadline = start + 75  # safety margin before 90s kill
    rng = random.Random(12345)

    def now():
        return time.time()

    # --- Seed candidate pool (structured + random) ---
    candidates = []

    # 1. Arithmetic progressions
    for a in range(0, 100):
        for d in range(1, 12):
            c = [a + i * d for i in range(12)]
            if all(0 <= x <= 128 for x in c):
                candidates.append(tuple(c))

    # 2. Symmetric pairs around various centers
    for center in range(20, 110):
        for spread in range(1, 15):
            c = []
            for k in range(6):
                a = center - spread * (5 - k)
                b = center + spread * (5 - k)
                if 0 <= a <= 128 and 0 <= b <= 128 and a != b:
                    c.append(a)
                    c.append(b)
            if len(c) == 12 and len(set(c)) == 12:
                candidates.append(tuple(sorted(c)))

    # 3. Clustered sets
    for base in range(0, 100, 3):
        for offset in range(0, 30):
            c = []
            for group in range(4):
                g_base = base + group * 25 + offset
                for k in range(3):
                    val = g_base + k
                    if 0 <= val <= 128 and val not in c:
                        c.append(val)
            if len(c) == 12:
                candidates.append(tuple(sorted(c)))

    # 4. Mixed AP + random
    for _ in range(500):
        a = rng.randint(0, 80)
        d = rng.randint(1, 8)
        c = [a + i * d for i in range(6)]
        remaining = [x for x in range(129) if x not in c]
        c.extend(rng.sample(remaining, 6))
        if all(0 <= x <= 128 for x in c) and len(set(c)) == 12:
            candidates.append(tuple(sorted(c)))

    candidates = list(set(candidates))

    # --- Evaluate structured candidates, track best ---
    best_set = None
    best_size = 12

    for cand in candidates:
        if now() > deadline:
            break
        sz = max_sidon_subset_size(list(cand))
        if sz < best_size:
            best_size = sz
            best_set = list(cand)
            if best_size <= 4:
                break

    # --- Simulated annealing to maximize collision density ---
    # Objective: maximize _collision_density (more collisions => smaller Sidon subsets)
    # We track the best by actual max_sidon_subset_size, but use density as the SA signal.

    def _sa_from(initial, time_budget):
        """Simulated annealing starting from initial set, maximizing collision density."""
        current = sorted(initial)
        current_density = _collision_density(current)
        best_sa_set = list(current)
        best_sa_density = current_density
        best_sa_sidon = max_sidon_subset_size(current)

        T = 10.0
        T_min = 0.01
        cooling = 0.9995

        in_set = set(current)
        eval_counter = 0

        while T > T_min and now() < time_budget:
            # Perturb: pick a random index and a random new value
            idx = rng.randrange(12)
            old_val = current[idx]
            new_val = rng.randrange(129)
            if new_val == old_val or new_val in in_set:
                continue

            current[idx] = new_val
            new_density = _collision_density(current)
            delta = new_density - current_density

            if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                # Accept
                current_density = new_density
                in_set.discard(old_val)
                in_set.add(new_val)
                if new_density > best_sa_density:
                    best_sa_density = new_density
                    best_sa_set = list(current)
                    # Evaluate true Sidon size more frequently when density improves significantly
                    eval_counter += 1
                    if eval_counter % 5 == 0 or delta > 2:
                        if now() < time_budget:
                            sa_sidon = max_sidon_subset_size(current)
                            if sa_sidon < best_sa_sidon:
                                best_sa_sidon = sa_sidon
            else:
                # Reject
                current[idx] = old_val

            T *= cooling

        return best_sa_set, best_sa_sidon

    # Run SA from several diverse starting points
    sa_starts = []
    if best_set is not None:
        sa_starts.append(best_set)

    # Add a few high-density structured starts
    for cand in candidates[:20]:
        sa_starts.append(list(cand))

    # Random starts
    for _ in range(15):
        sa_starts.append(sorted(rng.sample(range(129), 12)))

    # Add starts based on high collision density candidates
    if candidates:
        # Sort candidates by collision density and take top ones
        scored_candidates = [(c, _collision_density(list(c))) for c in candidates[:100]]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        for c, _ in scored_candidates[:10]:
            sa_starts.append(list(c))

    # Deduplicate starts
    seen = set()
    unique_starts = []
    for s in sa_starts:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_starts.append(s)

    # Allocate SA time budget
    sa_budget = deadline - 10  # leave 10s for final polish
    per_sa = max(0.5, (sa_budget - now()) / max(1, len(unique_starts)))

    for init in unique_starts:
        if now() > sa_budget:
            break
        sa_set, sa_sidon = _sa_from(init, min(now() + per_sa, sa_budget))
        if sa_sidon < best_size:
            best_size = sa_sidon
            best_set = sa_set
            if best_size <= 3:
                break

    # --- Final local polish: hill-climb on actual Sidon size ---
    if best_set is not None:
        current = sorted(best_set)
        in_set = set(current)
        polish_deadline = deadline
        no_improve_count = 0

        for _ in range(100):  # max polish iterations
            if now() > polish_deadline:
                break
            improved = False
            for i in range(12):
                if now() > polish_deadline:
                    break
                old_val = current[i]
                # Try values that create more collisions
                for new_val in range(129):
                    if new_val in in_set:
                        continue
                    candidate = list(current)
                    candidate[i] = new_val
                    candidate = sorted(candidate)
                    
                    # Quick check: does this increase collision density?
                    new_density = _collision_density(candidate)
                    old_density = _collision_density(current)
                    
                    # Only do expensive Sidon check if density improved or is equal
                    if new_density > old_density or (new_density == old_density and rng.random() < 0.3):
                        score = max_sidon_subset_size(candidate)
                        if score < best_size:
                            best_size = score
                            best_set = candidate
                            in_set.discard(old_val)
                            in_set.add(new_val)
                            current = candidate
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
            if not improved:
                no_improve_count += 1
                if no_improve_count >= 10:
                    break

    # Fallback
    if best_set is None:
        best_set = list(range(12))
        best_size = max_sidon_subset_size(best_set)

    return sorted(best_set)
