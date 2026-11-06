import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Helper to generate Polarity Graph of PG(2, q)
    def get_polarity_graph(q):
        # Points are non-zero vectors in F_q^3, normalized
        points = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv_x = pow(x, -1, q)
                        points.append((1, (y * inv_x) % q, (z * inv_x) % q))
                    elif y != 0:
                        inv_y = pow(y, -1, q)
                        points.append((0, 1, (z * inv_y) % q))
                    else:
                        inv_z = pow(z, -1, q)
                        points.append((0, 0, 1))
        
        unique_points = list(set(points))
        unique_points.sort()
        n_pts = len(unique_points)
        
        adj = np.zeros((n_pts, n_pts), dtype=np.int8)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                u = unique_points[i]
                v = unique_points[j]
                dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
                if dot == 0:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj, n_pts

    adj_full, n_full = get_polarity_graph(7)
    # n_full should be 57
    # Check degrees
    degs_full = adj_full.sum(axis=1)
    
    # Precompute neighbors for faster access
    neighbors = [np.where(adj_full[i] == 1)[0] for i in range(n_full)]

    def try_construct_from_subset(n_target, max_nodes=500000):
        if n_full < n_target:
            return None
        
        target_deg = n_target - 39
        num_remove = n_full - n_target
        
        # limits[v] = max neighbors of v that can be in R
        limits = degs_full - target_deg
        
        # If any limit is negative, impossible
        if np.any(limits < 0):
            return None
            
        # Backtracking Search
        # State:
        # in_R: boolean array, True if vertex is in R
        # counts: array, counts[v] = number of neighbors of v in R
        # candidates: list of vertices not yet decided (neither in R nor fixed out)
        # Actually, simpler: iterate through vertices, decide if in R or not.
        # But we must pick exactly num_remove vertices.
        
        in_R = np.zeros(n_full, dtype=bool)
        counts = np.zeros(n_full, dtype=np.int8)
        remaining_to_remove = num_remove
        nodes_visited = 0
        
        # Order vertices by degree descending? Or just natural order?
        # Natural order is fine, but heuristic might help.
        # Let's try ordering by degree descending for removal priority?
        # No, let's just use natural order but prune aggressively.
        
        def backtrack(idx):
            nonlocal nodes_visited
            if time.time() > deadline:
                return False
            
            if idx == n_full:
                return remaining_to_remove == 0
            
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return False
            
            v = idx
            
            # Option 1: Put v in R
            if remaining_to_remove > 0:
                # Check if we can put v in R
                # This increases counts[u] for all u in neighbors[v]
                # We need counts[u] + 1 <= limits[u] for all u in neighbors[v] that are NOT in R yet?
                # No, limits apply to final counts.
                # If u is in R, its count doesn't matter for the constraint on u (since u is removed).
                # But wait, the constraint is: for all v NOT in R, deg_S(v) >= target_deg.
                # deg_S(v) = deg_full(v) - counts[v].
                # So for v not in R, counts[v] <= limits[v].
                # If we put v in R, we don't care about counts[v] anymore.
                # But we DO care about counts[u] for neighbors u that might end up NOT in R.
                
                # However, a necessary condition for adding v to R is:
                # For any neighbor u, if u is forced to be NOT in R (or just currently not in R),
                # adding v to R increases counts[u].
                # If counts[u] + 1 > limits[u] AND u is not in R, can we still save it?
                # Only if we put u in R later. But if u is in R, the constraint on u doesn't apply.
                # So, if counts[u] + 1 > limits[u], we MUST put u in R.
                
                # Simple check: Can we add v to R without immediately violating a constraint for a vertex that CANNOT be put in R?
                # Vertices that cannot be put in R:
                # 1. If remaining_to_remove == 0, no one else can be added.
                # 2. If a vertex u has counts[u] == limits[u] + 1, it's already invalid.
                
                # Let's do a simple check:
                # For each neighbor u of v:
                # If u is not in R, and counts[u] + 1 > limits[u], then u MUST be in R.
                # If u MUST be in R, but we don't have enough slots or u is already "blocked"?
                
                # Actually, let's just try adding v to R and see if we can propagate.
                
                # Tentatively add v to R
                in_R[v] = True
                remaining_to_remove -= 1
                added_to_counts = []
                for u in neighbors[v]:
                    if not in_R[u]:
                        counts[u] += 1
                        added_to_counts.append(u)
                        # If counts[u] > limits[u], then u MUST be in R.
                        # This is a constraint propagation step.
                        # If we have many such u, we might run out of slots.
                
                # Check if any neighbor u (not in R) has counts[u] > limits[u]
                # If so, that u MUST be in R.
                # If we have already used all removal slots, we fail.
                # If there are too many such u, we fail.
                
                must_be_in_R = []
                for u in added_to_counts:
                    if counts[u] > limits[u]:
                        must_be_in_R.append(u)
                        
                if len(must_be_in_R) > remaining_to_remove:
                    # Not enough slots to remove all forced vertices
                    # Undo and backtrack
                    in_R[v] = False
                    remaining_to_remove += 1
                    for u in added_to_counts:
                        counts[u] -= 1
                    return False
                
                # We could recursively force these, but for simplicity, let's just check if it's a dead end immediately.
                # If we have forced vertices, we must pick them in future steps.
                # If we pick them, it's fine.
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                in_R[v] = False
                remaining_to_remove += 1
                for u in added_to_counts:
                    counts[u] -= 1
            
            # Option 2: Do NOT put v in R
            # This is always valid locally, as long as we can still remove enough vertices from the rest.
            # We have n_full - (idx + 1) vertices left.
            # We need to remove remaining_to_remove vertices from them.
            if n_full - (idx + 1) >= remaining_to_remove:
                if backtrack(idx + 1):
                    return True
            
            return False

        if backtrack(0):
            # Verify and construct result
            S = [i for i in range(n_full) if not in_R[i]]
            if len(S) != n_target:
                return None
            
            valid = True
            for v in S:
                if degs_full[v] - counts[v] < target_deg:
                    valid = False
                    break
            
            if valid:
                edges = []
                old_to_new = {old: new for new, old in enumerate(S)}
                for new_i, old_i in enumerate(S):
                    for new_j in range(new_i + 1, n_target):
                        old_j = S[new_j]
                        if adj_full[old_i][old_j]:
                            edges.append([new_i, new_j])
                return {'n': n_target, 'edges': edges}
        
        return None

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        res = try_construct_from_subset(n)
        if res:
            return res
    
    # Fallback: Random Greedy Construction for lower n
    # If polarity subset fails for all, try random construction for n=40, 39
    for n in range(40, 38, -1):
        if time.time() > deadline:
            break
        target_deg = n - 39
        adj = np.zeros((n, n), dtype=np.int8)
        degs = np.zeros(n, dtype=np.int8)
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check C4
            common = np.any(adj[u] & adj[v])
            if not common:
                adj[u][v] = 1
                adj[v][u] = 1
                degs[u] += 1
                degs[v] += 1
        
        if np.min(degs) >= target_deg:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}

    # Final fallback: n=39, empty graph
    return {'n': 39, 'edges': []}
