import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580.0  # safety margin before 630s kill

    # ---------- Base: all weight-2 vectors (220 points, known valid) ----------
    base_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    base_list.append(p)
    base_points = np.array(base_list, dtype=np.int64)
    best_points = base_points.copy()
    best_k = base_points.shape[0]  # 220

    # ---------- Candidate generation for higher norms ----------
    # We generate candidates with squared norms up to 12 (weight 1 to 6, but restricted to avoid explosion)
    # Norm 2: 220
    # Norm 4: 1540 (weight 2 with signs? No, weight 2 is norm 2. Weight 3 is norm 3? No, entries are +/-1.
    # Wait, the problem says "Integer kissing configuration". The points are in Z^11.
    # The current program uses entries -1, 0, 1.
    # Let's stick to entries in {-1, 0, 1} to keep the search space manageable but larger.
    # Weight w means w non-zero entries. Norm squared is w.
    # w=1: 22
    # w=2: 220
    # w=3: 1540
    # w=4: 6820
    # w=5: 21552
    # w=6: 51952
    # Total up to w=6 is ~81k. This is large but manageable with bitmasks if we are careful.
    # However, 81k points means a conflict graph of 81k nodes.
    # MIS on 81k nodes is hard. But we can use a greedy coloring bound or branch and bound with heuristics.
    
    # Let's try a slightly smaller pool first to ensure we finish, then expand if time allows.
    # Pool: w=1, 2, 3, 4. Total ~8600 points.
    # This is small enough for a decent B&B or even a good greedy + local search to find a large set.
    # But the current program already does w=1..4 and gets 208.
    # The insight says "Using norm 4 vectors increases... estimated >500".
    # The current program DOES use norm 4 (w=4). Why did it only get 208?
    # Maybe the greedy order was bad, or the local search was weak.
    # MIS is a different optimization method.
    
    def generate_candidates(max_weight):
        cands = []
        for w in range(1, max_weight + 1):
            for combo in combinations(range(d), w):
                for mask in range(1 << w):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, pos in enumerate(combo):
                        v[pos] = 1 if (mask >> idx) & 1 else -1
                    cands.append(v)
        return np.array(cands, dtype=np.int64)

    # We will use a pool up to weight 4 (norm^2 = 4) initially, and maybe 5 if fast.
    # Let's use weight 4 pool: ~8600 points.
    cands = generate_candidates(4)
    n = cands.shape[0]
    norms = (cands * cands).sum(axis=1).astype(np.int64)

    # Precompute pairwise squared distances and check validity
    # dist^2(i, j) = norms[i] + norms[j] - 2 * dot(i, j)
    # Valid if dist^2(i, j) >= max(norms[i], norms[j]) for all i != j.
    # Note: If norms are equal (common), condition is dist^2 >= norm.
    # dist^2 = 2*norm - 2*dot >= norm  =>  norm >= 2*dot.
    # If norms differ, say n1 < n2, condition is dist^2 >= n2.
    # n1 + n2 - 2*dot >= n2 => n1 >= 2*dot.
    
    # Build Conflict Graph
    # Two points conflict if they CANNOT be in the same set.
    # Condition for compatibility: dist^2(i, j) >= max(norms[i], norms[j])
    # Conflict if dist^2(i, j) < max(norms[i], norms[j])
    
    # For speed, we can compute the adjacency matrix or list.
    # With n=8600, n^2 is ~74M. This might be slow in Python loops.
    # Vectorized numpy is better.
    
    # Let's compute the conflict matrix in blocks to save memory if needed, or just do it all at once if memory allows.
    # 8600x8600 bool matrix is ~74MB. Acceptable.
    
    # Compute dots
    dots = cands @ cands.T  # (n, n)
    # dist_sq = norms[:, None] + norms[None, :] - 2 * dots
    dist_sq = norms[:, None] + norms[None, :] - 2 * dots
    
    # Required distance squared for pair (i, j) is max(norms[i], norms[j])
    required = np.maximum(norms[:, None], norms[None, :])
    
    # Conflict if dist_sq < required
    conflict = dist_sq < required
    
    # Diagonal is always "conflict" with self (dist 0 < norm > 0), which is fine.
    # We want the INDEPENDENT set, i.e., no conflicts.
    # So we want a set of indices such that conflict[i, j] is False for all i != j in the set.
    
    # Solve Maximum Independent Set (MIS)
    # Since n is ~8600, exact MIS is likely too slow for 600s if the graph is dense.
    # However, the graph of "compatible" points is likely sparse?
    # Let's check density.
    # For random high-dim vectors, dots are small relative to norms.
    # dist^2 is large. So most pairs are compatible.
    # This means the Conflict Graph is SPARSE.
    # The Independent Set is the Clique in the Compatibility Graph.
    # The Compatibility Graph is DENSE.
    # Finding a Max Clique in a dense graph is also hard, but greedy heuristics work well on dense graphs.
    
    # Actually, if the Compatibility Graph is dense, the MIS in the Conflict Graph is large.
    # Let's use a greedy algorithm with a twist: 
    # 1. Order nodes by degree in Conflict Graph (ascending, i.e., most compatible first).
    # 2. Greedily add nodes that don't conflict with already selected.
    # 3. Local search: try to swap.
    
    # But the directive asks for a DIFFERENT strategy. MIS via B&B is one.
    # Given the dense compatibility, a simple greedy might already get close to optimal.
    # Let's try a more robust greedy:
    # "Maximal Independent Set" via random restarts and different orderings.
    
    # To make it "MIS-like" and potentially better than simple greedy:
    # We can use a Branch and Bound with a simple upper bound (e.g., number of remaining nodes).
    # But with n=8600, B&B might explode.
    
    # Alternative: Use the structure.
    # The points are vectors in {-1,0,1}^11.
    # Compatibility condition: dist^2 >= max(norm_i, norm_j).
    
    # Let's try a different approach: **Simulated Annealing** or **Tabu Search** on the set of points.
    # Start with a greedy solution.
    # Move: Remove a point, add a different point.
    # Or: Swap a point.
    
    # Let's stick to the "MIS" framing but implement a **Greedy Coloring based B&B** or just a **Very Good Greedy + Local Search** that is significantly improved over the current one by using the full conflict matrix and better heuristics.
    
    # Actually, the current program's local search was weak.
    # Let's implement a **Maximal Independent Set** finder with **Randomized Greedy** and **Local Search (2-opt style swaps)**.
    
    # Step 1: Compute conflict list for each node to speed up checks.
    # conflict[i] is a boolean array of length n.
    # Memory: 8600 * 8600 bytes = 74MB. OK.
    
    # Greedy with ordering by "compatibility score" (number of compatible neighbors).
    # deg_compat[i] = sum(~conflict[i]) - 1 (excluding self)
    
    deg_compat = np.sum(~conflict, axis=1) - 1  # Exclude self
    
    # Order by descending compatibility (most connected in compat graph)
    order = np.argsort(-deg_compat)
    
    selected = []
    selected_set = set()
    # For fast lookup, we can maintain a boolean array 'in_selected'
    in_selected = np.zeros(n, dtype=bool)
    
    # Greedy pass
    for idx in order:
        if time.time() > deadline - 100:
            break
        # Check if idx conflicts with any selected
        # conflict[idx, selected]
        # If any is True, skip.
        if len(selected) == 0:
            selected.append(idx)
            in_selected[idx] = True
            continue
            
        # Vectorized check
        # We need to check conflict[idx, selected_indices]
        # This is a slice of the conflict matrix.
        # To avoid creating a new array each time, we can just check the dot product or sum.
        # if np.any(conflict[idx, selected]): continue
        
        # Optimization: Keep a running "forbidden" mask?
        # No, just check against current selected.
        # Since selected grows, this is O(k) per node.
        
        if np.any(conflict[idx, selected]):
            continue
            
        selected.append(idx)
        in_selected[idx] = True

    best_k = len(selected)
    best_points = cands[selected].copy()

    # Local Search: Try to improve
    # We can try to remove a point and see if we can add two points that were previously blocked.
    # Or simply try to add any unselected point that fits (shouldn't exist if greedy was maximal).
    # Since greedy produces a Maximal Independent Set (MIS), no single point can be added.
    # So we must swap.
    
    # Local Search:
    # For each selected point i, remove it.
    # Then try to add as many unselected points as possible (greedy again on the remaining).
    # If the size increases, keep the change.
    
    # This is expensive. Let's do a limited version.
    # Pick a random selected point, remove it, try to add a random unselected point that now fits.
    
    rng = random.Random(42)
    
    for _ in range(50): # 50 local search steps
        if time.time() > deadline - 50:
            break
            
        # Pick a random selected point to remove
        if not selected:
            break
        rem_idx = rng.choice(selected)
        rem_pos = selected.index(rem_idx)
        selected.pop(rem_pos)
        in_selected[rem_idx] = False
        
        # Try to add a random unselected point
        # Candidates are those not in selected and not conflicting with new selected
        # To be fast, pick a random unselected index
        unselected_indices = np.where(~in_selected)[0]
        if len(unselected_indices) == 0:
            # Revert
            selected.append(rem_idx)
            in_selected[rem_idx] = True
            break
            
        # Try a few random candidates
        added = False
        for _ in range(20):
            cand_idx = rng.choice(unselected_indices)
            # Check conflict with current selected
            if not np.any(conflict[cand_idx, selected]):
                selected.append(cand_idx)
                in_selected[cand_idx] = True
                added = True
                break
        
        if not added:
            # Revert removal
            selected.append(rem_idx)
            in_selected[rem_idx] = True
        else:
            # We removed 1 and added 1. Size is same.
            # We need to try to add MORE.
            # Now that one point is removed, maybe multiple new points fit?
            # Let's do a greedy pass on the "newly available" space.
            # Actually, just run a quick greedy on all unselected to see if we can add more.
            # This is O(n*k).
            for u_idx in unselected_indices:
                if in_selected[u_idx]:
                    continue
                if not np.any(conflict[u_idx, selected]):
                    selected.append(u_idx)
                    in_selected[u_idx] = True
            
            if len(selected) > best_k:
                best_k = len(selected)
                best_points = cands[selected].copy()

    # Try a few different random restarts of the greedy if time permits
    if time.time() < deadline - 100:
        for restart in range(10):
            if time.time() > deadline - 100:
                break
            
            # Random ordering
            order = np.random.permutation(n)
            selected = []
            in_selected = np.zeros(n, dtype=bool)
            
            for idx in order:
                if time.time() > deadline - 100:
                    break
                if not selected:
                    selected.append(idx)
                    in_selected[idx] = True
                    continue
                if np.any(conflict[idx, selected]):
                    continue
                selected.append(idx)
                in_selected[idx] = True
                
            if len(selected) > best_k:
                best_k = len(selected)
                best_points = cands[selected].copy()

    # Final safety
    if best_points is None or best_points.shape[0] < 2:
        best_points = base_points
        best_k = base_points.shape[0]

    return np.array(best_points, dtype=np.int64)
