"""Max-SAT style stochastic local search (WalkSAT) for kissing number in dimension 11."""
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    D = 11

    # Generate candidate vectors: weight-4 (norm^2=4) and weight-2 (norm^2=2)
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5280 + 220 = 5500

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (dot > threshold).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # For each vertex, precompute its conflict set as a bitset-like array
    # conflict[i] is a (n,) uint8 array
    # degree[i] = number of conflicts
    degree = conflict.sum(axis=1).astype(np.int32)

    # WalkSAT-style local search for maximum independent set
    # Variables: x[i] = 1 if vertex i is selected
    # Clauses: for each conflicting pair (i,j), NOT(x[i] AND x[j])
    # Objective: maximize sum(x[i])

    def evaluate_solution(x):
        """Return (num_selected, num_violations)."""
        sel = np.where(x == 1)[0]
        if len(sel) == 0:
            return 0, 0
        # Count violations: pairs of selected vertices that conflict
        sub_conflict = conflict[np.ix_(sel, sel)]
        violations = int(np.triu(sub_conflict, 1).sum())
        return len(sel), violations

    def walk_sat_search(x_init, time_limit, p_flip=0.3, max_steps=500000):
        """WalkSAT: flip variables to reduce violations, maximize selected count."""
        x = x_init.copy()
        sel, viol = evaluate_solution(x)
        best_x = x.copy()
        best_score = sel
        best_viol = viol

        step = 0
        while time.time() < time_limit and step < max_steps:
            step += 1

            if viol == 0:
                # No violations: try to add more vertices
                # Find vertices not selected that are compatible with all selected
                if sel == 0:
                    # Add a random vertex
                    idx = np.random.randint(n)
                    x[idx] = 1
                    sel += 1
                    # Check for new violations
                    new_viol = int(conflict[idx].sum()) - 1  # exclude self
                    # Actually need to check against selected only
                    # Recompute
                    sel_arr = np.where(x == 1)[0]
                    sub = conflict[np.ix_(sel_arr, sel_arr)]
                    viol = int(np.triu(sub, 1).sum())
                else:
                    sel_arr = np.where(x == 1)[0]
                    # Candidates: not selected, compatible with all selected
                    # A vertex v is compatible with all selected iff conflict[v, sel_arr].sum() == 0
                    # Vectorized: for each v, count conflicts with selected
                    conflict_with_sel = conflict[:, sel_arr].sum(axis=1)
                    addable = np.where((conflict_with_sel == 0) & (x == 0))[0]
                    if len(addable) > 0:
                        # Pick one with minimum degree (least constraining)
                        idx = addable[np.argmin(degree[addable])]
                        x[idx] = 1
                        sel += 1
                        # Recompute violations (should still be 0)
                        sel_arr2 = np.where(x == 1)[0]
                        sub = conflict[np.ix_(sel_arr2, sel_arr2)]
                        viol = int(np.triu(sub, 1).sum())
                    else:
                        # No more can be added without violation; try a random flip
                        # Flip a random selected vertex to 0, then try to add
                        if sel > 0:
                            flip_idx = sel_arr[np.random.randint(len(sel_arr))]
                            x[flip_idx] = 0
                            sel -= 1
                            sel_arr3 = np.where(x == 1)[0]
                            if len(sel_arr3) > 0:
                                sub = conflict[np.ix_(sel_arr3, sel_arr3)]
                                viol = int(np.triu(sub, 1).sum())
                            else:
                                viol = 0
                        else:
                            break
            else:
                # Has violations: try to fix
                # Find all violating pairs
                sel_arr = np.where(x == 1)[0]
                if len(sel_arr) == 0:
                    viol = 0
                    continue

                sub = conflict[np.ix_(sel_arr, sel_arr)]
                triu = np.triu(sub, 1)
                viol_pairs = np.where(triu > 0)
                if len(viol_pairs[0]) == 0:
                    viol = 0
                    continue

                viol = len(viol_pairs[0])

                # With probability p_flip, do a random walk (flip a random selected vertex)
                # Otherwise, pick a violating pair and flip one of them
                if np.random.random() < p_flip:
                    # Random flip: unselect a random selected vertex
                    flip_idx = sel_arr[np.random.randint(len(sel_arr))]
                    x[flip_idx] = 0
                    sel -= 1
                    # Recompute violations
                    sel_arr2 = np.where(x == 1)[0]
                    if len(sel_arr2) > 0:
                        sub2 = conflict[np.ix_(sel_arr2, sel_arr2)]
                        viol = int(np.triu(sub2, 1).sum())
                    else:
                        viol = 0
                else:
                    # Pick a random violating pair, flip one of the two
                    pi = np.random.randint(len(viol_pairs[0]))
                    i_local = viol_pairs[0][pi]
                    j_local = viol_pairs[1][pi]
                    i_global = sel_arr[i_local]
                    j_global = sel_arr[j_local]
                    # Flip the one with higher degree (less likely to cause new conflicts)
                    if degree[i_global] >= degree[j_global]:
                        flip_idx = i_global
                    else:
                        flip_idx = j_global
                    x[flip_idx] = 0
                    sel -= 1
                    # Recompute violations
                    sel_arr2 = np.where(x == 1)[0]
                    if len(sel_arr2) > 0:
                        sub2 = conflict[np.ix_(sel_arr2, sel_arr2)]
                        viol = int(np.triu(sub2, 1).sum())
                    else:
                        viol = 0

            # Track best
            if viol == 0 and sel > best_score:
                best_score = sel
                best_viol = 0
                best_x = x.copy()
            elif viol < best_viol and sel >= best_score:
                best_score = sel
                best_viol = viol
                best_x = x.copy()
            elif viol == best_viol and sel > best_score:
                best_score = sel
                best_x = x.copy()

        return best_x, best_score

    # Initial solution: greedy min-degree
    def greedy_min_degree(time_limit):
        remaining = np.ones(n, dtype=np.uint8)
        selected = []
        while time.time() < time_limit and remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        x = np.zeros(n, dtype=np.uint8)
        for s in selected:
            x[s] = 1
        return x, len(selected)

    # Run greedy to get initial solution
    x_init, greedy_count = greedy_min_degree(deadline - 30.0)
    best_x = x_init.copy()
    best_count = greedy_count

    # Run WalkSAT with multiple restarts
    np.random.seed(42)
    num_restarts = 8
    time_per_restart = (deadline - start_time - 10.0) / num_restarts

    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        # Create initial solution: start from greedy, then randomly perturb
        x_r = x_init.copy()
        # Randomly flip some variables
        num_flips = max(1, int(n * 0.05))
        flip_indices = np.random.choice(n, size=num_flips, replace=False)
        x_r[flip_indices] = 1 - x_r[flip_indices]

        # Run WalkSAT
        x_best, count = walk_sat_search(x_r, deadline - 5.0, p_flip=0.3, max_steps=200000)
        if count > best_count:
            best_count = count
            best_x = x_best

    # Final greedy fill on the best solution
    if time.time() < deadline - 2.0:
        sel_arr = np.where(best_x == 1)[0]
        if len(sel_arr) > 0:
            conflict_with_sel = conflict[:, sel_arr].sum(axis=1)
            addable = np.where((conflict_with_sel == 0) & (best_x == 0))[0]
            for idx in addable[np.argsort(degree[addable])]:
                if time.time() > deadline - 2.0:
                    break
                best_x[idx] = 1
                sel_arr = np.where(best_x == 1)[0]
                conflict_with_sel = conflict[:, sel_arr].sum(axis=1)
                addable = np.where((conflict_with_sel == 0) & (best_x == 0))[0]
        else:
            # No selected, add greedily
            remaining = np.ones(n, dtype=np.uint8)
            while time.time() < deadline - 2.0 and remaining.any():
                deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
                deg[remaining == 0] = n + 1
                min_deg = deg.min()
                if min_deg >= n:
                    break
                mask = (deg == min_deg) & (remaining > 0)
                picks = np.where(mask)[0]
                if len(picks) == 0:
                    break
                new_idx = picks[0]
                best_x[new_idx] = 1
                remaining = remaining & (1 - conflict[new_idx])
                remaining[new_idx] = 0

    # Extract result
    sel_indices = np.where(best_x == 1)[0]
    if len(sel_indices) == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[sel_indices.astype(np.int64)]
    return result.astype(np.int64)
