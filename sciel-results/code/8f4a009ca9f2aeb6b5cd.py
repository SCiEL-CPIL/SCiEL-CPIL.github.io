import itertools
import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # 1. Generate all t-subsets and k-blocks
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # 2. Build sparse incidence matrix A (num_tsubs x num_kblocks)
    row_indices = []
    col_indices = []
    data = []
    
    for j, block in enumerate(all_kblocks):
        for ts in itertools.combinations(block, t):
            i = tsub_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
    
    A = csc_matrix((data, (row_indices, col_indices)), shape=(num_tsubs, num_kblocks))
    
    # Precompute t-subset indices for each block
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsub_idxs.append(idxs)
    
    def greedy_cover(covered, selected_set, x0, time_limit):
        """Continue greedy from current state until time limit."""
        while time.monotonic() < time_limit:
            u = ~covered
            gains = A.T @ u
            best_gain = gains.max()
            if best_gain == 0:
                break
            max_gain_blocks = np.where(gains == best_gain)[0]
            # Pick first max-gain block (deterministic for reproducibility)
            best_idx = int(max_gain_blocks[0])
            if best_idx in selected_set:
                # All max-gain blocks already selected? Shouldn't happen if covered
                break
            selected_set.add(best_idx)
            x0[best_idx] = 1.0
            for idx in block_tsub_idxs[best_idx]:
                covered[idx] = True
            if np.all(covered):
                break
        return selected_set
    
    def polish(selected_set, time_limit):
        """Remove redundant blocks iteratively."""
        if not selected_set:
            return selected_set
        counts = np.zeros(num_tsubs, dtype=int)
        for b_idx in selected_set:
            for idx in block_tsub_idxs[b_idx]:
                counts[idx] += 1
        
        changed = True
        while changed and time.monotonic() < time_limit:
            changed = False
            for b_idx in sorted(selected_set):
                if b_idx not in selected_set:
                    continue
                can_remove = True
                for idx in block_tsub_idxs[b_idx]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    selected_set.discard(b_idx)
                    for idx in block_tsub_idxs[b_idx]:
                        counts[idx] -= 1
                    changed = True
        return selected_set
    
    def ilp_polish(selected_set, time_limit):
        """Use MILP to optimize the covering."""
        remaining = time_limit - time.monotonic()
        if remaining < 8.0 or not selected_set:
            return selected_set
        try:
            c = np.ones(num_kblocks)
            constraints = LinearConstraint(A, 1, np.inf)
            integrality = np.ones(num_kblocks)
            bounds = Bounds(0, 1)
            
            x0 = np.zeros(num_kblocks, dtype=float)
            for b_idx in selected_set:
                x0[b_idx] = 1.0
            
            ilp_time = max(3.0, remaining - 2.0)
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                x0=x0,
                time_limit=ilp_time
            )
            
            if res.x is not None:
                x_sol = np.round(res.x)
                if np.all((x_sol == 0) | (x_sol == 1)):
                    coverage = A @ x_sol
                    if np.all(coverage >= 1):
                        new_set = set(i for i in range(num_kblocks) if x_sol[i] == 1)
                        new_set = polish(new_set, time.monotonic() + 0.1)
                        if len(new_set) < len(selected_set):
                            return new_set
        except Exception:
            pass
        return selected_set
    
    def crossover(cover1, cover2, time_limit):
        """Merge two coverings: take union, then aggressively prune."""
        if not cover1 or not cover2:
            return cover1 if cover1 else cover2
        # Start with union
        merged = set(cover1) | set(cover2)
        # Polish aggressively
        merged = polish(merged, time_limit)
        return merged
    
    def perturb(selected_set, num_kicks, time_limit):
        """Remove random blocks and re-cover to escape local minima."""
        if not selected_set:
            return selected_set
        sel_list = list(selected_set)
        # Remove random blocks
        num_to_remove = min(num_kicks, len(sel_list))
        to_remove = np.random.choice(len(sel_list), size=num_to_remove, replace=False)
        for i in to_remove:
            selected_set.discard(sel_list[i])
        # Recompute covered
        covered = np.zeros(num_tsubs, dtype=bool)
        for b_idx in selected_set:
            for idx in block_tsub_idxs[b_idx]:
                covered[idx] = True
        # Re-cover
        x0 = np.zeros(num_kblocks, dtype=float)
        for b_idx in selected_set:
            x0[b_idx] = 1.0
        greedy_cover(covered, selected_set, x0, time_limit)
        selected_set = polish(selected_set, time_limit)
        return selected_set
    
    # Main search: multiple restarts with ILS
    best_set = None
    best_size = num_kblocks + 1
    
    # Phase 1: Initial greedy + polish
    covered = np.zeros(num_tsubs, dtype=bool)
    selected_set = set()
    x0 = np.zeros(num_kblocks, dtype=float)
    greedy_cover(covered, selected_set, x0, deadline - 60.0)
    selected_set = polish(selected_set, deadline - 50.0)
    # Additional polish pass to ensure local optimality
    if time.monotonic() < deadline - 45.0:
        selected_set = polish(selected_set, deadline - 45.0)
    
    if selected_set:
        best_set = set(selected_set)
        best_size = len(selected_set)
    
    # Phase 2: ILS with perturbations
    current_set = set(selected_set)
    np.random.seed(42)
    
    while time.monotonic() < deadline - 25.0:
        # Perturb current solution
        kick_size = max(1, len(current_set) // 10)
        perturbed = perturb(set(current_set), kick_size, deadline - 20.0)
        
        if perturbed and len(perturbed) < best_size:
            best_size = len(perturbed)
            best_set = set(perturbed)
            current_set = set(perturbed)
        else:
            # Accept with some probability (simulated annealing style)
            if perturbed and np.random.random() < 0.3:
                current_set = perturbed
        
        # Try crossover with best
        if best_set and len(best_set) > 0 and time.monotonic() < deadline - 20.0:
            crossed = crossover(current_set, best_set, deadline - 15.0)
            if crossed and len(crossed) < best_size:
                best_size = len(crossed)
                best_set = set(crossed)
                current_set = set(crossed)
        
        # Periodic ILP polish on current best
        if best_set and time.monotonic() < deadline - 30.0 and len(best_set) > 0:
            polished_best = ilp_polish(set(best_set), deadline - 28.0)
            if polished_best and len(polished_best) < best_size:
                best_size = len(polished_best)
                best_set = set(polished_best)
                current_set = set(polished_best)
    
    # Phase 3: Final ILP polish on best solution
    if best_set:
        best_set = ilp_polish(best_set, deadline - 2.0)
        # Additional polish after ILP
        if time.monotonic() < deadline - 1.0:
            best_set = polish(best_set, deadline - 1.0)
    
    # Final polish
    if best_set:
        best_set = polish(best_set, deadline - 0.5)
    
    selected_blocks = sorted(best_set) if best_set else []
    
    # Convert indices back to lists
    result_blocks = [list(all_kblocks[i]) for i in selected_blocks]
    
    return result_blocks
