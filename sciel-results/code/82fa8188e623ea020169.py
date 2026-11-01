import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)

    # Precompute: for each block, which t-subsets it covers
    # Use a list of lists to avoid zero-padding issues
    block_t_list = []
    for block in all_ks:
        subs = list(itertools.combinations(block, t))
        indices = [idx_map[sub] for sub in subs]
        block_t_list.append(indices)

    # Build incidence matrix: A[i][j] = 1 if block j covers t-subset i
    # This is sparse, so use a list of (row, col) pairs
    from scipy.sparse import csr_matrix
    row_indices = []
    col_indices = []
    for j, block_subs in enumerate(block_t_list):
        for i in block_subs:
            row_indices.append(i)
            col_indices.append(j)
    A = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(n_ts, n_ks))

    best_solution = None
    best_score = float('inf')

    def coverage_count_from_blocks(block_indices):
        """Compute coverage count for a set of blocks."""
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b_idx in block_indices:
            for sub_idx in block_t_list[b_idx]:
                coverage[sub_idx] += 1
        return coverage

    def is_valid(block_indices):
        """Check if all t-subsets are covered."""
        coverage = coverage_count_from_blocks(block_indices)
        return np.all(coverage > 0)

    def aggressive_deletion(current_blocks):
        """Remove redundant blocks."""
        if not current_blocks:
            return current_blocks

        coverage = coverage_count_from_blocks(current_blocks)
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs = block_t_list[b_idx]
                # Check if all t-subsets in this block are covered by other blocks
                can_remove = True
                for sub_idx in subs:
                    if coverage[sub_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    improved = True
                    for sub_idx in subs:
                        coverage[sub_idx] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts

        while remaining > 0:
            if time.time() - start_time > time_limit - 10.0:
                return None

            # Compute gain: number of new t-subsets covered
            best_gain = -1
            best_block = -1
            # Sample a subset of blocks for speed
            if n_ks > 5000:
                sample_size = min(5000, n_ks)
                candidates = random.sample(range(n_ks), sample_size)
            else:
                candidates = range(n_ks)

            for b_idx in candidates:
                subs = block_t_list[b_idx]
                gain = sum(1 for s in subs if not covered[s])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx

            if best_block == -1 or best_gain <= 0:
                break

            blocks_indices.append(best_block)
            for sub_idx in block_t_list[best_block]:
                if not covered[sub_idx]:
                    remaining -= 1
                covered[sub_idx] = True

        # Aggressive deletion
        blocks_indices = aggressive_deletion(blocks_indices)
        return blocks_indices

    # Phase 1: Run multiple greedy restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 40.0:
        seed = restart_count
        blocks = run_greedy(seed)

        if blocks is None:
            break

        if is_valid(blocks) and len(blocks) < best_score:
            best_score = len(blocks)
            best_solution = blocks

        restart_count += 1

    # Phase 2: Use ILP to refine the best solution
    if best_solution is not None and time.time() - start_time < time_limit - 15.0:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds

            # Get the current best solution as initial bounds
            current_blocks = set(best_solution)
            n_current = len(current_blocks)

            # Objective: minimize sum of x_j
            c = np.ones(n_ks)

            # Constraints: A @ x >= 1 for all t-subsets
            # x_j in {0, 1}
            # Use the current solution to set upper bounds: x_j <= 1 if j in current, else 0
            # This restricts the search to the current solution's support plus potential swaps

            # Actually, let's use a different approach: fix the blocks in the current solution
            # and allow the ILP to potentially remove some and add others

            # Better approach: use the current solution to initialize, then let ILP optimize
            # Set bounds: for blocks in current solution, upper bound = 1; for others, upper bound = 1
            # But we want to encourage keeping the current solution, so use a warm start

            # For simplicity, let's just run ILP on the full problem with a time limit
            # The ILP will find the optimal solution within the time limit

            # Define the problem
            # Minimize c^T x
            # Subject to A x >= 1
            # x_j in {0, 1}

            # Convert to standard form for milp
            # milp minimizes c^T x subject to constraints
            # We need A x >= 1, which is -A x <= -1

            # Create constraints
            # A x >= 1  =>  -A x <= -1
            # In scipy.optimize.milp, constraints are of the form lb <= A_eq @ x <= ub
            # or A_ineq @ x <= b_ineq

            # Use LinearConstraint
            # We want A @ x >= 1, which is equivalent to -A @ x <= -1
            # But scipy's LinearConstraint takes lb <= A @ x <= ub

            # So: 1 <= A @ x <= inf
            A_dense = A.toarray()
            constraint = LinearConstraint(A_dense, lb=np.ones(n_ts), ub=np.inf)

            # Integer variables
            integrality = np.ones(n_ks)

            # Bounds: 0 <= x_j <= 1
            lb = np.zeros(n_ks)
            ub = np.ones(n_ks)
            bounds = Bounds(lb, ub)

            # Run ILP with time limit
            time_remaining = time_limit - (time.time() - start_time) - 5.0
            if time_remaining > 2.0:
                result = milp(
                    c=c,
                    constraints=constraint,
                    integrality=integrality,
                    bounds=bounds,
                    options={'time_limit': max(1.0, time_remaining - 2.0)}
                )

                if result.status == 0 and result.x is not None:
                    # Extract solution
                    ilp_blocks = [i for i in range(n_ks) if result.x[i] > 0.5]
                    if is_valid(ilp_blocks) and len(ilp_blocks) < best_score:
                        best_score = len(ilp_blocks)
                        best_solution = ilp_blocks
        except Exception:
            pass

    # Phase 3: Final polish with aggressive deletion
    if best_solution is not None and time.time() - start_time < time_limit - 3.0:
        polished = aggressive_deletion(list(best_solution))
        if is_valid(polished) and len(polished) < best_score:
            best_score = len(polished)
            best_solution = polished

    if best_solution is None:
        # Fallback: run one more greedy
        blocks = run_greedy(0)
        if blocks is not None:
            best_solution = blocks

    # Convert to list of lists
    final_blocks = [list(all_ks[i]) for i in best_solution]

    return final_blocks
