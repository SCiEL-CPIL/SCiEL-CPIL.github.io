import itertools
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets (requirements)
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map t-subset to index for quick lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # 3. Build the incidence matrix A (sparse)
    # A[i, j] = 1 if t-subset i is contained in block j
    # We need A @ x >= 1, where x is binary vector of selected blocks
    # Using lil_matrix for efficient row-wise construction
    A = lil_matrix((n_tsubs, n_blocks), dtype=np.int8)
    
    # For each block, find all t-subsets it contains
    for j, block in enumerate(all_blocks):
        # Generate all t-subsets of this block
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                A[idx, j] = 1
    
    A = A.tocsr()
    
    # 4. Formulate Phase 1: Feasibility
    # Minimize sum(x) (dummy objective) subject to A @ x >= 1, 0 <= x <= 1
    # We use a small objective to guide the solver towards fewer blocks initially, 
    # but primarily we want a feasible integer solution quickly.
    # Actually, for Phase 1, we can just minimize sum(x) with a generous time limit 
    # or even just find *any* feasible solution. 
    # However, HiGHS is strong. Let's try to minimize directly but with a 
    # "warm start" mindset or just rely on the solver's speed.
    
    # Let's attempt a single MILP run that minimizes sum(x).
    # If it doesn't finish in time, we need a fallback.
    # Strategy: 
    # 1. Run MILP with objective min sum(x), time limit = deadline - 20s.
    # 2. If optimal or near-optimal found, return it.
    # 3. If not, we need a valid covering. 
    #    HiGHS might return a fractional solution or infeasible if time runs out.
    #    We need a robust fallback: Greedy Set Cover.
    
    # --- Fallback: Greedy Set Cover (fast, guaranteed valid) ---
    def greedy_fallback():
        """Fast greedy set cover to ensure we always return a valid result."""
        remaining = set(range(n_tsubs))
        # Precompute coverage for each block: list of t-subset indices it covers
        block_cov = []
        for j in range(n_blocks):
            # Get row j of A
            # A is CSR, so A[j, :] is not efficient. A[:, j] is column.
            # Since A is built row-wise, let's use the lil structure or just recompute?
            # Actually, let's precompute block_cov properly for the fallback.
            # To save time in main loop, we can compute this on the fly or cache.
            # For simplicity and speed of fallback, let's compute coverage lists.
            pass
            
        # Re-compute coverage lists for fallback (fast enough for v=19)
        block_cov = []
        for j, block in enumerate(all_blocks):
            cov_list = []
            for ts in itertools.combinations(block, t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    cov_list.append(idx)
            block_cov.append(cov_list)
        
        selected = []
        uncovered_mask = np.ones(n_tsubs, dtype=np.bool_)
        
        while uncovered_mask.any():
            # Find block with max gain
            best_j = -1
            best_gain = 0
            # Optimization: Only consider blocks that cover at least one uncovered t-sub
            # Simple loop
            for j in range(n_blocks):
                if not any(uncovered_mask[i] for i in block_cov[j]):
                    continue
                gain = sum(uncovered_mask[i] for i in block_cov[j])
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j == -1:
                break # Should not happen if cover exists
            selected.append(all_blocks[best_j])
            for i in block_cov[best_j]:
                uncovered_mask[i] = False
        
        return selected

    # --- Phase 1: MILP Attempt ---
    # Variables: x_j for each block j, binary
    # Objective: c = ones(n_blocks)
    # Constraints: A @ x >= 1
    # Bounds: 0 <= x <= 1
    # Integrality: 1 for all
    
    c = np.ones(n_blocks)
    lb = np.ones(n_tsubs)
    ub = np.inf * np.ones(n_tsubs)
    
    # LinearConstraint(A, lb, ub)
    cons = LinearConstraint(A, lb, ub)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_blocks)
    
    # Time limit for solver
    solver_time_limit = max(5, int(deadline - start_time - 15))
    
    try:
        res = milp(
            c=c,
            constraints=cons,
            integrality=integrality,
            bounds=bounds,
            options={
                "time_limit": solver_time_limit,
                "mip_rel_gap": 0.0, # Try for optimal
                "mip_abs_gap": 0.0
            }
        )
        
        # Check if we have a feasible integer solution
        # res.x contains the solution. If status is not optimal, it might be fractional or infeasible.
        # We need to round or check validity.
        # HiGHS typically returns integer solutions for MIP if feasible, but if time runs out, 
        # it might return the best incumbent (integer) or a fractional LP relaxation.
        # We must verify validity.
        
        if res.x is not None:
            # Round to nearest integer (should be 0 or 1 for binary)
            x_sol = np.round(res.x).astype(int)
            
            # Verify validity
            # Compute coverage: A @ x_sol >= 1
            coverage = A @ x_sol
            if np.all(coverage >= 1):
                # Valid solution
                selected_indices = np.where(x_sol == 1)[0]
                blocks = [list(all_blocks[i]) for i in selected_indices]
                return blocks
            else:
                # Solution invalid (likely fractional or infeasible cut off)
                pass
    except Exception as e:
        # Solver error or timeout behavior
        pass

    # --- Fallback to Greedy ---
    # If MILP failed or returned invalid, use greedy.
    # We can also try to improve the greedy solution with a simple local search if time permits.
    blocks = greedy_fallback()
    
    # Simple Local Search: Deletion pass
    if time.time() < deadline - 5:
        # Build coverage counts
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_cov_indices = []
        for block in blocks:
            cov_list = []
            for ts in itertools.combinations(block, t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    cov_list.append(idx)
            block_cov_indices.append(cov_list)
            for idx in cov_list:
                cov_count[idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                # If all t-subs in block i have cov_count > 1, we can remove it
                redundant = True
                for idx in block_cov_indices[i]:
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for idx in block_cov_indices[i]:
                        cov_count[idx] -= 1
                    blocks.pop(i)
                    block_cov_indices.pop(i)
                    improved = True
                    break # Restart loop as indices shifted
    
    return blocks
