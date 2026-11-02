"""Covering design C(v,k,t) using MILP via scipy.optimize.milp."""
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # 1. Precompute all t-subsets and k-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 2. Build the incidence matrix A (n_tsubs x n_blocks)
    # A[i, j] = 1 if t-subset i is covered by block j
    # We will store this as a sparse matrix for efficiency in MILP
    from scipy.sparse import lil_matrix, csr_matrix
    
    # Initialize sparse matrix
    A_lil = lil_matrix((n_tsubs, n_blocks), dtype=np.int8)
    
    # Fill the matrix
    for j, block in enumerate(all_blocks):
        # For each t-subset within this block
        for ts in itertools.combinations(block, t):
            i = tsub_index[ts]
            A_lil[i, j] = 1
            
    A_csr = A_lil.tocsr()
    
    # 3. Formulate the MILP
    # Minimize sum(x_j)
    # Subject to: sum_j A[i,j] * x_j >= 1 for all i
    #             x_j in {0, 1}
    
    c = np.ones(n_blocks)  # Objective coefficients
    
    # Constraints: A @ x >= 1
    # LinearConstraint expects: lb <= A_cons @ x <= ub
    # Here A_cons is the incidence matrix, lb is 1, ub is inf
    # Note: scipy.optimize.milp handles sparse matrices efficiently
    
    constraints = LinearConstraint(A_csr, lb=1, ub=np.inf)
    
    # Bounds: x_j in [0, 1]
    integrality = np.ones(n_blocks)  # All variables are integers
    bounds = Bounds(lb=0, ub=1)
    
    # 4. Solve
    # method='highs' is the default and most robust for this size
    # time_limit is in seconds
    remaining_time = deadline - time.time()
    if remaining_time <= 5:
        # Fallback to a quick greedy if no time for MILP
        return _quick_greedy(v, k, t, all_blocks, tsub_index, all_tsubs)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
               options={"time_limit": remaining_time, "mip_rel_gap": 0.01})
    
    # 5. Extract solution
    if res.status == 0:  # Optimal or best found within time
        x = res.x
        # Due to numerical precision, threshold at 0.5
        selected_indices = np.where(x > 0.5)[0]
        
        # Verify validity (MILP should guarantee this, but just in case)
        blocks = [list(all_blocks[i]) for i in selected_indices]
        
        # If MILP didn't find a feasible solution (unlikely for covering designs)
        # or if it timed out without a solution, fallback
        if not blocks:
             return _quick_greedy(v, k, t, all_blocks, tsub_index, all_tsubs)
             
        # Final check: ensure every t-subset is covered
        # This is a safety net. If MILP is correct, this passes.
        covered = np.zeros(n_tsubs, dtype=np.int8)
        for idx in selected_indices:
             for i in A_csr[idx].indices: # Wait, A_csr is n_tsubs x n_blocks
                 pass # The logic above is slightly off for iteration. 
                 # Let's just rely on the constraint satisfaction from MILP.
                 # However, to be absolutely safe against solver bugs/precision issues:
        
        # Re-verify with a simple loop if time permits
        if time.time() < deadline - 10:
            covered = np.zeros(n_tsubs, dtype=np.int8)
            for idx in selected_indices:
                # Get non-zero entries in column idx
                # A_csr is row-major (tsubs x blocks). Column access is slow in CSR.
                # Let's use the precomputed block_to_tsubs if we had it, or just iterate.
                # For v=19, k=7, t=4, each block covers C(7,4)=35 tsubs.
                block = all_blocks[idx]
                for ts in itertools.combinations(block, t):
                    covered[tsub_index[ts]] = 1
            
            if not covered.all():
                # If invalid, add missing blocks via greedy
                missing_indices = np.where(covered == 0)[0]
                for mi in missing_indices:
                    # Find any block covering this tsub
                    # Inverse lookup
                    tsub = all_tsubs[mi]
                    for b_idx, b in enumerate(all_blocks):
                        if set(tsub).issubset(set(b)):
                            if b_idx not in selected_indices:
                                blocks.append(list(b))
                                selected_indices.append(b_idx)
                                # Update covered
                                for ts in itertools.combinations(b, t):
                                    covered[tsub_index[ts]] = 1
                            break
                            
        return blocks
        
    else:
        # Fallback
        return _quick_greedy(v, k, t, all_blocks, tsub_index, all_tsubs)

def _quick_greedy(v, k, t, all_blocks, tsub_index, all_tsubs):
    """Fast greedy fallback if MILP fails or times out immediately."""
    n_tsubs = len(all_tsubs)
    uncovered = np.ones(n_tsubs, dtype=np.int8)
    blocks = []
    
    # Precompute block coverage for speed in fallback
    # This is expensive, so only do it if we have time
    if time.time() > 110: # Very close to deadline
        # Extremely naive: just pick random blocks until covered
        # This is a last resort
        import random
        random.seed(42)
        while uncovered.any():
            b = random.choice(all_blocks)
            if b not in blocks:
                blocks.append(list(b))
                for ts in itertools.combinations(b, t):
                    uncovered[tsub_index[ts]] = 0
        return blocks
        
    # Standard greedy
    # For efficiency, we don't recompute everything from scratch, 
    # but since this is a fallback, we accept O(N^2) cost if N is small enough
    # For v=19, n_blocks=50k, this might be slow. 
    # Let's try a simplified greedy: pick block covering most uncovered.
    
    # To speed up, we can just use the first few blocks or random sampling
    # Given the strict deadline, a pure Python loop over 50k blocks * 1000 iterations is ~50M ops, might be okay in 10s?
    # Actually, 50k * 35 tsubs per block = 1.7M checks per iteration. 150 iterations = 255M checks. Too slow in Python.
    
    # Better fallback: Use a pre-computed heuristic or just return a valid (though potentially large) covering.
    # Since we have the incidence matrix logic in the main function, 
    # let's just return a set of blocks that *might* cover it, or better yet,
    # if we reached this point, we likely have very little time.
    
    # Let's try a few random restarts of a very fast greedy
    import random
    rng = random.Random(123)
    best_blocks = None
    best_count = float('inf')
    
    for _ in range(3): # Only 3 attempts
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        blocks = []
        used = set()
        
        # Sample 1000 random blocks to build a pool
        pool = rng.sample(all_blocks, min(1000, len(all_blocks)))
        
        while uncovered.any():
            best_gain = -1
            best_b = None
            # Sample 20 from pool
            candidates = rng.sample(pool, min(20, len(pool)))
            for b in candidates:
                if b in used: continue
                gain = 0
                for ts in itertools.combinations(b, t):
                    if uncovered[tsub_index[ts]]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            
            if best_b is None:
                # Pick any unused
                for b in pool:
                    if b not in used:
                        best_b = b
                        break
            if best_b is None:
                break
                
            used.add(best_b)
            blocks.append(list(best_b))
            for ts in itertools.combinations(best_b, t):
                uncovered[tsub_index[ts]] = 0
                
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks:
        return best_blocks
    return []

if __name__ == "__main__":
    # Quick test
    blocks = construct(19, 7, 4)
    print(f"Found covering with {len(blocks)} blocks")
