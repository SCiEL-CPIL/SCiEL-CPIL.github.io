"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Strategy: Simulated Annealing on a mixed pool of vectors.
    # We mix shells m=2, m=3, m=4 to allow diverse norms and angles.
    # Constraint: min pairwise distance >= max norm.
    # Equivalent: ||u - v||^2 >= M^2 for all pairs, where M = max ||w||.
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # So we need: ||u||^2 + ||v||^2 - 2 u.v >= M^2
    # Rearranging: 2 u.v <= ||u||^2 + ||v||^2 - M^2
    
    # Generate candidate pool
    # m=2: 220 vectors, norm^2 = 2
    # m=3: 11 * 8 = 88 vectors? No, C(11,3)*2^3 = 165 * 8 = 1320 vectors, norm^2 = 3
    # m=4: 5280 vectors, norm^2 = 4
    
    # To keep search space manageable for SA, we might subsample m=4 or use all.
    # Let's start with m=2 and m=3 and a subset of m=4? 
    # Actually, m=4 vectors are "farther apart" in norm, allowing more packing if angles permit.
    # Let's try a pool of m=3 and m=4.
    # m=3: 1320 vectors
    # m=4: 5280 vectors
    # Total ~6600 candidates.
    
    # Let's include m=2 as well, just in case.
    # m=2: 220
    # Total ~6820.
    
    shell2 = _generate_shell(2)
    shell3 = _generate_shell(3)
    shell4 = _generate_shell(4)
    
    candidates = np.vstack([shell2, shell3, shell4])
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Initialize state
    # Start with a greedy solution to have a good starting point
    # Greedy: sort by norm desc? Or just random shuffle and pick valid ones.
    # Let's use a fast greedy first.
    
    order = np.random.permutation(n_cand)
    selected_indices = []
    max_ip = np.full(n_cand, -np.inf)
    
    for idx in order:
        # Check against current max constraints
        # We need to check if this vector can be added to the current set.
        # Constraint for new vector v against existing set S:
        # For all u in S: ||u||^2 + ||v||^2 - 2 u.v >= M_S^2 (where M_S is current max norm)
        # AND ||v||^2 <= new_M^2 (if v becomes new max)
        
        # This check is O(|S|) per candidate, which is slow if we do it naively for all 6000.
        # Let's just pick a random subset for initialization or a very simple greedy.
        # Simple Greedy:
        # Maintain current max norm M_sq.
        # Maintain max inner product with any selected vector? No, constraint depends on M_sq.
        
        # Let's just initialize with a small random valid set.
        # Or better: run a few iterations of a simple greedy to get a decent start.
        
        if len(selected_indices) == 0:
            selected_indices.append(idx)
            continue
            
        # Check compatibility with all selected
        # This is too slow for 6000 candidates * 100 selected.
        # Let's just pick 50 random vectors and check if valid, if not, add one by one.
        
        # Alternative: Start empty, add random candidates if valid.
        # To speed up, we can precompute dot products? 6000x6000 matrix is 36M entries. 
        # 36M * 8 bytes = 288 MB. This is feasible in memory.
        
    # Precompute full dot product matrix?
    # 6820 x 6820 float64 is ~370 MB. Might be tight but okay.
    # Let's try to be more memory efficient or use a smaller pool.
    # Let's restrict pool to m=3 and m=4 to reduce size?
    # m=3 (1320) + m=4 (5280) = 6600.
    # 6600^2 * 8 = 348 MB.
    # Let's try.
    
    # Actually, let's just use m=4 and m=3.
    # If memory is an issue, we can subsample m=4.
    
    # Let's stick to a smaller pool to be safe: m=3 and a subsample of m=4?
    # No, let's use m=2, m=3, m=4 but compute dots on the fly in batches?
    # SA needs to evaluate moves quickly.
    
    # Let's try a different approach: 
    # Use a smaller candidate pool: m=3 (1320) and m=4 (5280).
    # Total 6600.
    # Precompute dots: 6600 x 6600.
    
    try:
        dots = candidates @ candidates.T
    except MemoryError:
        # Fallback to smaller pool
        candidates = np.vstack([shell3, shell4[:2000]])
        n_cand = len(candidates)
        norms_sq = np.sum(candidates**2, axis=1)
        dots = candidates @ candidates.T

    # State: selected_indices (list of indices into candidates)
    # We want to maximize k.
    
    # Initialize with a greedy set
    # Sort candidates by norm descending? High norm vectors are "heavier" constraints.
    # Let's just shuffle and greedily add if valid.
    
    np.random.shuffle(order)
    selected = []
    
    # To make greedy faster, we can check against current selected set.
    # Since selected set is small initially, this is fast.
    
    for idx in order:
        if len(selected) == 0:
            selected.append(idx)
            continue
        
        # Check if idx is compatible with all in selected
        # Constraint: ||u - v||^2 >= M^2
        # M^2 = max(norms_sq[selected] + norms_sq[idx])? No, M is global max.
        # When adding idx, new M_sq = max(old_M_sq, norms_sq[idx])
        # We need to check all pairs (u, idx) for u in selected.
        # And also, if norms_sq[idx] > old_M_sq, we must re-check all pairs in selected?
        # Yes, because the constraint for existing pairs (u, w) becomes stricter if M increases.
        # This makes greedy addition complex.
        
        # Simplification: Only add vectors with the SAME max norm as the current max?
        # Or start with max norm vectors and only add those?
        # If we restrict to a single shell (e.g., m=4), M is constant.
        # This simplifies the constraint to: u.v <= (4+4-4)/2 = 2.
        # This is much easier to maintain.
        
        # Let's restart strategy:
        # 1. Pick a target norm shell, e.g., m=4.
        # 2. Find max subset of that shell with pairwise dot <= 2.
        # 3. Then try to add m=3 vectors?
        #    If we add m=3 (norm^2=3), new M^2 is still 4 (since 4>3).
        #    Constraint for u in S4, v in S3: 4+3-2u.v >= 4 => 2u.v <= 3 => u.v <= 1.5 => u.v <= 1 (integers).
        #    Constraint for u, v in S3: 3+3-2u.v >= 4 => 2u.v <= 2 => u.v <= 1.
        #    Constraint for u in S4, v in S4: u.v <= 2.
        
        # This looks promising.
        # Let's implement SA on the combined set, but with careful constraint checking.
        
        pass

    # Let's implement a simpler SA on the m=4 shell first, then extend.
    # Or just run SA on the mixed pool with a robust cost function.
    
    # Cost Function for a set S:
    # Let M_sq = max(norms_sq[S])
    # Violation = max_{u,v in S} (M_sq - (||u||^2 + ||v||^2 - 2 u.v))
    # We want Violation <= 0.
    # Score = -max(0, Violation) + lambda * |S|
    
    # To make SA fast, we need to update cost incrementally.
    # Move: Swap one vector out, one in.
    
    # Let's try a simpler approach: 
    # 1. Greedy on m=4 to get a large core.
    # 2. Use SA to swap vectors within m=4 and add/remove m=3, m=2.
    
    # Step 1: Greedy on m=4
    # Pool: shell4
    # Constraint: dot <= 2
    
    idx4 = np.arange(len(shell4))
    np.random.shuffle(idx4)
    
    core_indices = [] # indices into shell4
    max_dots = np.full(len(shell4), -np.inf)
    
    for i in idx4:
        if max_dots[i] > 2:
            continue
        core_indices.append(i)
        max_dots[i] = np.inf # mark as selected
        # Update others
        dots_i = shell4[i] @ shell4.T
        max_dots = np.maximum(max_dots, dots_i)
        
    # Now we have a core set of m=4 vectors.
    # Let's convert to global indices for the mixed pool.
    # shell2 (0..219), shell3 (220..1539), shell4 (1540..6819)
    
    # Map core_indices to global indices
    offset4 = len(shell2) + len(shell3)
    global_core = [i + offset4 for i in core_indices]
    
    # Now, let's try to add m=3 and m=2 vectors using SA or greedy.
    # Let's try a local search / SA on the global set.
    
    # State: set of global indices.
    # We will perform random swaps and additions.
    
    selected_set = set(global_core)
    
    # Precompute norms for global pool
    # We already have candidates and norms_sq.
    
    # Helper to check validity of a set
    def get_violation(indices):
        if len(indices) < 2:
            return 0
        idxs = np.array(indices)
        norms_idx = norms_sq[idxs]
        M_sq = np.max(norms_idx)
        # Dots among selected
        # This is O(k^2). If k is ~100, 10k ops. Fast.
        sub_dots = dots[np.ix_(idxs, idxs)]
        # Compute pairwise distance squared
        # dist_sq[i, j] = norms[i] + norms[j] - 2 * sub_dots[i, j]
        # We need min dist_sq >= M_sq
        # So we need: norms[i] + norms[j] - 2 * sub_dots[i, j] >= M_sq
        # Violation = max(0, M_sq - min(dist_sq))
        
        # Efficiently:
        # dist_sq = norms_idx[:, None] + norms_idx[None, :] - 2 * sub_dots
        # min_dist_sq = np.min(dist_sq)
        # return max(0, M_sq - min_dist_sq)
        
        # To avoid large temporary arrays, we can compute min dist_sq directly?
        # min_{i!=j} (norms_i + norms_j - 2 dot_ij)
        
        # For small k, just do it.
        dist_sq = norms_idx[:, None] + norms_idx[None, :] - 2 * sub_dots
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        
        return max(0, M_sq - min_dist_sq)

    # Initial violation should be 0 for the greedy core (mostly)
    # But greedy core might have issues if we didn't check carefully? 
    # Greedy ensured dot <= 2 for m=4. 
    # M_sq = 4. 
    # dist_sq = 4 + 4 - 2 dot = 8 - 2 dot.
    # If dot <= 2, dist_sq >= 8 - 4 = 4 = M_sq. 
    # So core is valid.
    
    current_violation = get_violation(list(selected_set))
    current_k = len(selected_set)
    
    best_k = current_k
    best_set = set(selected_set)
    
    # SA Loop
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    start_time = time.time()
    
    # Candidate pools for swapping in
    # We can swap any vector in the pool.
    # To speed up, we can pre-compute "promising" candidates?
    # Or just random sampling.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a move
        # 1. Remove a random vector from selected
        # 2. Add a random vector from pool (not in selected)
        
        if len(selected_set) == 0:
            # Add random
            new_idx = np.random.randint(n_cand)
            new_set = selected_set | {new_idx}
        else:
            # Remove
            to_remove = np.random.choice(list(selected_set))
            # Add
            # Sample from pool
            new_idx = np.random.randint(n_cand)
            
            # If new_idx is already in set, just remove? No, that decreases k.
            # We want to maintain size or increase.
            # Let's do: Remove one, Add one (size constant) OR Add one (size +1) OR Remove one (size -1)
            # To maximize k, we should bias towards adding.
            
            # Let's try: 
            # 50% chance: Swap (remove 1, add 1)
            # 30% chance: Add 1 (if valid or low violation)
            # 20% chance: Remove 1 (to escape deep local minima)
            
            r = np.random.random()
            if r < 0.5:
                new_set = (selected_set - {to_remove}) | {new_idx}
            elif r < 0.8:
                if new_idx not in selected_set:
                    new_set = selected_set | {new_idx}
                else:
                    new_set = selected_set # no change
            else:
                new_set = selected_set - {to_remove}
                
        new_violation = get_violation(list(new_set))
        new_k = len(new_set)
        
        # Objective: Maximize k, minimize violation.
        # Score = k - penalty * violation
        # Let's use: delta_score = (new_k - current_k) - 10 * (new_violation - current_violation)
        
        delta_k = new_k - current_k
        delta_v = new_violation - current_violation
        delta_score = delta_k - 10 * delta_v
        
        if delta_score >= 0 or np.random.random() < np.exp(delta_score / T):
            selected_set = new_set
            current_violation = new_violation
            current_k = new_k
            
            if current_k > best_k:
                best_k = current_k
                best_set = set(selected_set)
        
        T *= cooling
        
    # Return the best set found
    result_indices = list(best_set)
    if not result_indices:
        # Fallback
        return _generate_seed_vectors()
        
    result = candidates[result_indices]
    
    # Final validation
    if len(result) > 1:
        norms_sq_res = np.sum(result**2, axis=1)
        M_sq = np.max(norms_sq_res)
        dots_res = result @ result.T
        dist_sq = norms_sq_res[:, None] + norms_sq_res[None, :] - 2 * dots_res
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        if min_dist_sq < M_sq - 1e-6:
            # Invalid, try to repair or return greedy
            # For now, return the greedy core if it's valid
            # The greedy core was valid.
            core_result = shell4[core_indices]
            if len(core_result) > 0:
                return core_result
            return _generate_seed_vectors()
            
    return result

# EVOLVE-BLOCK-END
