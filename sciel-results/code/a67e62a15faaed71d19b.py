import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate candidate pool
    # Include vectors with weights 2, 3, 4, 5, 6
    # For weight w, choose w indices from D, and assign signs
    pool = []
    
    # Pre-compute combinations for efficiency
    # Weights 2 to 6 are good candidates. 
    # Weight 2: C(11,2)*4 = 55*4 = 220
    # Weight 3: C(11,3)*8 = 165*8 = 1320
    # Weight 4: C(11,4)*16 = 330*16 = 5280
    # Weight 5: C(11,5)*32 = 462*32 = 14784
    # Weight 6: C(11,6)*64 = 462*64 = 29568
    # Total ~51k vectors. This is manageable.
    
    for w in [2, 3, 4, 5, 6]:
        for comb in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(comb, signs):
                    v[idx] = sign
                pool.append(v)
                
    pool = np.array(pool, dtype=np.int64)
    n_pool = len(pool)
    
    # Precompute norms squared
    norms_sq = np.sum(pool * pool, axis=1)
    
    # Initial greedy solution to start SA
    # We want to maximize the number of vectors k such that min_{i!=j} ||v_i - v_j|| >= max_k ||v_k||
    # Let R = max_k ||v_k||. Condition: ||v_i - v_j||^2 >= R^2 for all i!=j.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j
    # So we need ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j >= R^2
    
    # Greedy initialization
    # Sort by norm? Or random?
    # Let's try a greedy approach that picks vectors that allow for larger R or just packs many.
    # Actually, the score is k. The constraint depends on the max norm in the set.
    # If we pick all weight 2, R^2=4. Condition: 4+4-2d >= 4 => d <= 2.
    # If we mix weights, R is determined by the max weight in the set.
    
    # Let's run Simulated Annealing on the subset of indices selected.
    # State: a set of indices S.
    # Move: Add a random vector not in S, or remove a random vector in S, or swap.
    # Objective: Maximize |S| subject to constraint.
    # Constraint check is expensive if done fully every time.
    
    # Optimization: 
    # We can maintain the set of selected vectors and their norms.
    # When adding v_new, we need to check if it conflicts with any existing v_old.
    # Conflict if ||v_new - v_old||^2 < R^2, where R^2 = max(norms_sq[S] U {||v_new||^2}).
    # Note: If ||v_new||^2 > current R^2, the new R^2 becomes ||v_new||^2.
    # This makes the constraint stricter for all existing pairs? No, only for pairs involving v_new?
    # Wait. The condition is "minimum pairwise distance is at least the maximum vector norm".
    # Let R_S = max_{i in S} ||v_i||.
    # Condition: For all i, j in S, i!=j, ||v_i - v_j|| >= R_S.
    # If we add v_new, R_S might increase. If R_S increases, previous valid pairs might become invalid?
    # No, the distance between existing pairs doesn't change. But the threshold R_S increases.
    # So if R_S increases, we must re-verify all existing pairs against the new R_S.
    # This is O(N^2) per insertion if R changes.
    
    # Alternative Strategy:
    # Fix the target norm squared R^2. 
    # Try R^2 = 4, 5, 6, 7, 8...
    # For a fixed R^2, we need to find the max subset of vectors with norm <= R (actually <= sqrt(R^2)?? No, max norm is R, so all vectors must have norm <= R. And pairwise dist >= R).
    # Wait, if all vectors have norm <= R, and pairwise dist >= R, then R is the "max norm" only if at least one vector has norm R.
    # Actually, if we fix a set S, let R_S be max norm. The condition is min dist >= R_S.
    # It is equivalent to: There exists R such that all vectors have norm <= R and min dist >= R.
    # We can iterate over possible R^2 values (integer sums of squares).
    # For each R^2, find the max size subset of vectors with norm^2 <= R^2 and pairwise dist^2 >= R^2.
    # This is a standard maximum independent set problem in a conflict graph.
    # Conflict graph G_R: Nodes are vectors with norm^2 <= R^2. Edge between u,v if ||u-v||^2 < R^2.
    # We want Max Independent Set.
    
    # Since R^2 can be small, the graph might be sparse or dense.
    # Greedy + Local Search on the graph is a good heuristic for MIS.
    
    best_k = 0
    best_set = np.array([], dtype=np.int64).reshape(0, D)
    
    # Possible R^2 values from weights 2..6
    # Min norm^2 = 4 (weight 2)
    # Max norm^2 = 36 (weight 6)
    # We only care about R^2 that are achievable as max norm.
    # Actually, we can just try all vectors as potential "definers" of R?
    # Simpler: Iterate through sorted unique norms squared in the pool.
    
    unique_norms_sq = np.unique(norms_sq)
    
    for R_sq in unique_norms_sq:
        if time.time() > deadline - 5:
            break
            
        # Candidate vectors: those with norm_sq <= R_sq
        mask = norms_sq <= R_sq
        candidates = pool[mask]
        cand_norms_sq = norms_sq[mask]
        n_cand = len(candidates)
        
        if n_cand < best_k + 1:
            continue # Can't beat best_k if we don't have enough candidates? 
            # Actually we can have at most n_cand.
            
        # Build conflict graph implicitly? Or just use greedy + local search.
        # Let's use a randomized greedy approach with restarts for this R_sq.
        
        # To speed up, we can pre-compute dot products? 
        # n_cand can be large (up to 50k). 
        # Dot product matrix is 50k x 50k -> 2.5e9 entries. Too big for memory.
        # We must compute on the fly or in batches.
        
        # Greedy Heuristic for MIS:
        # 1. Shuffle candidates.
        # 2. Iterate, add if compatible with all currently selected.
        # 3. Local Search: Try to swap or add/remove.
        
        # Given time constraints, a simple greedy with random restarts might suffice if we optimize the check.
        
        def check_compatibility(v_new, selected_vecs, R_sq):
            # selected_vecs is Nx D
            # dist_sq = ||v_new||^2 + ||v_sel||^2 - 2 v_new . v_sel
            # We need dist_sq >= R_sq
            # ||v_new||^2 + ||v_sel||^2 - 2 v_new . v_sel >= R_sq
            # 2 v_new . v_sel <= ||v_new||^2 + ||v_sel||^2 - R_sq
            
            if len(selected_vecs) == 0:
                return True, 0
                
            dots = selected_vecs @ v_new
            # norms_sq_sel = np.sum(selected_vecs*selected_vecs, axis=1) # Precomputed?
            # We don't have precomputed norms for selected easily without storing indices.
            
            # Let's store indices of selected candidates.
            pass
        
        # This on-the-fly check is O(K) per addition. Total O(K^2).
        # With 5 restarts and K ~ 100-200, it's fast.
        
        # Let's refine the SA/Greedy loop.
        
        for restart in range(3):
            if time.time() > deadline - 2:
                break
                
            # Shuffle candidates
            order = np.random.permutation(n_cand)
            
            selected_indices = []
            selected_vecs = []
            
            for idx in order:
                if time.time() > deadline - 1:
                    break
                    
                v = candidates[idx]
                v_norm_sq = cand_norms_sq[idx]
                
                if len(selected_vecs) == 0:
                    selected_indices.append(idx)
                    selected_vecs.append(v)
                    continue
                    
                # Check against all selected
                # Convert selected_vecs to array for dot product
                # To avoid overhead, keep as list and convert? Or maintain array?
                # Maintaining array is better.
                
                # For simplicity in this step, let's just do a basic greedy first to get a baseline.
                # If we have time, we can improve.
                
                # Optimization: Use numpy array for selected_vecs
                # But appending to numpy array is slow. 
                # Let's use a list of vectors and check one by one?
                # Or batch check?
                
                # Let's try a different approach: 
                # Use the fact that we are maximizing k.
                # Start with empty set.
                # Add vectors in random order.
                
                # We need efficient dot product.
                # If selected_vecs is a list, we can't do vectorized dot easily without conversion.
                # Let's maintain a numpy array and use a "buffer" or just accept the cost for small K.
                
                # Actually, K is likely small (< 50) for high R.
                # For low R (4), K can be larger.
                
                # Let's just do a simple loop check for now.
                
                is_compatible = True
                for sv in selected_vecs:
                    dist_sq = v_norm_sq + np.sum(sv*sv) - 2 * np.dot(v, sv)
                    if dist_sq < R_sq:
                        is_compatible = False
                        break
                        
                if is_compatible:
                    selected_indices.append(idx)
                    selected_vecs.append(v)
                    
            k_curr = len(selected_vecs)
            if k_curr > best_k:
                best_k = k_curr
                # Reconstruct full vectors
                best_set = np.array([candidates[i] for i in selected_indices], dtype=np.int64)
                
    return best_set
# EVOLVE-BLOCK-END
