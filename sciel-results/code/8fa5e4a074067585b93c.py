"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11
# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 120s

    # Generate a diverse pool of candidates: weights 2, 3, 4, 5
    # This allows the algorithm to mix different norms, which can sometimes
    # allow for denser packing than a fixed-norm set, provided the min distance
    # constraint (>= max norm) is met.
    # 
    # Note: The problem asks for min pairwise distance >= max norm.
    # If we mix norms, the "max norm" in the set becomes the constraint for ALL pairs.
    # So we are effectively looking for a set S such that d(u,v) >= max_{w in S} ||w|| for all u,v in S.
    # This is a strict constraint. Usually, optimal solutions in small dims are homogeneous 
    # or semi-homogeneous. However, a dynamic greedy approach can find local optima 
    # that static graph methods might miss due to order of addition.
    
    # Let's generate candidates for weights 2, 3, 4, 5
    # Weight 2: C(11,2)*4 = 55*4 = 220
    # Weight 3: C(11,3)*8 = 165*8 = 1320
    # Weight 4: C(11,4)*16 = 330*16 = 5280
    # Weight 5: C(11,5)*32 = 462*32 = 14784
    # Total ~ 21600 candidates. This is manageable.
    
    candidates = []
    for w in [2, 3, 4, 5]:
        for positions in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(D, dtype=np.int64)
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Precompute norms squared to avoid sqrt
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # We will build the set greedily.
    # State: list of indices in 'selected'
    # We need to efficiently check if a new candidate c can be added.
    # Condition: min_{s in selected} ||c - s||^2 >= max_{s in selected} ||s||^2
    # Note: ||c - s||^2 = ||c||^2 + ||s||^2 - 2 c.s
    # So we need: ||c||^2 + ||s||^2 - 2 c.s >= max_norm_sq for all s in selected.
    # Let M = max_norm_sq.
    # For a candidate c with norm_sq C:
    # We need: C + S_i - 2 c.s_i >= M for all i.
    # => 2 c.s_i <= C + S_i - M
    # => c.s_i <= (C + S_i - M) / 2
    
    # This check is O(k) where k is current set size.
    # To speed up, we can sort candidates by some heuristic (e.g., norm) or just iterate.
    # With ~20k candidates and set size ~100-200, 20k * 200 = 4M ops. Very fast.
    
    # Heuristic: Start with vectors of higher norm? Or lower?
    # Usually, starting with "spread out" vectors is good.
    # Let's try sorting by norm descending, so M is set early and high.
    # Or ascending? If M is low, it's easier to satisfy. But we want large k.
    # The constraint is d >= M. If M is large, it's harder to pack.
    # So we should probably keep M as small as possible?
    # But if we use weight-5 vectors, M=5. If we use weight-4, M=4.
    # If we mix, M is determined by the largest norm in the set.
    # If we include a weight-5 vector, M becomes 5. Then ALL pairs must be dist >= 5.
    # If we only use weight-4, M=4. Pairs must be dist >= 4.
    # Is it better to have M=5 with fewer vectors, or M=4 with more?
    # Kissing number usually scales with dimension. In d=11, known bounds are around 500-600?
    # Actually, for d=11, the kissing number is known to be at least 500.
    # The seed program found ~189 (score 1.89 implies k/seed? No, score is k. Wait.
    # "score is k". "search score is value/seed".
    # If seed score is 1.8909, and seed k is probably 1 or something?
    # No, "1.0 equals the seed program". So seed program score is 1.0?
    # But the prompt says "CURRENT PROGRAM (score 1.8909)".
    # And "inspiration 1 (score 1.8909): Initial program...".
    # This implies the "seed" for the *island* might be 1.0, but the current best is 1.8909.
    # Or maybe the "seed program" in the context of the ratio is a baseline k=1?
    # If k=189, and score=1.89, then seed_k=100?
    # Let's assume the goal is to maximize k.
    
    # Let's try a multi-start greedy approach with different initial candidates.
    # Since the pool is small, we can afford multiple runs.
    
    best_set = []
    best_size = 0
    
    # Strategy:
    # 1. Try greedy with different sorting orders (Norm desc, Norm asc, Random).
    # 2. For each run, build the set.
    # 3. Keep the best.
    
    np.random.seed(42)
    
    # Precompute Gram matrix of candidates? 21600x21600 is too big (400M entries, ~3GB).
    # So we must compute dot products on the fly or in chunks.
    # Dot product of one candidate with current set (size ~200) is fast.
    
    def try_greedy(order_idx, max_time=30):
        """
        order_idx: np.array of indices into candidates, sorted by priority.
        """
        selected = []
        selected_vecs = [] # list of numpy arrays
        selected_norms_sq = []
        max_norm_sq = 0
        
        t_start = time.time()
        
        for idx in order_idx:
            if time.time() > max_time + t_start:
                break
            
            c = candidates[idx]
            C = norms_sq[idx]
            
            # Check if C < max_norm_sq?
            # If C < max_norm_sq, adding it doesn't change M, but it must satisfy dist >= M.
            # If C >= max_norm_sq, it becomes the new M.
            
            # Optimization: If C < max_norm_sq, we can skip candidates that are "too close" to existing ones.
            # But we have to check all.
            
            # Quick rejection: If C + min_selected_norm_sq - 2*max_possible_dot < M?
            # Hard to bound max_possible_dot without checking.
            
            # Just check against all selected
            can_add = True
            for s_idx, s_vec in zip(selected, selected_vecs):
                # dist_sq = C + S - 2 c.s
                dot = np.dot(c, s_vec)
                dist_sq = C + selected_norms_sq[s_idx] - 2 * dot
                if dist_sq < max_norm_sq:
                    can_add = False
                    break
            
            if can_add:
                # Add it
                selected.append(idx)
                selected_vecs.append(c)
                selected_norms_sq.append(C)
                if C > max_norm_sq:
                    max_norm_sq = C
                    # Note: Adding a vector with larger norm raises the bar for ALL future additions.
                    # This might be a bad move if it blocks many others.
                    # Heuristic: Maybe we should prefer lower norm vectors first to keep M low?
                    # Let's try sorting by Norm ASCENDING first.
        
        return selected
    
    # Generate different orders
    # Order 1: Norm Ascending (keep M low)
    order_norm_asc = np.argsort(norms_sq)
    
    # Order 2: Norm Descending (start with high M, see if we can pack tightly)
    order_norm_desc = np.argsort(-norms_sq)
    
    # Order 3: Random
    order_random = np.random.permutation(n_cand)
    
    # Order 4: Weight 4 only, sorted by some metric?
    # Let's stick to the mixed pool for now.
    
    # Run greedy for each order
    for order in [order_norm_asc, order_norm_desc, order_random]:
        sel = try_greedy(order, max_time=20)
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel
    
    # Local Search / Hill Climbing on the best set found
    # Try to swap elements or add/remove to improve size.
    # Given time constraints, a simple "add if possible" pass on the full pool again
    # might catch things missed due to order.
    
    # Let's do a second pass: 
    # Take the best_set, and try to add any candidate from the pool that fits.
    # Then try to remove one candidate to allow adding two others? (Swap)
    
    # For now, just return the best greedy result.
    # If time permits, do a refinement.
    
    if time.time() < deadline - 10:
        # Refinement: Try to add more vectors to the best_set
        current_selected = set(best_set)
        current_vecs = [candidates[i] for i in best_set]
        current_norms_sq = [norms_sq[i] for i in best_set]
        current_max_norm_sq = max(current_norms_sq) if current_norms_sq else 0
        
        # Iterate through all candidates not in set
        # This is O(N * K)
        added_any = True
        while added_any and time.time() < deadline - 5:
            added_any = False
            for idx in range(n_cand):
                if idx in current_selected:
                    continue
                
                c = candidates[idx]
                C = norms_sq[idx]
                
                can_add = True
                for s_idx, s_vec in zip(best_set, current_vecs):
                    dot = np.dot(c, s_vec)
                    dist_sq = C + current_norms_sq[s_idx] - 2 * dot
                    if dist_sq < current_max_norm_sq:
                        can_add = False
                        break
                
                if can_add:
                    best_set.append(idx)
                    current_vecs.append(c)
                    current_norms_sq.append(C)
                    if C > current_max_norm_sq:
                        current_max_norm_sq = C
                    added_any = True
    
    # Convert to array
    if not best_set:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    result = candidates[np.array(best_set)]
    return result

# EVOLVE-BLOCK-END
