"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Strategy: greedy packing on the {-1, 0, 1}^11 lattice pool.
    # We want to maximize k such that for all i != j:
    #   ||v_i - v_j|| >= max_j ||v_j||
    #
    # Key insight: if we normalize all vectors to have norm 2, the condition
    # becomes ||v_i - v_j|| >= 2 for all i != j, which is equivalent to
    # <v_i, v_j> <= 0 (since ||v_i - v_j||^2 = 4 - 2<v_i,v_j> >= 4 means <v_i,v_j> <= 0).
    #
    # But we're working with integer vectors of possibly different norms.
    # Let M = max_i ||v_i||. We need ||v_i - v_j|| >= M for all i != j.
    #
    # A clean approach: work with vectors of norm 2 (i.e., exactly two nonzero entries, each ±1).
    # Then the condition is ||v_i - v_j|| >= 2, i.e., <v_i, v_j> <= 0.
    #
    # For norm-2 vectors, <v_i, v_j> is:
    #   0 if they share no coordinates
    #   ±1 if they share exactly one coordinate (with same or opposite sign)
    #   ±2 if they are the same vector (not allowed since we pick distinct)
    #   -1 if they share one coordinate with opposite signs
    #
    # So the constraint <v_i, v_j> <= 0 means:
    #   - They cannot share a coordinate with the same sign (that gives +1)
    #   - They can share a coordinate with opposite sign (gives -1, OK)
    #   - They can share no coordinates (gives 0, OK)
    #
    # Actually wait - let me reconsider. The task says:
    # "minimum pairwise distance is at least the maximum vector norm"
    # Normalizing each vector to norm 2 certifies a kissing configuration.
    #
    # If all vectors have norm 2, then max norm = 2, and we need min pairwise distance >= 2.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2<v_i,v_j> = 4 + 4 - 2<v_i,v_j> = 8 - 2<v_i,v_j>
    # We need ||v_i - v_j|| >= 2, so 8 - 2<v_i,v_j> >= 4, i.e., <v_i,v_j> <= 2.
    #
    # Hmm, but for kissing configurations, typically we want ||v_i - v_j|| >= 2 when each has norm 1.
    # Let me re-read: "Normalizing each vector to norm 2 certifies a kissing configuration of equal-radius spheres."
    #
    # If we normalize to norm 2, the spheres have radius 1 (half the norm). Two spheres don't overlap
    # if distance between centers >= 2 (sum of radii). So we need ||v_i/2 - v_j/2|| >= 2,
    # i.e., ||v_i - v_j|| >= 4.
    #
    # Wait, that's not right either. Let me think more carefully.
    # 
    # Actually, the standard kissing number problem: place k unit spheres tangent to a central unit sphere
    # such that no two of the k spheres overlap. In R^n, this means k vectors of norm 2 (centers at distance 2
    # from origin) such that pairwise distances are >= 2 (spheres of radius 1 don't overlap).
    #
    # So: vectors of norm 2, pairwise distance >= 2.
    # ||v_i - v_j||^2 = 8 - 2<v_i, v_j> >= 4  =>  <v_i, v_j> <= 2.
    #
    # But the task says "minimum pairwise distance is at least the maximum vector norm".
    # If all have norm 2, then we need min distance >= 2. Same thing.
    #
    # For integer vectors with entries in {-1, 0, 1}, norm 2 means exactly two non-zero entries.
    # <v_i, v_j> for two such vectors:
    #   If they share 0 positions: dot = 0 <= 2 ✓
    #   If they share 1 position: dot = ±1 <= 2 ✓
    #   If they share 2 positions: dot = ±2 <= 2 ✓ (if same, dot=2; if opposite on one, dot=0; if opposite on both, dot=-2)
    #
    # Wait, if they share 2 positions and are identical, dot = 2 which is <= 2, OK. But they'd be the same vector!
    # We need distinct vectors.
    #
    # So actually, ANY set of distinct norm-2 integer vectors (with entries in {-1,0,1}) satisfies the condition!
    # The total number is C(11,2) * 4 = 55 * 4 = 220.
    #
    # But wait - can we do better with vectors of different norms?
    #
    # If we use vectors with larger norms, the constraint becomes tighter. Let's say we use vectors of norm r (integer).
    # Then we need pairwise distances >= r.
    # ||v_i - v_j||^2 >= r^2
    # ||v_i||^2 + ||v_j||^2 - 2<v_i,v_j> >= r^2
    #
    # If all have norm r: 2r^2 - 2<v_i,v_j> >= r^2 => <v_i,v_j> <= r^2/2.
    #
    # For r=2: <v_i,v_j> <= 2. As shown, all distinct norm-2 vectors work. Count = 220.
    #
    # Can we add vectors of norm > 2? If we have a vector of norm 3 (three nonzero entries, each ±1),
    # and a vector of norm 2, we need ||v_3 - v_2|| >= 3 (since max norm would be 3).
    # ||v_3 - v_2||^2 = 9 + 4 - 2<v_3,v_2> = 13 - 2<v_3,v_2> >= 9 => <v_3,v_2> <= 2.
    #
    # And between two norm-3 vectors: 18 - 2<v_3a,v_3b> >= 9 => <v_3a,v_3b> <= 4.5, so <v_3a,v_3b> <= 4.
    # Max dot product for two norm-3 vectors is 3 (if identical), so this is always satisfied for distinct vectors.
    #
    # Between norm-3 and norm-2: <v_3,v_2> <= 2. Since v_2 has 2 nonzero entries and v_3 has 3,
    # the dot product can be -2,-1,0,1,2. So we need <v_3,v_2> <= 2, which is always true!
    #
    # So it seems like we can potentially include vectors of norm 3 as well!
    #
    # Let's think about this more carefully. If we include vectors of various norms, the max norm M
    # determines the constraint. Let's try to maximize the total count.
    #
    # Approach: Start with all norm-2 vectors (220 vectors). Try to add norm-3 vectors one by one,
    # checking if they satisfy the distance constraint to all existing vectors.
    #
    # For a norm-3 vector v and existing norm-2 vector u:
    #   We need ||v - u|| >= M where M is the current max norm.
    #   If M = 3 (after adding v), we need ||v - u|| >= 3.
    #   ||v-u||^2 = 9 + 4 - 2<v,u> = 13 - 2<v,u> >= 9 => <v,u> <= 2.
    #   This is always satisfied since max dot product is 2.
    #
    # Between two norm-3 vectors v, w:
    #   ||v-w||^2 = 18 - 2<v,w> >= 9 => <v,w> <= 4.5 => <v,w> <= 4.
    #   Max dot product is 3, so always satisfied.
    #
    # So it seems like we can include ALL norm-2 and norm-3 vectors?
    # Number of norm-3 vectors: C(11,3) * 8 = 165 * 8 = 1320.
    # Total: 220 + 1320 = 1540.
    #
    # Wait, but I need to be more careful. When we add a norm-3 vector, the max norm becomes 3.
    # Then ALL pairwise distances must be >= 3.
    #
    # Between two norm-2 vectors u, v:
    #   ||u-v||^2 = 8 - 2<u,v>. We need this >= 9. So 8 - 2<u,v> >= 9 => -2<u,v> >= 1 => <u,v> <= -0.5.
    #   Since <u,v> is integer, <u,v> <= -1.
    #
    # So two norm-2 vectors can only coexist if their dot product is <= -1!
    # This means they must share at least one coordinate with opposite signs, OR share no coordinates with... 
    # wait, if they share no coordinates, dot = 0 which is > -1. Not allowed!
    #
    # So this is much more restrictive. Let me reconsider.
    #
    # The key constraint is: ALL pairwise distances >= max norm M.
    #
    # If we have any vector of norm M, then every pair (including pairs of norm-2 vectors) must have
    # distance >= M.
    #
    # So if M=3, two norm-2 vectors must have distance >= 3:
    #   ||u-v||^2 >= 9 => 8 - 2<u,v> >= 9 => <u,v> <= -1 (integer)
    #
    # This is very restrictive. Let's think about what configurations are possible.
    #
    # Alternative approach: maybe it's better to stick with all vectors having the same norm.
    # For norm 2, all 220 vectors work (as shown, any pair has dot <= 2, giving distance >= 2).
    #
    # Can we do better with norm 1 vectors? No, norm-1 vectors are just unit basis vectors and their negatives: 2*11 = 22 vectors. Much worse.
    #
    # What about a mix where we carefully select which vectors to include?
    #
    # Let me try a greedy approach:
    # 1. Generate all vectors with entries in {-1, 0, 1} (nonzero), sorted by norm.
    # 2. Use greedy packing: add vectors one at a time if they satisfy the constraint.
    #
    # But the constraint depends on the max norm of all selected vectors. This makes greedy tricky.
    #
    # Let me try a different strategy: fix the max norm M, and find the maximum number of vectors
    # with norm <= M such that all pairwise distances >= M.
    #
    # For M=2: vectors with norm 1 or 2.
    #   Norm 1: 22 vectors (e_i and -e_i)
    #   Norm 2: 220 vectors
    #   Constraint: all pairwise distances >= 2.
    #   For two norm-2 vectors: distance >= 2 always holds (shown above).
    #   For norm-1 and norm-2: ||e_i - v||^2 where v has norm 2.
    #     If i is in support of v: ||e_i - v||^2 = 1 + 4 - 2*1 = 3 >= 4? No, 3 < 4. FAIL!
    #     Wait, we need distance >= 2, so distance^2 >= 4. 3 < 4, so this fails.
    #
    #   So norm-1 vectors can't be paired with norm-2 vectors that include their coordinate.
    #
    # This is getting complex. Let me just implement the greedy algorithm on the full pool.
    #
    # Actually, let me reconsider the problem. The seed program returns 220 vectors (all norm-2 vectors).
    # Score is k = 220. The best found was 1.082 * 220 = 238. So we need to find ~238+ vectors.
    #
    # Let me think about what gives 238. Maybe a careful selection mixing norms.
    #
    # New idea: use vectors of norm 2 and norm 3, but carefully selected.
    #
    # Actually, let me just implement a randomized greedy approach on the {-1,0,1} lattice.
    # Generate random orderings of all nonzero vectors, and greedily add each if it fits.
    # Repeat multiple times, keep the best.

    def all_vectors():
        """Generate all nonzero vectors in {-1, 0, 1}^11."""
        import itertools
        vecs = []
        for pattern in itertools.product([-1, 0, 1], repeat=D):
            if any(x != 0 for x in pattern):
                vecs.append(pattern)
        return vecs

    # Precompute all vectors
    all_vecs = all_vectors()
    n_total = len(all_vecs)
    
    # Convert to numpy for efficient distance computation
    vecs_arr = np.array(all_vecs, dtype=np.int64)
    norms_sq = np.sum(vecs_arr ** 2, axis=1)
    norms = np.sqrt(norms_sq)
    
    # Sort by norm (ascending) for potential bias
    # Actually, for greedy, let's try different strategies
    
    def check_valid(selected_indices):
        """Check if the selected vectors satisfy the constraint."""
        if len(selected_indices) == 0:
            return True, 0
        sel_vecs = vecs_arr[selected_indices]
        sel_norms_sq = norms_sq[selected_indices]
        max_norm_sq = np.max(sel_norms_sq)
        max_norm = np.sqrt(max_norm_sq)
        
        # Compute all pairwise distances
        # For efficiency, do it in batches
        n_sel = len(selected_indices)
        if n_sel <= 1:
            return True, max_norm
        
        # Compute pairwise squared distances
        # dist^2 = ||a||^2 + ||b||^2 - 2<a,b>
        # For each pair, need dist^2 >= max_norm_sq
        for i in range(n_sel):
            ai = sel_vecs[i]
            ai_norm_sq = sel_norms_sq[i]
            for j in range(i + 1, n_sel):
                bj = sel_vecs[j]
                bj_norm_sq = sel_norms_sq[j]
                dot = np.dot(ai, bj)
                dist_sq = ai_norm_sq + bj_norm_sq - 2 * dot
                if dist_sq < max_norm_sq - 1e-9:
                    return False, max_norm
        return True, max_norm

    def greedy_pack(order):
        """Greedy pack using the given ordering of vector indices."""
        selected = []
        selected_vecs = []
        selected_norms_sq = []
        max_norm_sq = 0
        
        for idx in order:
            v = vecs_arr[idx]
            v_norm_sq = norms_sq[idx]
            new_max_norm_sq = max(max_norm_sq, v_norm_sq)
            
            # Check against all existing vectors
            valid = True
            for k, existing in enumerate(selected_vecs):
                dot = np.dot(v, existing)
                dist_sq = v_norm_sq + selected_norms_sq[k] - 2 * dot
                if dist_sq < new_max_norm_sq - 1e-9:
                    valid = False
                    break
            
            if valid:
                selected.append(idx)
                selected_vecs.append(v)
                selected_norms_sq.append(v_norm_sq)
                max_norm_sq = new_max_norm_sq
        
        return selected

    # Strategy 1: Sort by norm ascending, then greedy
    # This tends to pack small vectors first, which is usually better
    
    # Precompute norms for sorting
    norm_order = np.argsort(norms_sq)
    
    best_selected = []
    best_count = 0
    
    # Try greedy with norm-sorted order
    selected = greedy_pack(norm_order.tolist())
    best_selected = selected
    best_count = len(selected)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return np.array([vecs_arr[i] for i in best_selected], dtype=np.int64)
    
    # Try multiple random orders, biased towards smaller norms
    np.random.seed(42)
    
    for trial in range(200):
        if time.time() - start_time > deadline - 10:
            break
        
        # Create a random order with bias towards smaller norms
        # Assign weights inversely proportional to norm
        weights = 1.0 / (norms_sq + 1)
        weights = weights / weights.sum()
        
        # Sample without replacement using weights
        order = np.random.choice(n_total, size=n_total, replace=False, p=weights)
        
        selected = greedy_pack(order.tolist())
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected
    
    # Also try a local improvement: try to add more vectors
    if best_count > 0 and time.time() - start_time < deadline - 20:
        # Get the current best set
        current_set = set(best_selected)
        current_vecs = [vecs_arr[i] for i in best_selected]
        current_norms_sq = [norms_sq[i] for i in best_selected]
        max_norm_sq = max(current_norms_sq) if current_norms_sq else 0
        
        # Try adding vectors not in the set
        added_any = True
        while added_any and time.time() - start_time < deadline - 15:
            added_any = False
            for idx in range(n_total):
                if idx in current_set:
                    continue
                v = vecs_arr[idx]
                v_norm_sq = norms_sq[idx]
                new_max = max(max_norm_sq, v_norm_sq)
                
                valid = True
                for k, existing in enumerate(current_vecs):
                    dot = np.dot(v, existing)
                    dist_sq = v_norm_sq + current_norms_sq[k] - 2 * dot
                    if dist_sq < new_max - 1e-9:
                        valid = False
                        break
                
                if valid:
                    current_set.add(idx)
                    current_vecs.append(v)
                    current_norms_sq.append(v_norm_sq)
                    max_norm_sq = new_max
                    added_any = True
    
    result = [vecs_arr[i] for i in best_selected]
    return np.array(result, dtype=np.int64)
# EVOLVE-BLOCK-END
