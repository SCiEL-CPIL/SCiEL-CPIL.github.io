import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute conflict masks using Python big ints as bitsets
    # conflict_mask[v] has bit j set if (v*j) % (v+j) == 0 and j != v
    # Bit i (0-indexed) represents vertex (i+1)
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask

    all_mask = (1 << N) - 1  # bits 0..255 all set

    def bit_count(x):
        return x.bit_count()

    def get_vertex(x):
        """Get the lowest set bit position (0-indexed)."""
        return (x & (-x)).bit_length() - 1

    def greedy_min_degree(available, randomize=False, k=1, score_func=None):
        """
        Greedy independent set using min-degree heuristic.
        available: bitmask of available vertices
        score_func: optional function (v, rem) -> float to rank vertices
        Returns: (solution_list, solution_mask)
        """
        sol = []
        sol_mask = 0
        rem = available
        
        while rem:
            # Find vertex with minimum degree in induced subgraph
            if score_func is not None:
                # Use custom scoring
                best_v = -1
                best_score = float('-inf')
                temp = rem
                while temp:
                    lsb = temp & (-temp)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    score = score_func(v, rem, deg)
                    if score > best_score:
                        best_score = score
                        best_v = v
                    temp ^= lsb
                v = best_v
            elif randomize and k > 1:
                # Find top-k min-degree vertices and pick randomly
                temp = rem
                best_deg = float('inf')
                # First pass: find min degree
                t = rem
                while t:
                    lsb = t & (-t)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg < best_deg:
                        best_deg = deg
                    t ^= lsb
                
                # Collect vertices with degree within best_deg + small tolerance
                candidates = []
                t = rem
                while t:
                    lsb = t & (-t)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg <= best_deg + 1:  # allow slight tolerance
                        candidates.append(v)
                    t ^= lsb
                
                if not candidates:
                    # fallback: pick lowest
                    lsb = rem & (-rem)
                    v = lsb.bit_length()
                else:
                    kk = min(k, len(candidates))
                    v = candidates[random.randrange(kk)]
            else:
                # Deterministic min-degree
                min_v = -1
                min_deg = float('inf')
                temp = rem
                while temp:
                    lsb = temp & (-temp)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                    temp ^= lsb
                v = min_v
            
            if v < 0:
                break
                
            sol.append(v)
            sol_mask |= (1 << (v - 1))
            
            # Remove v and its neighbors from rem
            rem &= ~(1 << (v - 1))
            rem &= ~conflict_mask[v]
        
        return sol, sol_mask

    # --- Phase 1: Get a good initial solution via multiple greedy runs ---
    best_sol = []
    best_mask = 0
    best_size = 0
    
    # Deterministic greedy
    sol, sol_mask = greedy_min_degree(all_mask, randomize=False)
    if len(sol) > best_size:
        best_size = len(sol)
        best_sol = sol
        best_mask = sol_mask

    # Randomized greedy restarts to find better initial solution
    for it in range(300):
        if time.time() > deadline:
            break
        sol, sol_mask = greedy_min_degree(all_mask, randomize=True, k=5)
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_mask = sol_mask
            # If we've done well, do more
            if best_size >= 130:
                break

    # --- Phase 2: Local improvement - try to add vertices ---
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find all vertices that can be added
        for v in range(1, N + 1):
            if best_mask & (1 << (v - 1)):
                continue
            # Check if v conflicts with any in best_sol
            if conflict_mask[v] & best_mask:
                continue
            # Can add v
            best_mask |= (1 << (v - 1))
            best_size += 1
            improved = True

    # --- Phase 3: Spectral-inspired greedy using eigenvector-like scores ---
    # Compute a "centrality" score for each vertex based on its non-neighbors
    # Vertices with many non-neighbors in the current solution's complement are good to add
    # We use a power iteration-like approach on the complement graph
    
    # Build complement adjacency: non_conflict_mask[v] = all_mask ^ conflict_mask[v] ^ (1 << (v-1))
    non_conflict_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        non_conflict_mask[v] = all_mask ^ conflict_mask[v] ^ (1 << (v - 1))
    
    # Power iteration to find principal eigenvector of the complement graph
    # This identifies vertices that are "central" in the low-conflict core
    def power_iteration(complement_masks, n_iter=20, dim=N):
        """Find principal eigenvector using power iteration on complement graph."""
        # Initialize with random vector
        vec = [random.random() for _ in range(dim)]
        for _ in range(n_iter):
            new_vec = [0.0] * dim
            for i in range(dim):
                v = i + 1
                # Sum of vec[j] for all j non-conflicting with v
                mask = complement_masks[v]
                s = 0.0
                t = mask
                while t:
                    lsb = t & (-t)
                    j_idx = lsb.bit_length() - 1
                    s += vec[j_idx]
                    t ^= lsb
                new_vec[i] = s
            # Normalize
            norm = sum(x * x for x in new_vec) ** 0.5
            if norm > 1e-12:
                vec = [x / norm for x in new_vec]
            else:
                break
        return vec
    
    # Compute spectral scores
    spectral_scores = power_iteration(non_conflict_mask, n_iter=15)
    
    # Use spectral scores to guide greedy: prefer vertices with high spectral score
    # and low degree in remaining set
    def spectral_score_func(v, rem, deg):
        return spectral_scores[v - 1] * 1000.0 - deg
    
    # Try spectral-guided greedy multiple times
    for it in range(200):
        if time.time() > deadline:
            break
        sol, sol_mask = greedy_min_degree(all_mask, score_func=spectral_score_func)
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_mask = sol_mask
            # Polish
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if best_mask & bit:
                    continue
                if conflict_mask[v] & best_mask:
                    continue
                best_mask |= bit
                best_size += 1

    # --- Phase 4: 1-for-2 swap improvements ---
    def try_improve_1for2(best_mask, best_size):
        """Try to find a vertex v to remove and two vertices a,b to add."""
        for _ in range(200):
            if time.time() > deadline:
                break
            # Pick a random vertex to remove
            sol_list = [i+1 for i in range(N) if best_mask & (1 << i)]
            if not sol_list:
                break
            v_out = random.choice(sol_list)
            new_mask = best_mask & ~(1 << (v_out - 1))
            
            # Find vertices that can be added given new_mask
            addable = []
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if new_mask & bit:
                    continue
                if conflict_mask[v] & new_mask:
                    continue
                addable.append(v)
            
            if len(addable) >= 2:
                # Try adding 2 random addable vertices
                a, b = random.sample(addable, 2)
                # Check a and b don't conflict
                if conflict_mask[a] & (1 << (b - 1)):
                    continue
                new_mask2 = new_mask | (1 << (a - 1)) | (1 << (b - 1))
                return new_mask2, best_size + 1
        
        return None, best_size

    # Try several rounds of 1-for-2 improvements
    for round_num in range(50):
        if time.time() > deadline:
            break
        new_mask, new_size = try_improve_1for2(best_mask, best_size)
        if new_mask is not None:
            best_mask = new_mask
            best_size = new_size
        else:
            # Also try adding single vertices
            added = False
            for v in range(1, N + 1):
                if time.time() > deadline:
                    break
                bit = 1 << (v - 1)
                if best_mask & bit:
                    continue
                if conflict_mask[v] & best_mask:
                    continue
                best_mask |= bit
                best_size += 1
                added = True
            if not added:
                break

    # --- Phase 5: More randomized greedy with different orderings ---
    for it in range(3000):
        if time.time() > deadline:
            break
        
        # Randomized greedy with random vertex ordering
        rem = all_mask
        sol = []
        sol_mask = 0
        
        # Randomize the order by permuting vertices
        verts = list(range(1, N + 1))
        random.shuffle(verts)
        
        for v in verts:
            if time.time() > deadline:
                break
            bit = 1 << (v - 1)
            if not (rem & bit):
                continue
            # Check if v can be added (no conflict with current sol)
            if conflict_mask[v] & sol_mask:
                continue
            sol.append(v)
            sol_mask |= bit
            rem &= ~conflict_mask[v]
            rem &= ~bit
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_mask = sol_mask
            best_sol = sol
            # Polish: try to add more
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if best_mask & bit:
                    continue
                if conflict_mask[v] & best_mask:
                    continue
                best_mask |= bit
                best_size += 1

    # Final polish: greedy add
    for v in range(1, N + 1):
        if time.time() > deadline:
            break
        bit = 1 << (v - 1)
        if best_mask & bit:
            continue
        if conflict_mask[v] & best_mask:
            continue
        best_mask |= bit
        best_size += 1

    # Reconstruct solution list from mask
    final_sol = [i + 1 for i in range(N) if best_mask & (1 << i)]
    
    # Ensure nonempty
    if not final_sol:
        final_sol = [1]
    
    return final_sol
