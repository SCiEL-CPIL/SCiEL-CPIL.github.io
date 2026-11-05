import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142  # 8s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute triplet ID lookup for fast access during greedy construction
    # For t=3, we can encode (a,b,c) with a<b<c as a*325 + b*12 + c for v=24
    # But let's use a dict for generality
    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]
    
    # Precompute for each point p and each (t-1)-subset S, the triplet ID of S ∪ {p}
    # This is expensive for large v, so we'll compute on-the-fly but cache per (p, frozenset)
    triplet_id_cache = {}
    def get_triplet_id_cached(p, subset_tuple):
        """Get triplet ID for point p and (t-1)-subset subset_tuple"""
        key = (p, subset_tuple)
        if key not in triplet_id_cache:
            triplet = tuple(sorted(list(subset_tuple) + [p]))
            triplet_id_cache[key] = triplet_to_id[triplet]
        return triplet_id_cache[key]
    
    # Precompute for each point p and each (t-1)-subset, the list of triplet IDs
    # that contain both p and the (t-1)-subset
    # For efficiency: for each point p, precompute which triplets contain p
    point_triplets = [set() for _ in range(v)]
    for tid, triplet in enumerate(all_triplets):
        for p in triplet:
            point_triplets[p].add(tid)

    # Greedy Construction with Incremental Block Building
    def greedy_construct(seed=0):
        rng = random.Random(seed)
        # List of uncovered triplet IDs
        uncovered_ids = list(range(n_triplets))
        uncovered_set = set(uncovered_ids)
        blocks = []
        
        while uncovered_ids:
            if time.time() > deadline - 15:
                break
                
            # Pick a random uncovered triplet to seed the block
            # Using index to allow O(1) removal later
            idx = rng.randrange(len(uncovered_ids))
            start_tid = uncovered_ids[idx]
            start_triplet = all_triplets[start_tid]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered_set:
                            gain += 1
                            
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p
                        if max_gain == len(pairs_in_block):
                            break
                            
                if max_gain == -1:
                    best_point = rng.choice(candidates)
                elif max_gain == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            to_remove = []
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                if tid in uncovered_set:
                    to_remove.append(tid)
            
            for tid in to_remove:
                uncovered_set.discard(tid)
                # Find index and remove from list (swapping with last for O(1))
                # This is O(N) worst case per removal, but total removals is N_triplets
                # We can just mark as removed and filter later, or use a dict?
                # Simpler: just rebuild uncovered_ids if it gets too sparse?
                # Or just use the set for membership, and list for random choice?
                # If we use list for random choice, we need to remove.
                # Let's just use the set for "is it uncovered" and a list for "pick one".
                # To pick one: 
                # 1. If len(uncovered_ids) is small, pick random.
                # 2. Remove from list.
                
                # Efficient removal from list:
                # Find index
                # Actually, doing linear search for removal is slow.
                # Let's just use the set `uncovered_set` for truth.
                # And for picking a random one, we can just pick a random index from `uncovered_ids` 
                # and check if it's in `uncovered_set`. If not, pick another?
                # Or better: maintain a list of "active" IDs.
                # Since N_triplets is 2024, linear search is actually fine (2024 ops).
                
                # Let's stick to simple: 
                # Remove from list
                if tid in uncovered_ids:
                    # Find and remove
                    # This is O(N)
                    uncovered_ids = [x for x in uncovered_ids if x != tid]
                    # Wait, this creates a new list every time. Expensive.
                    
            # Better approach for uncovered tracking:
            # Use a boolean array `is_uncovered`
            # And a list `uncovered_list`
            # When covering, set `is_uncovered[tid]=False`
            # When picking, pick random index from `uncovered_list`, 
            # while not `is_uncovered[id]`, pick again? 
            # Or just filter `uncovered_list` periodically.
            
            # Let's restart the uncovered tracking logic in the function for clarity and speed.
            
        return blocks

    # Precompute triplet ID matrix: point_pair_triplet[p][a][b] = tid for sorted triple (min, a, b)
    # For efficiency, precompute for each point p, an array mapping (i,j) pairs to triplet IDs
    # where i < j are the other two points in the triplet
    # Actually, let's use a 3D approach: for each point p, for each pair (a,b) with a<b,
    # store the triplet ID of sorted({p,a,b})
    
    # Precompute: triplet_id_3d[p][a][b] where a < b, and the triplet is sorted({p,a,b})
    # This is v x v x v which is 24^3 = 13824 entries, very manageable
    triplet_id_3d = np.full((v, v, v), -1, dtype=np.int32)
    for tid, (a, b, c) in enumerate(all_triplets):
        # For each point in the triplet, record the other two
        triplet_id_3d[a][b][c] = tid
        triplet_id_3d[a][c][b] = tid
        triplet_id_3d[b][a][c] = tid
        triplet_id_3d[b][c][a] = tid
        triplet_id_3d[c][a][b] = tid
        triplet_id_3d[c][b][a] = tid
    
    def get_triplet_id_fast(p, a, b):
        """Get triplet ID for points p, a, b (order doesn't matter)"""
        return int(triplet_id_3d[p][a][b])
    
    # Vectorized greedy construction
    def greedy_construct_fast(seed=0, mode='abs'):
        """
        mode: 'abs' - absolute gain, 'smooth' - smooth log-sum-exp based scoring,
              'diversity' - penalize points that overlap heavily with existing blocks
        """
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_count = n_triplets
        blocks = []
        block_point_sets = []  # list of sets for each block
        
        # For smooth scoring, track coverage counts for diversity penalty
        coverage_counts = np.zeros(n_triplets, dtype=np.int32)
        
        while uncovered_count > 0:
            if time.time() > deadline - 20:
                break
                
            # Pick random uncovered
            # Find a random uncovered triplet
            attempts = 0
            while attempts < 100:
                start_tid = rng.randrange(n_triplets)
                if is_uncovered[start_tid]:
                    break
                attempts += 1
            else:
                # Fallback: find first uncovered
                for tid in range(n_triplets):
                    if is_uncovered[tid]:
                        start_tid = tid
                        break
            
            start_triplet = all_triplets[start_tid]
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                best_score = -1.0
                candidates = [p for p in range(v) if p not in block_set]
                
                pairs_in_block = list(itertools.combinations(block, t - 1))
                num_pairs = len(pairs_in_block)
                
                # Compute gains for all candidates using vectorized lookup
                # For each candidate p, gain = sum over pairs (a,b) in block of is_uncovered[triplet_id_3d[p][a][b]]
                if mode == 'smooth':
                    scores = np.zeros(len(candidates))
                    for idx, p in enumerate(candidates):
                        gain = 0
                        redundant = 0
                        for a, b in pairs_in_block:
                            tid = int(triplet_id_3d[p][a][b])
                            if is_uncovered[tid]:
                                gain += 1
                            else:
                                redundant += 1
                        scores[idx] = gain * 10.0 - redundant * 0.5 + rng.random() * 0.1
                    best_idx = int(np.argmax(scores))
                    best_point = candidates[best_idx]
                    best_score = scores[best_idx]
                elif mode == 'diversity':
                    scores = np.zeros(len(candidates))
                    for idx, p in enumerate(candidates):
                        gain = 0
                        overlap = 0
                        for a, b in pairs_in_block:
                            tid = int(triplet_id_3d[p][a][b])
                            if is_uncovered[tid]:
                                gain += 1
                        for existing_block in block_point_sets:
                            if p in existing_block:
                                overlap += 1
                        scores[idx] = gain * 5.0 - overlap * 1.0 + rng.random() * 0.5
                    best_idx = int(np.argmax(scores))
                    best_point = candidates[best_idx]
                    best_score = scores[best_idx]
                else:  # 'abs'
                    # Vectorized: for each candidate p, count uncovered triplets with pairs in block
                    gains = np.zeros(len(candidates), dtype=np.int32)
                    for idx, p in enumerate(candidates):
                        for a, b in pairs_in_block:
                            tid = int(triplet_id_3d[p][a][b])
                            if is_uncovered[tid]:
                                gains[idx] += 1
                    best_idx = int(np.argmax(gains))
                    best_point = candidates[best_idx]
                    best_score = float(gains[best_idx])
                    if best_score == num_pairs:
                        break
                
                if best_point == -1:
                    best_point = rng.choice(candidates)
                elif best_score <= 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            block_point_sets.append(set(block))
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    uncovered_count -= 1
                coverage_counts[tid] += 1

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds with different modes
    # Cycle through modes: abs, smooth, diversity
    modes = ['abs', 'smooth', 'diversity']
    # Use more seeds in the earlier phase, then refine
    num_seeds = 200
    for seed in range(num_seeds):
        if time.time() > deadline - 60:
            break
        mode = modes[seed % 3]
        blocks = greedy_construct_fast(seed, mode=mode)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct_fast(0, mode='abs')

    current_blocks = list(best_blocks)
    
    # Simulated Annealing phase: try to reduce block count by removing blocks
    # and re-covering uncovered triplets with fewer new blocks
    def simulated_annealing(blocks, seed=0, max_iterations=500):
        """Try to reduce block count via simulated annealing."""
        rng = random.Random(seed)
        blocks = list(blocks)
        best_blocks = list(blocks)
        best_count = len(blocks)
        
        # Compute coverage
        count = np.zeros(n_triplets, dtype=np.int32)
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                count[get_id(triplet)] += 1
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        alpha = 0.95
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            T = T_start * (alpha ** iteration)
            if T < T_end:
                T = T_end
            
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.4 and len(blocks) > 1:
                # Try to remove a block
                idx = rng.randrange(len(blocks))
                removed_block = blocks[idx]
                
                # Check if removing this block leaves any triplet uncovered
                uncovered_after_removal = []
                for triplet in itertools.combinations(removed_block, t):
                    tid = get_id(triplet)
                    if count[tid] == 1:
                        uncovered_after_removal.append(tid)
                
                if not uncovered_after_removal:
                    # Safe to remove
                    blocks.pop(idx)
                    for triplet in itertools.combinations(removed_block, t):
                        count[get_id(triplet)] -= 1
                    delta = -1  # Improvement
                else:
                    # Need to add blocks to cover uncovered triplets
                    # Try to cover them with as few new blocks as possible
                    is_uncov = np.zeros(n_triplets, dtype=bool)
                    for tid in uncovered_after_removal:
                        is_uncov[tid] = True
                    uncov_count = len(uncovered_after_removal)
                    
                    new_blocks_needed = []
                    while uncov_count > 0:
                        # Pick random uncovered
                        tid = rng.choice(uncovered_after_removal)
                        start_triplet = all_triplets[tid]
                        block = list(start_triplet)
                        block_set = set(block)
                        
                        while len(block) < k:
                            best_p = -1
                            best_gain = -1
                            candidates = [p for p in range(v) if p not in block_set]
                            pairs_in_block = list(itertools.combinations(block, t - 1))
                            
                            for p in candidates:
                                gain = 0
                                for a, b in pairs_in_block:
                                    tid2 = int(triplet_id_3d[p][a][b])
                                    if is_uncov[tid2]:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_p = p
                            
                            if best_p == -1:
                                best_p = rng.choice(candidates)
                            block.append(best_p)
                            block_set.add(best_p)
                        
                        block.sort()
                        new_blocks_needed.append(block)
                        
                        # Update uncovered
                        for triplet in itertools.combinations(block, t):
                            tid2 = get_id(triplet)
                            if is_uncov[tid2]:
                                is_uncov[tid2] = False
                                uncov_count -= 1
                    
                    delta = len(new_blocks_needed) - 1  # Remove 1, add len(new_blocks_needed)
                    
                    if delta <= 0:
                        # Accept: remove old, add new
                        blocks.pop(idx)
                        for triplet in itertools.combinations(removed_block, t):
                            count[get_id(triplet)] -= 1
                        for nb in new_blocks_needed:
                            blocks.append(nb)
                            for triplet in itertools.combinations(nb, t):
                                count[get_id(triplet)] += 1
                    else:
                        # Might accept if T is high enough
                        if rng.random() < np.exp(-delta / T):
                            blocks.pop(idx)
                            for triplet in itertools.combinations(removed_block, t):
                                count[get_id(triplet)] -= 1
                            for nb in new_blocks_needed:
                                blocks.append(nb)
                                for triplet in itertools.combinations(nb, t):
                                    count[get_id(triplet)] += 1
                        else:
                            # Reject: do nothing (block stays)
                            pass
                    
            elif move_type < 0.7:
                # Try to swap a point in a block
                if len(blocks) < 2:
                    continue
                idx = rng.randrange(len(blocks))
                block = blocks[idx]
                old_point = rng.choice(block)
                
                # Find a new point
                candidates = [p for p in range(v) if p not in block]
                if not candidates:
                    continue
                new_point = rng.choice(candidates)
                
                # Compute delta: how many triplets become uncovered vs newly covered
                # Triplets containing old_point in this block
                old_triplets = []
                for other1, other2 in itertools.combinations([x for x in block if x != old_point], 2):
                    old_triplets.append((old_point, other1, other2))
                
                # Triplets containing new_point that would be in the new block
                new_block_temp = [x for x in block if x != old_point] + [new_point]
                new_triplets = []
                for other1, other2 in itertools.combinations([x for x in new_block_temp if x != new_point], 2):
                    new_triplets.append((new_point, other1, other2))
                
                # Count how many old triplets become uncovered
                lost = 0
                for tri in old_triplets:
                    tid = get_id(tuple(sorted(tri)))
                    if count[tid] == 1:
                        lost += 1
                
                # Count how many new triplets are newly covered
                gained = 0
                for tri in new_triplets:
                    tid = get_id(tuple(sorted(tri)))
                    if count[tid] == 0:
                        gained += 1
                
                delta = lost - gained  # Positive delta means worse
                
                if delta <= 0 or (delta > 0 and rng.random() < np.exp(-delta / T)):
                    # Accept
                    for tri in old_triplets:
                        tid = get_id(tuple(sorted(tri)))
                        count[tid] -= 1
                    for tri in new_triplets:
                        tid = get_id(tuple(sorted(tri)))
                        count[tid] += 1
                    block[block.index(old_point)] = new_point
                    block.sort()
            
            elif move_type < 0.9:
                # Try to swap two points in a block
                if len(blocks) < 2:
                    continue
                idx = rng.randrange(len(blocks))
                block = blocks[idx]
                if len(block) < 4:
                    continue
                p1_idx, p2_idx = rng.sample(range(len(block)), 2)
                p1, p2 = block[p1_idx], block[p2_idx]
                
                candidates = [p for p in range(v) if p not in block]
                if len(candidates) < 2:
                    continue
                new_p1, new_p2 = rng.sample(candidates, 2)
                
                # Compute delta
                # Old triplets involving p1 or p2
                other_points = [x for x in block if x not in (p1, p2)]
                old_triplets_p1 = []
                for a, b in itertools.combinations(other_points, 2):
                    old_triplets_p1.append((p1, a, b))
                old_triplets_p2 = []
                for a, b in itertools.combinations(other_points, 2):
                    old_triplets_p2.append((p2, a, b))
                old_triplets_p1p2 = [(p1, p2, x) for x in other_points]
                
                new_block_temp = [x for x in block if x not in (p1, p2)] + [new_p1, new_p2]
                other_points_new = [x for x in new_block_temp if x not in (new_p1, new_p2)]
                new_triplets_np1 = []
                for a, b in itertools.combinations(other_points_new, 2):
                    new_triplets_np1.append((new_p1, a, b))
                new_triplets_np2 = []
                for a, b in itertools.combinations(other_points_new, 2):
                    new_triplets_np2.append((new_p2, a, b))
                new_triplets_np1np2 = [(new_p1, new_p2, x) for x in other_points_new]
                
                lost = 0
                for tri in old_triplets_p1 + old_triplets_p2 + old_triplets_p1p2:
                    tid = get_id(tuple(sorted(tri)))
                    if count[tid] == 1:
                        lost += 1
                
                gained = 0
                for tri in new_triplets_np1 + new_triplets_np2 + new_triplets_np1np2:
                    tid = get_id(tuple(sorted(tri)))
                    if count[tid] == 0:
                        gained += 1
                
                delta = lost - gained
                
                if delta <= 0 or (delta > 0 and rng.random() < np.exp(-delta / T)):
                    # Accept
                    for tri in old_triplets_p1 + old_triplets_p2 + old_triplets_p1p2:
                        tid = get_id(tuple(sorted(tri)))
                        count[tid] -= 1
                    for tri in new_triplets_np1 + new_triplets_np2 + new_triplets_np1np2:
                        tid = get_id(tuple(sorted(tri)))
                        count[tid] += 1
                    block[p1_idx] = new_p1
                    block[p2_idx] = new_p2
                    block.sort()
            
            else:
                # Try to add a block (useful if we're stuck)
                if len(blocks) > 70:  # Don't add if we're already large
                    continue
                
                # Find an uncovered triplet
                uncovered_tids = [tid for tid in range(n_triplets) if count[tid] == 0]
                if not uncovered_tids:
                    continue
                
                tid = rng.choice(uncovered_tids)
                start_triplet = all_triplets[tid]
                block = list(start_triplet)
                block_set = set(block)
                
                while len(block) < k:
                    best_p = -1
                    best_gain = -1
                    candidates = [p for p in range(v) if p not in block_set]
                    pairs_in_block = list(itertools.combinations(block, t - 1))
                    
                    for p in candidates:
                        gain = 0
                        for a, b in pairs_in_block:
                            tid2 = int(triplet_id_3d[p][a][b])
                            if count[tid2] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                    
                    if best_p == -1:
                        best_p = rng.choice(candidates)
                    block.append(best_p)
                    block_set.add(best_p)
                
                block.sort()
                blocks.append(block)
                for triplet in itertools.combinations(block, t):
                    count[get_id(triplet)] += 1
                delta = 1  # Added a block
            
            # Track best
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
        
        return best_blocks
    
    # Run simulated annealing on the best solution found
    if time.time() < deadline - 30:
        annealed_blocks = simulated_annealing(current_blocks, seed=42, max_iterations=300)
        if len(annealed_blocks) < len(current_blocks):
            current_blocks = annealed_blocks
            best_count = len(current_blocks)
    
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                count[get_id(triplet)] += delta

    update_counts(current_blocks, 1)

    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_merge(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        # Precompute unique triplets per block for efficiency
        block_unique = []
        for b in blocks:
            ids_b = set()
            for triplet in itertools.combinations(b, t):
                tid = get_id(triplet)
                ids_b.add(tid)
            unique_b = [tid for tid in ids_b if count[tid] == 1]
            block_unique.append(unique_b)
        
        # Try to merge pairs where the union of unique triplets fits in k points
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            if not block_unique[i]:
                continue
            
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks, False
                if not block_unique[j]:
                    continue
                
                all_unique_ids = set(block_unique[i]) | set(block_unique[j])
                
                points_needed = set()
                for tid in all_unique_ids:
                    points_needed.update(all_triplets[tid])
                    
                if len(points_needed) <= k:
                    new_block = list(points_needed)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    update_counts([blocks[i], blocks[j]], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks.append(b)
                    new_blocks.append(new_block)
                    
                    return new_blocks, True

        return blocks, False

    def try_swap_2_for_1(blocks):
        """Replace two blocks with one block if the union of their unique triplets
        fits in a single k-subset."""
        n = len(blocks)
        if n < 2:
            return blocks, False

        # Precompute unique triplets per block
        block_unique = []
        for b in blocks:
            ids_b = set()
            for triplet in itertools.combinations(b, t):
                tid = get_id(triplet)
                ids_b.add(tid)
            unique_b = [tid for tid in ids_b if count[tid] == 1]
            block_unique.append(unique_b)

        for i in range(n):
            if time.time() > deadline - 3:
                return blocks, False
            if not block_unique[i]:
                continue
            for j in range(i + 1, n):
                if time.time() > deadline - 3:
                    return blocks, False
                if not block_unique[j]:
                    continue

                all_unique = set(block_unique[i]) | set(block_unique[j])
                points_needed = set()
                for tid in all_unique:
                    points_needed.update(all_triplets[tid])

                if len(points_needed) <= k:
                    new_block = list(points_needed)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()

                    update_counts([blocks[i], blocks[j]], -1)
                    update_counts([new_block], 1)

                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_replace_1_for_1(blocks):
        """Try to replace a single block with another block that covers all its
        unique triplets and potentially more. This helps escape local minima."""
        n = len(blocks)
        if n < 1:
            return blocks, False

        # Precompute unique triplets per block
        block_unique = []
        for b in blocks:
            ids_b = set()
            for triplet in itertools.combinations(b, t):
                tid = get_id(triplet)
                ids_b.add(tid)
            unique_b = [tid for tid in ids_b if count[tid] == 1]
            block_unique.append(unique_b)

        for i in range(n):
            if time.time() > deadline - 3:
                return blocks, False
            unique_b = block_unique[i]
            if not unique_b:
                continue

            # The new block must cover all unique triplets of block i
            points_needed = set()
            for tid in unique_b:
                points_needed.update(all_triplets[tid])

            if len(points_needed) > k:
                continue

            # Try to find a better block: one that covers unique_b and also
            # covers some triplets that are currently covered by other blocks
            # Strategy: start with points_needed, then greedily add points that
            # maximize additional coverage of triplets with count >= 2

            current_block = list(blocks[i])
            current_block_set = set(current_block)
            
            # Compute what triplets the current block covers
            current_triplet_ids = set()
            for triplet in itertools.combinations(current_block, t):
                tid = get_id(triplet)
                current_triplet_ids.add(tid)

            # Build new block starting from points_needed
            new_block = list(points_needed)
            new_block_set = set(new_block)
            
            # Greedily add points to maximize coverage of triplets with count >= 2
            while len(new_block) < k:
                candidates = [p for p in range(v) if p not in new_block_set]
                if not candidates:
                    break
                
                best_point = -1
                best_gain = 0
                pairs_in_block = list(itertools.combinations(new_block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        tid = get_triplet_id_cached(p, pair)
                        # Count how many additional coverings this provides
                        # We want to cover triplets that are currently covered by this block
                        # but might not be covered by the new block otherwise
                        if tid in current_triplet_ids:
                            if count[tid] >= 2:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1 or best_gain == 0:
                    # Add a random point
                    best_point = random.choice(candidates)
                
                new_block.append(best_point)
                new_block_set.add(best_point)
            
            new_block.sort()
            
            # Verify: the new block must cover all unique triplets of block i
            new_triplet_ids = set()
            for triplet in itertools.combinations(new_block, t):
                tid = get_id(triplet)
                new_triplet_ids.add(tid)
            
            if not set(unique_b).issubset(new_triplet_ids):
                continue

            # Check if this replacement is beneficial: it should cover at least
            # as many triplets as the old block, ideally more
            old_covered = len(current_triplet_ids)
            new_covered = len(new_triplet_ids)
            
            # Only replace if it's strictly better or equal (to avoid cycling)
            if new_covered > old_covered:
                update_counts([blocks[i]], -1)
                update_counts([new_block], 1)
                new_blocks = blocks[:i] + [new_block] + blocks[i+1:]
                return new_blocks, True

        return blocks, False

    progress_made = True
    while progress_made and time.time() < deadline - 8:
        progress_made = False
        
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True

        if not progress_made:
            current_blocks, merged = try_swap_2_for_1(current_blocks)
            if merged:
                progress_made = True

        if not progress_made:
            current_blocks, replaced = try_replace_1_for_1(current_blocks)
            if replaced:
                progress_made = True

    # Final validation: ensure all triplets are covered
    # If not, we need to add blocks to cover missing triplets
    final_count = np.zeros(n_triplets, dtype=np.int32)
    for block in current_blocks:
        for triplet in itertools.combinations(block, t):
            final_count[get_id(triplet)] += 1
    
    # Find uncovered triplets
    uncovered_tids = [tid for tid in range(n_triplets) if final_count[tid] == 0]
    
    if uncovered_tids:
        # Add blocks to cover uncovered triplets
        remaining_blocks = list(current_blocks)
        is_uncov = np.ones(n_triplets, dtype=bool)
        is_uncov[:n_triplets] = (final_count == 0)
        uncov_count = len(uncovered_tids)
        
        while uncov_count > 0 and time.time() < deadline - 2:
            # Pick a random uncovered triplet
            start_tid = random.choice(uncovered_tids)
            start_triplet = all_triplets[start_tid]
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for a, b in pairs_in_block:
                        tid = int(triplet_id_3d[p][a][b])
                        if is_uncov[tid]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1:
                    best_point = random.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            remaining_blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                if is_uncov[tid]:
                    is_uncov[tid] = False
                    uncov_count -= 1
        
        current_blocks = remaining_blocks

    # Remove duplicate blocks if any
    seen = set()
    unique_blocks = []
    for b in current_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    current_blocks = unique_blocks

    return current_blocks

if __name__ == "__main__":
    pass
