import itertools
import time
import random
import numpy as np
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s kill

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Precompute for each point p, the list of triplet IDs containing p
    point_triplets = [set() for _ in range(v)]
    for tid, triplet in enumerate(all_triplets):
        for p in triplet:
            point_triplets[p].add(tid)

    # Precompute for each pair of points (a,b), the list of triplets containing both
    # This is useful for gain calculation when adding a point to a block
    # Actually, let's precompute for each (t-1)-subset S and point p not in S, the triplet ID
    # But that's just get_id.
    # Instead, precompute for each block prefix (list of points), the set of triplet IDs it covers?
    # No, that's too big.
    
    # Fast greedy construction
    def greedy_construct_fast(seed=0, mode='abs'):
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_count = n_triplets
        blocks = []
        
        # Track coverage count for redundancy detection later? No, just boolean for now.
        # For 'smooth' or 'diversity' we might need more.
        
        while uncovered_count > 0:
            if time.time() > deadline - 20:
                break
                
            # Pick random uncovered triplet
            # Optimization: maintain a list of uncovered IDs?
            # Since n_triplets is ~2000, scanning is okay but let's be smarter.
            # We can just pick a random index and check.
            start_tid = -1
            attempts = 0
            while attempts < 20:
                start_tid = rng.randrange(n_triplets)
                if is_uncovered[start_tid]:
                    break
                attempts += 1
            if not is_uncovered[start_tid]:
                # Fallback: find first uncovered
                for tid in range(n_triplets):
                    if is_uncovered[tid]:
                        start_tid = tid
                        break
                if not is_uncovered[start_tid]:
                    break # Should not happen if count > 0

            start_triplet = all_triplets[start_tid]
            block = list(start_triplet)
            block_set = set(block)
            
            # Greedily add points
            while len(block) < k:
                best_point = -1
                best_score = -1.0
                candidates = [p for p in range(v) if p not in block_set]
                
                if not candidates:
                    break

                # Precompute pairs in current block for gain calculation
                pairs_in_block = list(itertools.combinations(block, t - 1))
                num_pairs = len(pairs_in_block)
                
                # Calculate gain for each candidate
                # gain = number of newly covered triplets
                # For point p, new triplets are {p} U pair for pair in pairs_in_block
                # We need to check if these triplets are uncovered.
                
                # Optimization: Instead of checking each triplet individually,
                # we can precompute for each point p, the set of triplets it forms with any pair.
                # But pairs change as block grows.
                
                # Let's just compute gains.
                # To speed up, we can vectorize?
                # For each candidate p, we have num_pairs triplets.
                # We can create an array of triplet IDs and check is_uncovered on it.
                
                # Let's try to batch the ID lookups.
                # For all candidates, collect all potential triplets?
                # Total triplets to check = len(candidates) * num_pairs
                # For v=24, k=7, t=3.
                # Block size grows 3->7.
                # Size 3: pairs=3, candidates=21. 63 checks.
                # Size 4: pairs=6, candidates=20. 120 checks.
                # Size 5: pairs=10, candidates=19. 190 checks.
                # Size 6: pairs=15, candidates=18. 270 checks.
                # Total ~643 checks per block.
                # 76 blocks * 643 = ~49k checks. Very fast.
                
                # So simple loop is fine.
                
                gains = np.zeros(len(candidates), dtype=np.int32)
                for idx, p in enumerate(candidates):
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if is_uncovered[tid]:
                            gain += 1
                    gains[idx] = gain
                
                if mode == 'abs':
                    # Pick max gain
                    max_gain = np.max(gains)
                    if max_gain > 0:
                        # Pick random among those with max gain
                        best_indices = np.where(gains == max_gain)[0]
                        best_point = candidates[rng.choice(best_indices)]
                    else:
                        best_point = rng.choice(candidates)
                elif mode == 'smooth':
                    # Soft max / diversity
                    # Score = gain * 10 - redundancy * 0.5 + noise
                    # Redundancy: how many of these triplets are ALREADY covered?
                    # Actually, gain is new. Redundancy = num_pairs - gain.
                    # So score = gain * 10 - (num_pairs - gain) * 0.5 + noise
                    # = gain * 10.5 - num_pairs * 0.5 + noise
                    # This is linear in gain. So it's the same as abs, just different tie-breaking.
                    # To get diversity, we need to look at global state.
                    # Let's add a penalty for points that appear in many existing blocks.
                    point_freq = Counter()
                    for b in blocks:
                        for p in b:
                            point_freq[p] += 1
                            
                    scores = []
                    for idx, p in enumerate(candidates):
                        gain = gains[idx]
                        freq_penalty = point_freq.get(p, 0) * 0.1
                        score = gain * 10.0 - freq_penalty + rng.random() * 0.1
                        scores.append(score)
                    best_idx = np.argmax(scores)
                    best_point = candidates[best_idx]
                else: # 'random'
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    uncovered_count -= 1

        return blocks

    # Local Search / Improvement
    def improve_blocks(blocks, seed=0):
        rng = random.Random(seed)
        
        # Count coverage
        count = np.zeros(n_triplets, dtype=np.int32)
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                count[get_id(triplet)] += 1
        
        # 1. Remove redundant blocks
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if b is redundant
                redundant = True
                for triplet in itertools.combinations(b, t):
                    if count[get_id(triplet)] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b
                    for triplet in itertools.combinations(b, t):
                        count[get_id(triplet)] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        
        # 2. Try to replace two blocks with one
        # This is expensive, so only do a few iterations
        n = len(blocks)
        for _ in range(50): # Limit iterations
            if time.time() > deadline - 10:
                break
            if n < 2:
                break
            
            # Pick two random blocks
            i, j = rng.sample(range(n), 2)
            b1 = blocks[i]
            b2 = blocks[j]
            
            # Find triplets uniquely covered by b1 or b2
            unique_to_b1 = set()
            unique_to_b2 = set()
            
            for triplet in itertools.combinations(b1, t):
                tid = get_id(triplet)
                if count[tid] == 1:
                    unique_to_b1.add(tid)
            
            for triplet in itertools.combinations(b2, t):
                tid = get_id(triplet)
                if count[tid] == 1:
                    unique_to_b2.add(tid)
            
            all_unique = unique_to_b1 | unique_to_b2
            if not all_unique:
                continue
                
            # Check if all these triplets fit in a single block of size k
            points_needed = set()
            for tid in all_unique:
                points_needed.update(all_triplets[tid])
                
            if len(points_needed) <= k:
                # Try to merge
                new_block = list(points_needed)
                # Fill with random points
                while len(new_block) < k:
                    p = rng.randint(0, v-1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()
                
                # Check if new_block covers all unique triplets
                # (It should, by construction, but let's be safe)
                # Also, it must not break coverage of other triplets?
                # We are removing b1 and b2.
                # Any triplet covered ONLY by b1 or b2 is in all_unique.
                # So if new_block covers all_unique, we are safe.
                
                # Update counts
                for triplet in itertools.combinations(b1, t):
                    count[get_id(triplet)] -= 1
                for triplet in itertools.combinations(b2, t):
                    count[get_id(triplet)] -= 1
                for triplet in itertools.combinations(new_block, t):
                    count[get_id(triplet)] += 1
                    
                # Verify no triplet has count 0
                valid = True
                for tid in range(n_triplets):
                    if count[tid] == 0:
                        valid = False
                        break
                if valid:
                    blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    blocks.append(new_block)
                    n = len(blocks)
                    changed = True
                else:
                    # Rollback
                    for triplet in itertools.combinations(new_block, t):
                        count[get_id(triplet)] -= 1
                    for triplet in itertools.combinations(b1, t):
                        count[get_id(triplet)] += 1
                    for triplet in itertools.combinations(b2, t):
                        count[get_id(triplet)] += 1

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    modes = ['abs', 'smooth']
    num_seeds = 100
    
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        mode = modes[seed % 2]
        blocks = greedy_construct_fast(seed, mode=mode)
        
        if len(blocks) < best_count:
            # Try to improve
            improved = improve_blocks(blocks, seed=seed)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
            else:
                best_count = len(blocks)
                best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct_fast(0, mode='abs')
        best_blocks = improve_blocks(best_blocks, seed=0)

    return best_blocks

if __name__ == "__main__":
    pass
