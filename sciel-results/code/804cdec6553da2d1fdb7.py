import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Precompute pair_to_triples for fast lookup
    # pair_to_triples[(a,b)] = list of (z, idx) where {a,b,z} is a t-subset
    # This allows O(1) lookup of triples containing a specific pair
    pair_to_triples = defaultdict(list)
    for idx, sub in enumerate(all_subsets):
        if t == 3:
            a, b, c = sub
            pair_to_triples[(a,b)].append((c, idx))
            pair_to_triples[(a,c)].append((b, idx))
            pair_to_triples[(b,c)].append((a, idx))
        else:
            # General case: for each subset of size t-1
            for i in range(t):
                # Create a "super-point" or just store the subset of t-1 points
                # For simplicity in this specific task (t=3), we optimize for t=3.
                # If t != 3, fallback to slower method.
                pass

    if t != 3:
        # Fallback for general t (slower)
        def get_triples_for_block(block):
            return list(itertools.combinations(block, t))
    else:
        # Optimized for t=3
        def get_triples_for_block(block):
            # Returns list of indices of t-subsets in block
            # block is a list of points
            res = []
            n = len(block)
            for i in range(n):
                for j in range(i+1, n):
                    a, b = block[i], block[j]
                    if a > b: a, b = b, a
                    # pair_to_triples contains (z, idx)
                    # We need to check if z is in block
                    # This is O(k^3) if we check membership
                    # But k is small.
                    for z, idx in pair_to_triples[(a,b)]:
                        if z in block:
                            res.append(idx)
            return res

    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(seed_rng, alpha=0.5):
        """
        Greedy construction with redundancy penalty.
        alpha controls the penalty for covering already covered triples.
        """
        blocks = []
        uncovered = np.ones(n_subsets, dtype=bool)
        cov_count = np.zeros(n_subsets, dtype=np.int32)
        num_uncovered = n_subsets
        
        while num_uncovered > 0:
            if time.time() > deadline:
                return None
            
            # Find a random uncovered subset to seed the block
            seed_idx = -1
            # Optimization: maintain a list of uncovered indices? 
            # np.where is O(N). N=2024. This is fast.
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            seed_idx = seed_rng.choice(unc_idx)
            seed_sub = all_subsets[seed_idx]
            block = list(seed_sub)
            
            # Greedily add points to maximize Score = Gain - alpha * Redundancy
            while len(block) < k:
                block_set = set(block)
                best_point = None
                best_score = -1.0
                
                # Precompute pairs in current block for faster gain calculation
                block_pairs = list(itertools.combinations(block, 2))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    redundancy_cost = 0.0
                    
                    # For each pair (a,b) in block, the new triples are {x,a,b}
                    # We need to check if {x,a,b} is a valid t-subset (it is, if distinct)
                    # and get its index.
                    # Using pair_to_triples:
                    for a, b in block_pairs:
                        pa, pb = (a, b) if a < b else (b, a)
                        # Get all triples containing a,b
                        # One of them is {x,a,b}
                        # We can just construct the triple and lookup idx
                        triple = tuple(sorted((x, a, b)))
                        idx = subset_to_idx.get(triple, -1)
                        if idx != -1:
                            if uncovered[idx]:
                                gain += 1
                            else:
                                # Penalty for redundancy
                                # Using inverse square of count + 1
                                c = cov_count[idx]
                                redundancy_cost += 1.0 / ((c + 1) * (c + 1))
                    
                    score = gain - alpha * redundancy_cost
                    
                    if score > best_score:
                        best_score = score
                        best_point = x
                    elif score == best_score and best_score > 0:
                        # Tie break randomly to diversify
                        if seed_rng.random() < 0.5:
                            best_point = x
                
                if best_point is None:
                    # If no point improves score, pick a random one to fill
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = seed_rng.choice(available)
                    else:
                        break
                
                block.append(best_point)
            
            if len(block) < k:
                # Fill remaining with random points
                block_set = set(block)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered and cov_count
            # Get all t-subsets of the new block
            block_triples = get_triples_for_block(block)
            for idx in block_triples:
                if uncovered[idx]:
                    uncovered[idx] = False
                    num_uncovered -= 1
                cov_count[idx] += 1
                
        return blocks
    
    def local_search(blocks):
        """
        1. Remove redundant blocks.
        2. Try to replace blocks to reduce count.
        """
        blocks = [list(b) for b in blocks]
        
        def compute_coverage_counts(blks):
            counts = np.zeros(n_subsets, dtype=np.int32)
            for b in blks:
                tris = get_triples_for_block(b)
                for idx in tris:
                    counts[idx] += 1
            return counts
        
        # Step 1: Remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            counts = compute_coverage_counts(blocks)
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                b = blocks[i]
                tris = get_triples_for_block(b)
                redundant = True
                for idx in tris:
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        
        # Step 2: Block Replacement / Swap
        # Try to find a block that can be removed if we modify another block?
        # Or just try to find a better block for a specific "hole"?
        # Simple heuristic: Try to replace a random block with a new greedy block
        # constrained to cover the unique subsets of the old block.
        
        max_iter = 50
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if len(blocks) < 2:
                break
            
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            
            # Find subsets uniquely covered by old_block
            counts_without_i = compute_coverage_counts([b for j, b in enumerate(blocks) if j != i])
            unique_subs = []
            for idx in get_triples_for_block(old_block):
                if counts_without_i[idx] == 0:
                    unique_subs.append(all_subsets[idx])
            
            if not unique_subs:
                blocks.pop(i)
                continue
            
            # Try to construct a new block that covers all unique_subs
            # and minimizes redundancy with the rest.
            # Candidates must contain the union of unique_subs
            candidates = set()
            for sub in unique_subs:
                candidates.update(sub)
            
            if len(candidates) > k:
                continue
            
            # Complete the block with random points
            best_new_block = None
            best_score = -1
            for _ in range(20): # Try a few completions
                cand = set(candidates)
                while len(cand) < k:
                    x = random.randint(0, v-1)
                    if x not in cand:
                        cand.add(x)
                new_block = tuple(sorted(cand))
                
                # Check if it covers all unique_subs
                valid = True
                new_block_set = set(new_block)
                for sub in unique_subs:
                    if not all(x in new_block_set for x in sub):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate redundancy score against other blocks
                # Score = - sum(1/count^2) for triples in new_block that are covered by others
                # We want to MINIMIZE redundancy.
                # Let's just count how many triples in new_block are ALREADY covered by others.
                # Fewer is better.
                covered_by_others = 0
                for idx in get_triples_for_block(list(new_block)):
                    if counts_without_i[idx] > 0:
                        covered_by_others += 1
                
                if covered_by_others < best_score or best_new_block is None:
                    best_score = covered_by_others
                    best_new_block = list(new_block)
            
            if best_new_block is not None:
                blocks[i] = best_new_block
                blocks[i].sort()
                
        return blocks

    # Run multiple restarts with different alphas
    alphas = [0.0, 0.2, 0.5, 1.0]
    num_restarts = 0
    while time.time() < deadline - 20:
        alpha = alphas[num_restarts % len(alphas)]
        seed_rng = random.Random(num_restarts)
        blocks = greedy_construct(seed_rng, alpha)
        if blocks is None:
            continue
        
        if len(blocks) < best_count:
            improved = local_search(blocks)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
        
        num_restarts += 1
        
        # Early exit if we hit theoretical lower bound? 
        # Lower bound is 69. If we find 69, we can stop.
        if best_count <= 69:
            break

    if best_blocks is None:
        # Fallback
        seed_rng = random.Random(999)
        blocks = greedy_construct(seed_rng, 0.5)
        if blocks:
            return blocks
        return []
    
    return best_blocks
