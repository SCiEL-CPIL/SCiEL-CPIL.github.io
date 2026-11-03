"""Improved covering design solver using dual-guided block selection with local improvement."""
import time
import random
import itertools
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    
    # For each k-block, precompute which t-subsets it covers
    # block_covers[b_idx] = frozenset of t-subset indices covered by block b_idx
    block_covers = []
    for b_idx, block in enumerate(all_kblocks):
        covered = set()
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            covered.add(idx)
        block_covers.append(frozenset(covered))
    
    # For each t-subset, precompute which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubsets)]
    for b_idx, covers in enumerate(block_covers):
        for ts_idx in covers:
            tsub_to_blocks[ts_idx].append(b_idx)
    
    # Dual-guided greedy: iteratively adjust dual weights for uncovered t-subsets
    def dual_guided_cover(seed, alpha=1.0, beta=0.5):
        rng = random.Random(seed)
        uncovered = set(range(num_tsubsets))
        selected_blocks = []
        
        # Dual variables: weight for each t-subset, initialized to 1.0
        dual = [1.0] * num_tsubsets
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Find the block that maximizes the sum of dual weights of uncovered t-subsets it covers
            best_block = -1
            best_score = -1.0
            
            # Sample a subset of blocks to speed up search, or use all if feasible
            # For v=19, k=8, there are C(19,8) = 75582 blocks - too many to check all each iteration
            # Use a sampling approach: check blocks that cover at least one uncovered t-subset
            # But even that could be many. Let's use a smarter approach.
            
            # Alternative: pick a random uncovered t-subset, get blocks covering it, 
            # then evaluate those blocks plus some random blocks
            if len(uncovered) > 0:
                # Pick a few random uncovered t-subsets to guide the search
                sample_size = min(10, len(uncovered))
                sample_tsubs = rng.sample(list(uncovered), sample_size)
                
                # Get candidate blocks from these t-subsets
                candidate_blocks = set()
                for ts_idx in sample_tsubs:
                    for b_idx in tsub_to_blocks[ts_idx]:
                        candidate_blocks.add(b_idx)
                
                # Also add some random blocks for diversity
                if num_kblocks > 100:
                    for _ in range(min(20, num_kblocks)):
                        candidate_blocks.add(rng.randint(0, num_kblocks - 1))
                
                # Evaluate each candidate block
                for b_idx in candidate_blocks:
                    score = 0.0
                    for ts_idx in block_covers[b_idx]:
                        if ts_idx in uncovered:
                            score += dual[ts_idx]
                    
                    if score > best_score:
                        best_score = score
                        best_block = b_idx
            
            if best_block == -1 or best_score <= 0:
                break
            
            selected_blocks.append(all_kblocks[best_block])
            covered_by_best = block_covers[best_block]
            
            # Update uncovered
            uncovered -= covered_by_best
            
            # Update dual variables: increase for still-uncovered, decrease for covered
            for ts_idx in covered_by_best:
                if ts_idx in uncovered:
                    dual[ts_idx] *= (1 + alpha)
                else:
                    dual[ts_idx] *= (1 - beta)
            
            # Normalize dual to prevent overflow
            max_dual = max(dual) if dual else 1.0
            if max_dual > 1000:
                dual = [d / max_dual for d in dual]
        
        return selected_blocks
    
    # Redundancy removal: remove blocks whose t-subsets are all covered by other blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Convert to frozensets for fast lookup
        block_sets = [frozenset(b) for b in blocks]
        
        # For each t-subset, track which blocks cover it
        tsub_cover_count = defaultdict(int)
        for i, bs in enumerate(block_sets):
            for tsub in itertools.combinations(bs, t):
                idx = tsub_to_idx[tsub]
                tsub_cover_count[idx] += 1
        
        # Remove redundant blocks
        new_blocks = []
        for i, bs in enumerate(block_sets):
            redundant = True
            for tsub in itertools.combinations(bs, t):
                idx = tsub_to_idx[tsub]
                if tsub_cover_count[idx] == 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(list(bs))
        
        return new_blocks
    
    # Local improvement: try to remove blocks
    def local_improve(blocks, deadline):
        if not blocks:
            return blocks
        
        blocks = remove_redundant(blocks)
        
        # Try to find blocks that can be removed without losing coverage
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                
                # Temporarily remove block i
                candidate_blocks = blocks[:i] + blocks[i+1:]
                
                # Check if all t-subsets are still covered
                covered = set()
                for b in candidate_blocks:
                    bs = frozenset(b)
                    for tsub in itertools.combinations(bs, t):
                        covered.add(tsub)
                
                if len(covered) == num_tsubsets:
                    blocks = candidate_blocks
                    improved = True
                    break
        
        return blocks
    
    # Try multiple strategies with different parameters and seeds
    best_blocks = None
    best_count = float('inf')
    
    strategies = [
        (42, 1.0, 0.5),
        (101, 1.5, 0.3),
        (102, 0.8, 0.7),
        (103, 2.0, 0.2),
        (201, 1.2, 0.4),
        (202, 0.9, 0.6),
        (301, 1.8, 0.35),
        (302, 1.1, 0.55),
        (401, 2.5, 0.15),
    ]
    
    for seed, alpha, beta in strategies:
        if time.time() > deadline - 15:
            break
        
        blocks = dual_guided_cover(seed, alpha, beta)
        if blocks is not None:
            blocks = remove_redundant(blocks)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
    
    # Try local improvement on the best solution
    if best_blocks is not None and time.time() < deadline - 20:
        improved = local_improve(best_blocks, deadline)
        if improved and len(improved) < best_count:
            best_blocks = improved
            best_count = len(improved)
    
    # Try a few more seeds if time remains
    seed = 1000
    while time.time() < deadline - 10:
        blocks = dual_guided_cover(seed, 1.0, 0.5)
        if blocks is not None:
            blocks = remove_redundant(blocks)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        return []
    
    return best_blocks
