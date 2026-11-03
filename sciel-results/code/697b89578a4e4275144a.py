"""Optimized covering C(24,7,3) using multi-phase heuristic search."""
import itertools
import random
import time


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a multi-phase approach:
    1. Smart greedy with pair-awareness
    2. Local search optimization (block replacement)
    3. Final refinement
    """
    all_triples = set(itertools.combinations(range(v), t))
    
    # Phase 1: Smart greedy with pair tracking
    blocks = _greedy_covering(v, k, t, all_triples)
    
    # Phase 2: Local search - try to replace or remove blocks
    blocks = _local_search(v, k, t, all_triples, blocks)
    
    # Phase 3: Final refinement - try to merge or swap
    blocks = _final_refine(v, k, t, all_triples, blocks)
    
    return blocks


def _greedy_covering(v, k, t, all_triples):
    """Greedy covering with pair-awareness to reduce redundancy."""
    uncovered = set(all_triples)
    blocks = []
    
    # Track pair frequencies to prefer "fresh" pairs
    pair_count = {}
    
    while uncovered:
        # Pick the uncovered triple with the most "constrained" pairs
        # (pairs that appear in fewer remaining possibilities)
        # For speed, just pick the lexicographically smallest uncovered triple
        seed_triple = min(uncovered)
        
        # Build a block around this triple
        block = list(seed_triple)
        
        # Score function: gain = #new triples covered + bonus for new pairs
        while len(block) < k:
            best_point = None
            best_score = -1
            for x in range(v):
                if x in block:
                    continue
                # Calculate immediate gain
                new_triples = 0
                new_pairs = 0
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        pair = (block[i], block[j])
                        # This triple would be formed
                        triple = tuple(sorted([block[i], block[j], x]))
                        if triple in uncovered:
                            new_triples += 1
                        # Count new pairs
                        p1 = tuple(sorted((block[i], x)))
                        p2 = tuple(sorted((block[j], x)))
                        if pair_count.get(p1, 0) == 0:
                            new_pairs += 1
                        if pair_count.get(p2, 0) == 0:
                            new_pairs += 1
                
                # Combined score: prioritize immediate coverage, then freshness
                score = new_triples * 10 + new_pairs
                if score > best_score:
                    best_score = score
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered and pair counts
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
        for pair in itertools.combinations(block, 2):
            pair_count[tuple(sorted(pair))] = pair_count.get(tuple(sorted(pair)), 0) + 1
    
    return blocks


def _local_search(v, k, t, all_triples, blocks, max_iters=5000):
    """Local search: try removing blocks that are redundant, or replacing with better ones."""
    blocks = list(blocks)
    
    def is_valid(block_list):
        """Check if all triples are covered."""
        covered = set()
        for b in block_list:
            for triple in itertools.combinations(b, t):
                covered.add(triple)
        return covered == all_triples
    
    # First pass: try to remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            trial = blocks[:i] + blocks[i+1:]
            if is_valid(trial):
                blocks = trial
                changed = True
                break
    
    # Second pass: try to replace blocks with better ones
    random.seed(42)
    for iteration in range(max_iters):
        # Pick a random block to potentially improve
        idx = random.randint(0, len(blocks) - 1)
        old_block = set(blocks[idx])
        
        # Try to find a replacement block that covers all triples from old_block
        # that aren't covered by other blocks
        other_blocks = blocks[:idx] + blocks[idx+1:]
        
        # Find triples that ONLY this block covers
        covered_by_others = set()
        for b in other_blocks:
            for triple in itertools.combinations(b, t):
                covered_by_others.add(triple)
        
        must_cover = set()
        for triple in itertools.combinations(blocks[idx], t):
            if triple not in covered_by_others:
                must_cover.add(triple)
        
        if not must_cover:
            # This block is fully redundant, remove it
            blocks = blocks[:idx] + blocks[idx+1:]
            continue
        
        # Try to build a new block that covers must_cover
        # Start with points from must_cover triples
        candidate_points = set()
        for triple in must_cover:
            candidate_points.update(triple)
        
        if len(candidate_points) > k:
            continue
        
        # Try to extend to a full block
        new_block = list(candidate_points)
        remaining_must = set(must_cover)
        
        while len(new_block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in new_block:
                    continue
                # Count how many must_cover triples this would help
                gain = 0
                for i in range(len(new_block)):
                    for j in range(i+1, len(new_block)):
                        triple = tuple(sorted([new_block[i], new_block[j], x]))
                        if triple in remaining_must:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain == 0:
                break
            new_block.append(best_point)
        
        new_block.sort()
        if len(new_block) == k and len(set(new_block)) == k:
            # Check if this new block covers all must_cover
            new_covered = set(itertools.combinations(new_block, t))
            if must_cover.issubset(new_covered):
                blocks[idx] = new_block
    
    return blocks


def _final_refine(v, k, t, all_triples, blocks):
    """Final refinement: try pairwise swaps and small perturbations."""
    blocks = list(blocks)
    
    def is_valid(block_list):
        covered = set()
        for b in block_list:
            for triple in itertools.combinations(b, t):
                covered.add(triple)
        return covered == all_triples
    
    # Try to remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            trial = blocks[:i] + blocks[i+1:]
            if is_valid(trial):
                blocks = trial
                changed = True
                break
    
    return blocks
