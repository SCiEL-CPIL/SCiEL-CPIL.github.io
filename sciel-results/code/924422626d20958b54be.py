"""
Group Testing / Information-Theoretic Covering C(24,7,3)

Formulates the problem as minimizing the number of 7-sets such that every
3-subset is covered. Uses entropy reduction to prioritize blocks that
maximally reduce the "information content" of the uncovered triple set,
with a DP-style pruning over candidate block counts per triple.
"""
import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Construct covering design C(v,k,t) using Information-Theoretic
    Group Testing with dynamic candidate-count pruning.
    """
    all_triples = set(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    
    # Phase 1: Build initial covering via entropy-greedy with DP pruning
    blocks = _info_greedy_dp(v, k, t, all_triples)
    
    # Phase 2: Aggressive local search — removal + replacement
    blocks = _local_search_aggressive(v, k, t, all_triples, blocks)
    
    return blocks

def _info_greedy_dp(v, k, t, all_triples):
    """
    Greedy block selection maximizing entropy reduction of uncovered triples.
    
    Key idea: each uncovered triple has a 'candidate count' = number of
    potential 7-blocks that could cover it. Triples with low candidate counts
    are 'scarce' — they constrain the solution more. We prioritize blocks
    that cover triples with the lowest candidate counts, and among those,
    maximize the number of newly covered triples weighted by scarcity.
    
    DP-style pruning: we precompute for each triple how many blocks can cover
    it (C(v-t, k-t) = C(21,4) for v=24,k=7,t=3). As we add blocks, we
    decrement candidate counts. Triples whose candidate count drops to 1
    become 'forced' — any valid completion MUST include a block covering them.
    """
    uncovered = set(all_triples)
    blocks = []
    
    # Precompute: for each triple, the set of points NOT in the triple
    # A block covering triple T must include T plus (k-t) more points from V\T
    # So number of candidate blocks for T = C(v-t, k-t)
    from math import comb
    candidate_count_initial = comb(v - t, k - t)  # C(21,4) = 5985
    
    # Track candidate counts per triple
    # candidate_count[triple] = how many potential blocks can still cover this triple
    # We track this implicitly: a triple is covered by a block iff all its
    # points are in the block.
    
    # For efficiency, track point frequencies and use a scoring heuristic
    # that approximates entropy reduction.
    
    # Precompute all C(v,k) possible blocks? C(24,7) = 346104 — too many to enumerate.
    # Instead, use a greedy construction that builds blocks around critical triples.
    
    # Strategy:
    # 1. Find the 'most constrained' uncovered triple (fewest candidate blocks
    #    that don't conflict with already-placed blocks in a way that wastes coverage).
    # 2. Build a block around it that maximizes weighted coverage of other
    #    uncovered triples, where weight = 1/candidate_count (scarcity weight).
    
    # To make this tractable, we use a two-level heuristic:
    # Level 1: Score triples by scarcity (inverse of how many blocks could cover them
    #          given current block set — approximated by point co-occurrence).
    # Level 2: For the chosen seed triple, greedily extend to a full block maximizing
    #          weighted gain.
    
    # We maintain a "residual candidate count" approximation:
    # For triple T, residual candidates = blocks containing T that aren't yet placed.
    # We approximate this by: C(v-t, k-t) - (number of placed blocks containing T)
    # but since we discard covered triples, uncovered triples have 0 placed blocks
    # containing them. So the constraint is really about FUTURE blocks.
    
    # Better approach: use a 'degree' measure.
    # For each point p, let deg(p) = number of uncovered triples containing p.
    # A triple T = {a,b,c} has scarcity score = 1/deg(a) + 1/deg(b) + 1/deg(c).
    # Higher score = more isolated = harder to cover = prioritize.
    
    while uncovered:
        # Compute point degrees in uncovered set
        point_deg = [0] * v
        for triple in uncovered:
            for p in triple:
                point_deg[p] += 1
        
        # Score each uncovered triple
        best_triple = None
        best_score = -1
        for triple in uncovered:
            score = 0.0
            for p in triple:
                score += 1.0 / (point_deg[p] + 1)
            if score > best_score:
                best_score = score
                best_triple = triple
        
        # Build block around best_triple
        block = list(best_triple)
        
        # Greedily extend: pick point that maximizes weighted new coverage
        while len(block) < k:
            best_point = None
            best_gain = -1.0
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0.0
                # New triples formed by adding x to block
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        tri = tuple(sorted((block[i], block[j], x)))
                        if tri in uncovered:
                            # Weight by scarcity of this triple
                            w = 0.0
                            for p in tri:
                                w += 1.0 / (point_deg[p] + 1)
                            gain += w
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain <= 0:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for tri in itertools.combinations(block, t):
            uncovered.discard(tri)
        
        # Update point degrees
        for tri in itertools.combinations(block, t):
            if tri not in uncovered:  # was just covered
                for p in tri:
                    point_deg[p] -= 1
    
    return blocks

def _local_search_aggressive(v, k, t, all_triples, blocks, max_time=100):
    """
    Aggressive local search:
    1. Iteratively remove any block whose removal still leaves a valid cover.
    2. Try replacing each block with a better one that covers the same
       'essential' triples plus more.
    3. Random restarts within the local search to escape local optima.
    """
    blocks = [list(b) for b in blocks]
    start_time = time.time()
    random.seed(123)
    
    def is_valid(blk_list):
        covered = set()
        for b in blk_list:
            for tri in itertools.combinations(b, t):
                covered.add(tri)
        return covered == all_triples
    
    def essential_triples(blk_list, idx):
        """Find triples covered ONLY by blk_list[idx]."""
        other_covered = set()
        for i, b in enumerate(blk_list):
            if i == idx:
                continue
            for tri in itertools.combinations(b, t):
                other_covered.add(tri)
        must = set()
        for tri in itertools.combinations(blk_list[idx], t):
            if tri not in other_covered:
                must.add(tri)
        return must
    
    # Pass 1: Remove redundant blocks (repeat until stable)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            trial = blocks[:i] + blocks[i+1:]
            if is_valid(trial):
                blocks = trial
                changed = True
                break
    
    # Pass 2: Replace blocks to improve efficiency
    for _ in range(5000):
        if time.time() - start_time > max_time:
            break
        if len(blocks) == 0:
            break
        
        idx = random.randint(0, len(blocks) - 1)
        must_cover = essential_triples(blocks, idx)
        
        if not must_cover:
            blocks = blocks[:idx] + blocks[idx+1:]
            continue
        
        # If must_cover is too large, skip
        if len(must_cover) > 15:
            continue
        
        # Try to find a block covering all must_cover triples
        # Start with the union of points in must_cover
        candidate_points = set()
        for tri in must_cover:
            candidate_points.update(tri)
        
        if len(candidate_points) > k:
            continue
        
        new_block = list(candidate_points)
        
        # Greedily fill remaining slots to maximize additional coverage
        remaining_must = set(must_cover)
        while len(new_block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in new_block:
                    continue
                gain = 0
                for i in range(len(new_block)):
                    for j in range(i+1, len(new_block)):
                        tri = tuple(sorted((new_block[i], new_block[j], x)))
                        if tri in remaining_must:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None or best_gain == 0:
                break
            new_block.append(best_point)
        
        new_block.sort()
        if len(new_block) == k and len(set(new_block)) == k:
            new_covered = set(itertools.combinations(new_block, t))
            if must_cover.issubset(new_covered):
                # Check if this replacement actually helps (covers more)
                old_covered = set(itertools.combinations(blocks[idx], t))
                if len(new_covered) >= len(old_covered):
                    blocks[idx] = new_block
    
    # Pass 3: Final removal pass
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
