import itertools
import time
import random
import math
import numpy as np

# Precomputation for v=24, t=3
V = 24
T = 3
K = 7

def _precompute():
    global triple_to_idx, idx_to_triple, all_triples, block_to_triples
    all_triples = list(itertools.combinations(range(V), T))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    idx_to_triple = all_triples
    
    # Precompute triples for all possible 7-subsets (blocks)
    # C(24, 7) = 346104. This is manageable.
    block_to_triples = {}
    for b in itertools.combinations(range(V), K):
        tris = []
        for tri in itertools.combinations(b, T):
            tris.append(triple_to_idx[tri])
        block_to_triples[b] = tris

_precompute()

def _generate_candidates(v, k, t, rng, num_candidates=50000):
    """Generate a diverse pool of candidate blocks."""
    candidates = set()
    # Add some random blocks
    for _ in range(num_candidates):
        b = tuple(sorted(rng.sample(range(v), k)))
        candidates.add(b)
    
    # Add some blocks based on "hot" points or triples? 
    # For now, random is a good baseline for the pool.
    # We could also add blocks that cover rare triples, but we don't know rarity yet.
    return list(candidates)

def _greedy_set_cover(candidates, v, k, t, deadline, rng):
    """
    Greedy set cover using logarithmic heuristic.
    Score(block) = (newly_covered_triples) / log(total_covered_triples + 1)
    """
    start_time = time.time()
    uncovered = [True] * len(idx_to_triple)
    blocks = []
    num_uncovered = len(idx_to_triple)
    
    # Precompute triples for each candidate for fast lookup
    # candidate_triples[i] = list of triple indices covered by candidate i
    candidate_triples = []
    for c in candidates:
        candidate_triples.append(block_to_triples[c])
        
    # To speed up, we can precompute the intersection sizes or just use numpy?
    # Let's stick to lists for simplicity, but optimize the inner loop.
    
    while num_uncovered > 0:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_score = -1
        best_new_count = 0
        
        # Heuristic: Gain / log(Gain + 1) is not quite right.
        # Standard greedy: Maximize Gain.
        # Log heuristic: Maximize Gain / log(1 + current_coverage_of_block)
        # But here "current_coverage" is 0 for new blocks.
        # Actually, the "Logarithmic Heuristic" mentioned in inspiration 3 usually refers to:
        # Select block maximizing (newly covered) / log(1 + total covered by block so far)
        # But since we are building from scratch, "total covered by block so far" is just the block's size in terms of triples (C(7,3)=35).
        # So the denominator is constant log(36).
        # So it reduces to standard greedy: Maximize newly covered.
        
        # Wait, the "Logarithmic Heuristic" in set cover often refers to the analysis bound, not the selection rule itself which is usually just "max marginal gain".
        # However, sometimes it means: Maximize (newly covered) / log(1 + number of elements already covered by this set).
        # Since we are adding new sets, the "already covered by this set" is 0.
        
        # Let's just use standard greedy: Maximize newly covered triples.
        # Tie-breaking: Randomly among ties.
        
        best_gain = -1
        best_rand = 0
        
        for i, tris in enumerate(candidate_triples):
            gain = 0
            for tri_idx in tris:
                if uncovered[tri_idx]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_rand = rng.random()
            elif gain == best_gain and gain > 0:
                r = rng.random()
                if r > best_rand:
                    best_idx = i
                    best_rand = r
        
        if best_idx == -1:
            # No more gains? Should not happen if candidates cover everything.
            # If it does, we might need to add a random block or break.
            break
            
        blocks.append(candidates[best_idx])
        
        # Update uncovered
        for tri_idx in candidate_triples[best_idx]:
            if uncovered[tri_idx]:
                uncovered[tri_idx] = False
                num_uncovered -= 1
                
    return blocks

def _remove_redundant(blocks, v, t, deadline):
    """Remove blocks that are not uniquely covering any triple."""
    if not blocks:
        return blocks
        
    cover_count = [0] * len(idx_to_triple)
    block_triples = []
    
    for b in blocks:
        tris = block_to_triples[b]
        block_triples.append(tris)
        for tri_idx in tris:
            cover_count[tri_idx] += 1
            
    changed = True
    active = [True] * len(blocks)
    
    while changed:
        changed = False
        if time.time() > deadline:
            break
            
        for i in range(len(blocks)):
            if not active[i]:
                continue
                
            is_redundant = True
            for tri_idx in block_triples[i]:
                if cover_count[tri_idx] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                active[i] = False
                changed = True
                for tri_idx in block_triples[i]:
                    cover_count[tri_idx] -= 1
                    
    return [blocks[i] for i in range(len(blocks)) if active[i]]

def _simulated_annealing(blocks, v, k, t, deadline, rng):
    """
    Local search: Swap points in blocks to reduce count.
    State: List of blocks.
    Move: Pick a block, pick a point in it, pick a point not in it. Swap.
    Cost: Number of blocks.
    But we want to *reduce* the number of blocks.
    How?
    1. Try to merge two blocks? No, size is fixed.
    2. Try to remove a block if it becomes redundant after a swap?
    
    Strategy:
    1. Start with a valid covering.
    2. Randomly pick a block B and a point p in B.
    3. Replace p with q not in B.
    4. Check if the covering is still valid.
    5. If valid, check if any block has become redundant (its unique coverage is gone).
    6. If a block becomes redundant, remove it. This reduces the count!
    
    This is a "swap-and-remove" strategy.
    """
    current_blocks = [list(b) for b in blocks]
    current_count = len(current_blocks)
    
    # Precompute cover counts for current state
    def compute_cover_counts(blks):
        counts = [0] * len(idx_to_triple)
        for b in blks:
            for tri in itertools.combinations(b, t):
                counts[triple_to_idx[tri]] += 1
        return counts
        
    counts = compute_cover_counts(current_blocks)
    
    # For faster redundancy check, we need to know which blocks cover which triples.
    # And for each triple, which blocks cover it.
    # triple_to_blocks: dict from triple_idx to list of block indices
    def build_triple_map(blks):
        t_map = {i: [] for i in range(len(idx_to_triple))}
        for bi, b in enumerate(blks):
            for tri in itertools.combinations(b, t):
                t_map[triple_to_idx[tri]].append(bi)
        return t_map
        
    t_map = build_triple_map(current_blocks)
    
    # To check if a block is redundant, we check if it has any triple where it is the ONLY block.
    # i.e., for all tri in block, len(t_map[tri]) > 1.
    
    def is_redundant(bi, blks, t_map):
        b = blks[bi]
        for tri in itertools.combinations(b, t):
            if len(t_map[triple_to_idx[tri]]) == 1:
                return False
        return True
        
    # Simulated Annealing Loop
    T_start = 1.0
    T_end = 0.01
    steps = 20000 # Approx
    
    # We will just do a fixed number of steps or until time limit.
    step = 0
    max_steps = 50000
    
    while step < max_steps:
        if time.time() > deadline:
            break
        step += 1
        
        # Pick a random block
        bi = rng.randint(0, len(current_blocks) - 1)
        b = current_blocks[bi]
        
        # Pick a point to swap out
        p = b[rng.randint(0, K - 1)]
        
        # Pick a point to swap in
        # Candidates: 0..v-1 not in b
        candidates = [x for x in range(v) if x not in b]
        if not candidates:
            continue
        q = rng.choice(candidates)
        
        # Perform swap
        b[b.index(p)] = q
        b.sort()
        new_block = tuple(b)
        
        # Check validity of the new covering
        # We need to update t_map and counts.
        # This is expensive if done naively.
        # Let's try a faster check:
        # 1. Identify triples lost from old block (containing p).
        # 2. Identify triples gained by new block (containing q).
        # 3. Check if any lost triple becomes uncovered (count drops to 0).
        
        old_block = tuple([x for x in b if x != q] + [p]) # Reconstruct old block? No, we modified b in place.
        # Let's revert and do it properly.
        b[b.index(q)] = p
        b.sort()
        old_block = tuple(b)
        
        # Triples in old_block containing p
        lost_triples = []
        for tri in itertools.combinations(old_block, t):
            if p in tri:
                lost_triples.append(tri)
                
        # Triples in new_block containing q
        gained_triples = []
        for tri in itertools.combinations(new_block, t):
            if q in tri:
                gained_triples.append(tri)
        
        # Check if any lost triple is uniquely covered by block bi
        valid_swap = True
        for tri in lost_triples:
            t_idx = triple_to_idx[tri]
            if len(t_map[t_idx]) == 1:
                valid_swap = False
                break
                
        if not valid_swap:
            continue
            
        # Swap is valid in terms of coverage (no triple becomes uncovered)
        # Now, does this swap allow us to remove another block?
        # Or does it make block bi redundant?
        
        # Apply swap to t_map
        for tri in lost_triples:
            t_idx = triple_to_idx[tri]
            t_map[t_idx].remove(bi)
            
        for tri in gained_triples:
            t_idx = triple_to_idx[tri]
            t_map[t_idx].append(bi)
            
        # Update current_blocks
        current_blocks[bi] = list(new_block)
        
        # Check if any block is now redundant
        # We only need to check blocks that were previously "critical" or block bi itself?
        # Actually, any block could become redundant if its unique coverage was "shared" by the new configuration?
        # No, redundancy means a block has NO unique triples.
        # The swap changed the coverage of block bi.
        # It might have removed the last unique triple of block bi.
        # It might have added a unique triple to block bi (making it less redundant).
        # It might have affected other blocks? No, other blocks didn't change.
        # So only block bi's redundancy status changes.
        # Wait, if block bi loses a triple, and that triple was covered ONLY by block bi, the swap was invalid (checked above).
        # So block bi does not lose any unique triples.
        # But block bi might have had unique triples that are now covered by others? No, we didn't change others.
        # So if block bi had a unique triple, it still has it (unless it was one of the lost ones, but we checked that).
        # Wait, if a triple was unique to bi, and we lost it, it would be uncovered. So we didn't lose unique triples.
        # So block bi cannot become redundant if it wasn't before?
        # Actually, if block bi had a unique triple, it stays unique.
        # So block bi's redundancy status is:
        # - If it was redundant before, it might still be redundant (if gained triples are not unique) or become non-redundant (if gained triples are unique).
        # - If it was non-redundant before, it remains non-redundant (since it didn't lose unique triples).
        
        # So we can only REMOVE a block if it was already redundant? No, we remove blocks to reduce count.
        # We remove a block if it is redundant.
        # If we make a redundant block non-redundant, that's bad.
        # If we make a non-redundant block redundant, that's good (we can remove it).
        
        # Can a non-redundant block become redundant?
        # Only if it loses all its unique triples.
        # But we checked that we don't lose any unique triples (because they would become uncovered).
        # So a non-redundant block cannot become redundant via a single swap that maintains coverage.
        
        # Conclusion: This simple swap strategy cannot reduce the block count directly by making a non-redundant block redundant.
        # It can only potentially make a redundant block non-redundant (bad) or keep it redundant.
        
        # So we need a different move.
        # Move: Remove a block.
        # If we remove a block, some triples become uncovered.
        # We must cover them by modifying other blocks.
        
        # This is getting complex. Let's try a simpler "Random Restart" with a better heuristic.
        # Or, just use the greedy set cover with a larger pool and better tie-breaking.
        
        # Let's revert the swap for now and just return the initial blocks.
        # The SA idea as formulated is flawed for reducing count without a more complex repair mechanism.
        
        # Revert t_map
        for tri in gained_triples:
            t_idx = triple_to_idx[tri]
            t_map[t_idx].remove(bi)
            
        for tri in lost_triples:
            t_idx = triple_to_idx[tri]
            t_map[t_idx].append(bi)
            
        # Revert block
        current_blocks[bi] = list(old_block)
        
    return current_blocks

def construct(v, k, t):
    start = time.time()
    deadline = start + 145.0  # Safety margin for 150s limit
    
    # Generate candidates
    rng = random.Random(42)
    candidates = _generate_candidates(v, k, t, rng, num_candidates=20000)
    
    # Greedy Set Cover
    blocks = _greedy_set_cover(candidates, v, k, t, deadline, rng)
    
    # Remove Redundant
    blocks = _remove_redundant(blocks, v, k, t, deadline)
    
    # Try a few restarts with different seeds and candidate pools
    best_blocks = blocks
    best_count = len(blocks)
    
    seed = 43
    while time.time() < deadline - 10.0:
        rng2 = random.Random(seed)
        # Generate new candidates
        candidates2 = _generate_candidates(v, k, t, rng2, num_candidates=10000)
        
        blocks2 = _greedy_set_cover(candidates2, v, k, t, deadline, rng2)
        blocks2 = _remove_redundant(blocks2, v, k, t, deadline)
        
        if len(blocks2) < best_count:
            best_blocks = blocks2
            best_count = len(blocks2)
            
        seed += 1
        if seed > 50:
            break
            
    return best_blocks
