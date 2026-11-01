"""Covering design C(24,7,3): Optimized greedy with vectorized gain calculation."""
import itertools
import time
import random
import numpy as np


def _precompute_triple_ids(v, t):
    """
    Precompute a 3D array triple_id of shape (v, v, v).
    triple_id[a, b, c] is the index of the sorted t-subset {a, b, c} in the list of all t-subsets,
    or -1 if {a,b,c} is not a valid t-subset (e.g., duplicates or size != t).
    For t=3, we just need distinct a,b,c.
    """
    # Generate all combinations and map them to indices
    all_combos = list(itertools.combinations(range(v), t))
    combo_to_idx = {combo: i for i, combo in enumerate(all_combos)}
    
    # Initialize triple_id with -1
    triple_id = np.full((v, v, v), -1, dtype=np.int32)
    
    for combo, idx in combo_to_idx.items():
        if t == 3:
            a, b, c = combo
            # Fill all permutations
            triple_id[a, b, c] = idx
            triple_id[a, c, b] = idx
            triple_id[b, a, c] = idx
            triple_id[b, c, a] = idx
            triple_id[c, a, b] = idx
            triple_id[c, b, a] = idx
            
    return triple_id, all_combos


def _build_greedy_vectorized(v, k, t, triple_id, all_combos, rng, deadline):
    """
    Build a covering using vectorized gain calculation.
    """
    total_triples = len(all_combos)
    uncovered = np.ones(total_triples, dtype=np.int8)
    blocks = []
    
    # Precompute M: M[x, y, z] = 1 if triple {x,y,z} is uncovered, else 0.
    # We update M whenever uncovered changes.
    # Shape: (v, v, v)
    M = np.zeros((v, v, v), dtype=np.int8)
    
    def update_M():
        nonlocal M, uncovered
        # Reset M to 0
        M[:] = 0
        # Set M[x,y,z] = 1 for all uncovered triples
        # Iterate over uncovered triples? Or iterate over all and check?
        # Since v=24, 24^3 = 13824. Iterating all is fine.
        # But we only need to set 1s.
        # We can use the fact that M is derived from uncovered.
        # Let's just loop through indices where uncovered is 1.
        unc_indices = np.where(uncovered)[0]
        for idx in unc_indices:
            combo = all_combos[idx]
            if t == 3:
                a, b, c = combo
                M[a, b, c] = 1
                M[a, c, b] = 1
                M[b, a, c] = 1
                M[b, c, a] = 1
                M[c, a, b] = 1
                M[c, b, a] = 1

    update_M()
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Pick a random uncovered triple
        unc_indices = np.where(uncovered)[0]
        start_idx = unc_indices[rng.randint(0, len(unc_indices) - 1)]
        a, b, c = all_combos[start_idx]
        block = [a, b, c]
        block_set = set(block)
        
        # Greedily add points
        while len(block) < k:
            # Calculate gains for all x not in block
            # gains[x] = sum_{y,z in block} M[x, y, z]
            gains = np.zeros(v, dtype=np.int32)
            
            # Iterate over pairs in current block
            for i in range(len(block)):
                for j in range(i + 1, len(block)):
                    y = block[i]
                    z = block[j]
                    # M[:, y, z] gives gains for all x from this pair
                    gains += M[:, y, z]
            
            # Mask out points already in block
            for p in block:
                gains[p] = -1000  # Very low penalty
            
            # Pick best x
            best_x = int(np.argmax(gains))
            
            if best_x == -1 or gains[best_x] < 0:
                # Fallback: pick random unused
                unused = [x for x in range(v) if x not in block_set]
                best_x = rng.choice(unused)
            
            block.append(best_x)
            block_set.add(best_x)
            
            # Update M and uncovered for the new triples formed by adding best_x
            # New triples are {best_x, y, z} for y, z in block[:-1]
            new_triples = []
            for i in range(len(block) - 1):
                for j in range(i + 1, len(block) - 1):
                    y = block[i]
                    z = block[j]
                    idx = triple_id[best_x, y, z]
                    if idx != -1:
                        new_triples.append(idx)
                        
            for idx in new_triples:
                if uncovered[idx]:
                    uncovered[idx] = 0
                    # Update M
                    combo = all_combos[idx]
                    if t == 3:
                        a2, b2, c2 = combo
                        M[a2, b2, c2] = 0
                        M[a2, c2, b2] = 0
                        M[b2, a2, c2] = 0
                        M[b2, c2, a2] = 0
                        M[c2, a2, b2] = 0
                        M[c2, b2, a2] = 0

        # Finalize block
        block.sort()
        blocks.append(block)
        
        # Mark all triples in this block as covered in uncovered and M
        # (We already marked some during construction, but need to ensure all are covered)
        for combo in itertools.combinations(block, t):
            idx = triple_id[combo[0], combo[1], combo[2]]
            if idx != -1 and uncovered[idx]:
                uncovered[idx] = 0
                a2, b2, c2 = all_combos[idx]
                M[a2, b2, c2] = 0
                M[a2, c2, b2] = 0
                M[b2, a2, c2] = 0
                M[b2, c2, a2] = 0
                M[c2, a2, b2] = 0
                M[c2, b2, a2] = 0

    return blocks


def _local_search_fast(blocks, v, k, t, triple_id, all_combos, deadline):
    """
    Fast local search: remove redundant blocks.
    """
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    total_triples = len(all_combos)
    
    # Count coverage
    coverage = np.zeros(total_triples, dtype=np.int16)
    block_triple_indices = []
    
    for b in blocks:
        idxs = []
        for combo in itertools.combinations(b, t):
            idx = triple_id[combo[0], combo[1], combo[2]]
            if idx != -1:
                idxs.append(idx)
                coverage[idx] += 1
        block_triple_indices.append(idxs)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Remove redundant blocks
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            redundant = True
            for idx in block_triple_indices[i]:
                if coverage[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_triple_indices[i]:
                    coverage[idx] -= 1
                blocks.pop(i)
                block_triple_indices.pop(i)
                n -= 1
                improved = True
                
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    triple_id, all_combos = _precompute_triple_ids(v, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    attempt = 0
    while time.time() < deadline - 5:
        attempt += 1
        rng = random.Random(attempt * 12345 + 678)
        blocks = _build_greedy_vectorized(v, k, t, triple_id, all_combos, rng, deadline - 5)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            # Try local search on the new best
            if time.time() < deadline - 2:
                ls_blocks = _local_search_fast(best_blocks, v, k, t, triple_id, all_combos, deadline - 2)
                if len(ls_blocks) < best_count:
                    best_count = len(ls_blocks)
                    best_blocks = ls_blocks

    return best_blocks
