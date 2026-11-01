import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute triple indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate structured block pool using AG(3,4)-inspired symmetry
    # Map 24 points to GF(4)^3 (first 24 of 64 points)
    # GF(4) elements: 0, 1, alpha, alpha+1 where alpha^2 = alpha + 1
    # Represent GF(4) as pairs (a, b) in GF(2)^2 with multiplication
    gf4_mult = {
        (0,0): {(0,0): (0,0), (0,1): (0,0), (1,0): (0,0), (1,1): (0,0)},
        (0,1): {(0,0): (0,0), (0,1): (1,0), (1,0): (1,1), (1,1): (1,0)},
        (1,0): {(0,0): (0,0), (0,1): (1,1), (1,0): (1,0), (1,1): (1,1)},
        (1,1): {(0,0): (0,0), (0,1): (1,0), (1,0): (1,1), (1,1): (0,1)},
    }
    
    def gf4_add(a, b):
        return (a[0] ^ b[0], a[1] ^ b[1])
    
    def gf4_mul(a, b):
        return gf4_mult[a][b]
    
    def gf4_neg(a):
        return a  # In GF(4), -a = a for characteristic 2
    
    # Map 0..23 to GF(4)^3
    point_to_gf4 = {}
    gf4_to_point = {}
    count = 0
    for x in [(0,0), (0,1), (1,0), (1,1)]:
        for y in [(0,0), (0,1), (1,0), (1,1)]:
            for z in [(0,0), (0,1), (1,0), (1,1)]:
                if count < 24:
                    point_to_gf4[count] = (x, y, z)
                    gf4_to_point[(x, y, z)] = count
                    count += 1
    
    # Generate structured blocks:
    # 1. All 7-subsets from affine lines extended
    # 2. Structured patterns based on symmetry
    block_set = set()
    
    # Add all possible 7-subsets from structured patterns
    # Pattern 1: Take a point and all points at "distance" patterns
    # Pattern 2: Use the 24-point structure to create balanced blocks
    
    # For efficiency, generate a diverse pool of ~5000-10000 blocks
    # that covers the space well, then let greedy select optimally
    
    # Generate blocks from affine subspaces of GF(4)^3 restricted to our 24 points
    # An affine line in GF(4)^3 has 4 points; take unions of lines
    
    # Simpler: generate random-like but structured blocks
    # Use the symmetry: for each point, generate blocks containing it with structured partners
    
    all_points = list(range(v))
    
    # Generate pool: combine structured and random blocks
    structured_blocks = []
    
    # For each point, create blocks with points from different "directions"
    for center in all_points:
        center_gf4 = point_to_gf4[center]
        # Get points in different "directions" from center
        directions = set()
        for other in all_points:
            if other == center:
                continue
            other_gf4 = point_to_gf4[other]
            # Direction = other - center in GF(4)^3
            dx = gf4_add(other_gf4[0], center_gf4[0])
            dy = gf4_add(other_gf4[1], center_gf4[1])
            dz = gf4_add(other_gf4[2], center_gf4[2])
            if (dx, dy, dz) != (0,0):
                directions.add((dx, dy, dz))
        
        # Pick 6 points from different directions
        dir_list = list(directions)
        random.seed(center * 7919 + 13)  # Deterministic per center
        random.shuffle(dir_list)
        
        selected = [center]
        used_points = {center}
        for d in dir_list:
            if len(selected) >= k:
                break
            # Find a point in this direction not already selected
            for other in all_points:
                if other in used_points:
                    continue
                other_gf4 = point_to_gf4[other]
                dx = gf4_add(other_gf4[0], center_gf4[0])
                dy = gf4_add(other_gf4[1], center_gf4[1])
                dz = gf4_add(other_gf4[2], center_gf4[2])
                if (dx, dy, dz) == d:
                    selected.append(other)
                    used_points.add(other)
                    break
        
        if len(selected) == k:
            selected.sort()
            block_set.add(tuple(selected))
    
    # Add some random blocks for diversity
    rng = random.Random(42)
    for _ in range(3000):
        block = tuple(sorted(rng.sample(all_points, k)))
        block_set.add(block)
    
    # Also add some greedy-friendly blocks: for each triple, add blocks containing it
    # This ensures good coverage in the pool
    for triple in triples:
        if time.time() > deadline - 60:
            break
        # Add a few blocks containing this triple
        remaining = [p for p in all_points if p not in triple]
        for _ in range(3):
            extra = rng.sample(remaining, k - t)
            block = tuple(sorted(list(triple) + extra))
            block_set.add(block)
    
    blocks = list(block_set)
    
    # Compute block masks
    block_masks = []
    for block in blocks:
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[tuple(triple)]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    if time.time() > deadline - 30:
        # Fallback: use all possible blocks
        blocks = list(itertools.combinations(range(v), k))
        block_masks = []
        for block in blocks:
            mask = 0
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                mask |= (1 << idx)
            block_masks.append(mask)
    
    # Greedy Set Cover with bitmask
    uncovered = (1 << num_triples) - 1
    selected_indices = []
    remaining_indices = list(range(len(blocks)))
    
    while uncovered:
        if time.time() > deadline - 20:
            # Emergency: finish quickly with remaining uncovered
            # Just pick blocks greedily without full optimization
            while uncovered:
                best_idx = -1
                best_gain = 0
                for i in remaining_indices:
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_gain == 0:
                    break
                selected_indices.append(best_idx)
                uncovered &= ~block_masks[best_idx]
                remaining_indices.remove(best_idx)
            break
        
        best_idx = -1
        best_gain = 0
        
        for i in remaining_indices:
            gain = (uncovered & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        remaining_indices.remove(best_idx)
    
    # Local search: remove redundant blocks
    selected_blocks = [blocks[i] for i in selected_indices]
    n_sel = len(selected_blocks)
    
    if n_sel == 0:
        return []
    
    # Build coverage counts
    triple_count = defaultdict(int)
    block_triples = []
    for b in selected_blocks:
        trs = [frozenset(c) for c in itertools.combinations(b, t)]
        block_triples.append(trs)
        for tr in trs:
            triple_count[tr] += 1
    
    # Remove redundant blocks (multiple passes)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(selected_blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            redundant = True
            for tr in block_triples[i]:
                if triple_count[tr] <= 1:
                    redundant = False
                    break
            if redundant:
                for tr in block_triples[i]:
                    triple_count[tr] -= 1
                    if triple_count[tr] == 0:
                        del triple_count[tr]
                selected_blocks.pop(i)
                block_triples.pop(i)
                improved = True
    
    # Try 2-for-1 replacement
    if time.time() < deadline - 10:
        n = len(selected_blocks)
        replaced = True
        while replaced and time.time() < deadline - 10:
            replaced = False
            for i in range(n - 1):
                if time.time() > deadline - 10:
                    break
                for j in range(n - 1, i, -1):
                    if time.time() > deadline - 10:
                        break
                    union_triples = set(block_triples[i]) | set(block_triples[j])
                    points = set()
                    for tr in union_triples:
                        points.update(tr)
                    if len(points) <= k:
                        new_block = sorted(points)
                        if len(new_block) < k:
                            for p in range(v):
                                if p not in new_block:
                                    new_block.append(p)
                                if len(new_block) == k:
                                    break
                        new_block = sorted(new_block[:k])
                        new_block_set = set(new_block)
                        covers_all = all(
                            set(tr).issubset(new_block_set) 
                            for tr in union_triples
                        )
                        if covers_all:
                            # Remove i and j, add new_block
                            for tr in block_triples[i]:
                                triple_count[tr] -= 1
                                if triple_count[tr] == 0:
                                    del triple_count[tr]
                            for tr in block_triples[j]:
                                triple_count[tr] -= 1
                                if triple_count[tr] == 0:
                                    del triple_count[tr]
                            for tr in itertools.combinations(new_block, t):
                                triple_count[frozenset(tr)] += 1
                            selected_blocks[i] = new_block
                            block_triples[i] = [frozenset(c) for c in itertools.combinations(new_block, t)]
                            selected_blocks.pop(j)
                            block_triples.pop(j)
                            n -= 1
                            replaced = True
                            break
                if replaced:
                    break
    
    # Final redundancy removal
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(selected_blocks) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                redundant = True
                for tr in block_triples[i]:
                    if triple_count[tr] <= 1:
                        redundant = False
                        break
                if redundant:
                    for tr in block_triples[i]:
                        triple_count[tr] -= 1
                        if triple_count[tr] == 0:
                            del triple_count[tr]
                    selected_blocks.pop(i)
                    block_triples.pop(i)
                    improved = True
    
    return selected_blocks
