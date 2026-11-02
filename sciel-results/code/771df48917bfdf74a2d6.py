import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    all_t_set = set(all_t_subsets)
    
    # Precompute: for each point, which triples contain it (as a tuple for set lookup)
    point_to_triples = {p: [] for p in all_points}
    for triple in all_t_subsets:
        for p in triple:
            point_to_triples[p].append(triple)
    
    # Precompute combinations for gain calculation
    # For a block of size b, adding point x covers triples that include x and (t-1) points from block
    # We'll compute gains on the fly using the precomputed mapping
    
    def random_greedy_cover():
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple
            seed_triple = random.choice(list(uncovered))
            block = list(seed_triple)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_points = []
                
                for x in all_points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # For t=3: triples with x and 2 points from block
                    # For general t: triples with x and (t-1) points from block
                    for sub in itertools.combinations(block, t - 1):
                        triple = tuple(sorted((x,) + sub))
                        if triple in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_points:
                    block.append(random.choice(best_points))
                    block_set.add(block[-1])
                else:
                    available = [x for x in all_points if x not in block_set]
                    if available:
                        block.append(random.choice(available))
                        block_set.add(block[-1])
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            for triple in itertools.combinations(block, t):
                uncovered.discard(tuple(triple))
        
        return blocks
    
    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        n_blocks = len(blocks)
        
        # Build coverage map: triple -> list of block indices
        triple_to_blocks = {triple: [] for triple in all_t_subsets}
        for i, block in enumerate(blocks):
            for triple in itertools.combinations(block, t):
                triple_to_blocks[tuple(triple)].append(i)
        
        changed = True
        while changed:
            changed = False
            
            # Phase 1: Remove redundant blocks
            to_remove = set()
            for i, block in enumerate(blocks):
                if i in to_remove:
                    continue
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    covering = triple_to_blocks[tuple(triple)]
                    if len(covering) == 1 and covering[0] == i:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.add(i)
            
            if to_remove:
                new_blocks = [b for i, b in enumerate(blocks) if i not in to_remove]
                blocks = new_blocks
                # Rebuild coverage map
                triple_to_blocks = {triple: [] for triple in all_t_subsets}
                for i, block in enumerate(blocks):
                    for triple in itertools.combinations(block, t):
                        triple_to_blocks[tuple(triple)].append(i)
                changed = True
                continue
            
            # Phase 2: Try block replacement (swap) - replace one block with a better one
            # This is more expensive, so we only do it if we have time
            if time.time() > deadline - 5:
                break
            
            n_blocks = len(blocks)
            found_improvement = False
            
            # Try to replace each block with a new block that covers more uncovered triples
            # while maintaining coverage
            for i in range(n_blocks):
                if time.time() > deadline - 5:
                    break
                
                # Compute which triples are covered ONLY by block i
                exclusive_triples = []
                for triple in itertools.combinations(blocks[i], t):
                    if len(triple_to_blocks[tuple(triple)]) == 1:
                        exclusive_triples.append(tuple(triple))
                
                if not exclusive_triples:
                    # Block i is fully redundant, should have been caught in Phase 1
                    # But maybe it wasn't because of ordering; skip
                    continue
                
                # Try to find a replacement block that covers all exclusive_triples
                # and possibly more
                # Heuristic: pick a random exclusive triple, build a block around it
                seed = random.choice(exclusive_triples)
                new_block = list(seed)
                new_block_set = set(new_block)
                
                # Extend to cover remaining exclusive triples
                remaining_exclusive = [tr for tr in exclusive_triples if tr not in itertools.combinations(new_block, t)]
                
                while len(new_block) < k and remaining_exclusive:
                    best_gain = -1
                    best_points = []
                    for x in all_points:
                        if x in new_block_set:
                            continue
                        gain = 0
                        for triple in remaining_exclusive:
                            if x in triple:
                                # Check if adding x would cover this triple
                                # The triple must be a subset of new_block + {x}
                                if all(p in new_block_set or p == x for p in triple):
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_points = [x]
                        elif gain == best_gain:
                            best_points.append(x)
                    
                    if best_points:
                        new_block.append(random.choice(best_points))
                        new_block_set.add(new_block[-1])
                        # Update remaining_exclusive
                        remaining_exclusive = [tr for tr in remaining_exclusive 
                                               if not all(p in new_block_set for p in tr)]
                    else:
                        break
                
                if len(new_block) < k:
                    continue
                
                new_block.sort()
                
                # Check if this replacement is valid: every triple in old block must still be covered
                # by some other block OR by the new block
                valid = True
                for triple in itertools.combinations(blocks[i], t):
                    triple_t = tuple(triple)
                    covering = triple_to_blocks[triple_t]
                    # Remove block i from covering
                    other_covering = [j for j in covering if j != i]
                    if not other_covering:
                        # Must be covered by new block
                        if not all(p in new_block_set for p in triple_t):
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Count how many new triples the new block covers that weren't covered before
                # This is optional; we just need validity
                # Accept the swap
                blocks[i] = new_block
                found_improvement = True
                
                # Rebuild coverage map
                triple_to_blocks = {triple: [] for triple in all_t_subsets}
                for j, block in enumerate(blocks):
                    for triple in itertools.combinations(block, t):
                        triple_to_blocks[tuple(triple)].append(j)
                
                changed = True
                break  # Restart the while loop after a swap
        
        return blocks
    
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        blocks = random_greedy_cover()
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    return best_blocks if best_blocks is not None else []
