import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    triple_set = set(all_triples)
    
    # Precompute all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    block_set = set(all_blocks)
    
    # For each block, precompute the set of triples it covers
    block_to_triples = {}
    for b in all_blocks:
        block_to_triples[b] = set(itertools.combinations(b, t))
    
    # Precompute for each triple, which blocks cover it
    triple_to_blocks = defaultdict(set)
    for b in all_blocks:
        for tri in block_to_triples[b]:
            triple_to_blocks[tri].add(b)
    
    # Greedy construction with bitset acceleration
    def greedy_construct():
        # Start with a random subset of uncovered triples
        uncovered = set(all_triples)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start from
            start_triple = rng.choice(list(uncovered))
            
            # Build a block containing this triple that maximizes coverage of remaining uncovered triples
            best_block = None
            best_gain = -1
            
            # Get all blocks that contain the start triple
            candidate_blocks = list(triple_to_blocks[start_triple])
            rng.shuffle(candidate_blocks)
            
            # Limit candidates for speed
            max_candidates = 500
            if len(candidate_blocks) > max_candidates:
                candidate_blocks = candidate_blocks[:max_candidates]
            
            for b in candidate_blocks:
                if b in blocks:
                    continue
                gain = 0
                for tri in block_to_triples[b]:
                    if tri in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None:
                # Fallback: pick any block containing start_triple
                best_block = candidate_blocks[0] if candidate_blocks else None
                if best_block is None:
                    break
            
            blocks.append(best_block)
            
            # Update uncovered
            for tri in block_to_triples[best_block]:
                if tri in uncovered:
                    uncovered.discard(tri)
            
            if not uncovered:
                break
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        # If we still have uncovered triples (due to time limit), add blocks greedily
        while uncovered and time.time() < deadline - 2:
            start_triple = rng.choice(list(uncovered))
            candidate_blocks = list(triple_to_blocks[start_triple])
            rng.shuffle(candidate_blocks)
            for b in candidate_blocks:
                if b not in blocks:
                    blocks.append(b)
                    for tri in block_to_triples[b]:
                        uncovered.discard(tri)
                    break
        
        return blocks
    
    def remove_redundant(blocks):
        """Remove blocks that are redundant (all their triples are covered by other blocks)"""
        if not blocks:
            return []
        
        # Count how many times each triple is covered
        triple_counts = defaultdict(int)
        for b in blocks:
            for tri in block_to_triples[b]:
                triple_counts[tri] += 1
        
        active = [True] * len(blocks)
        changed = True
        
        while changed and time.time() < deadline - 2:
            changed = False
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                b = blocks[i]
                can_remove = True
                for tri in block_to_triples[b]:
                    if triple_counts[tri] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active[i] = False
                    changed = True
                    for tri in block_to_triples[b]:
                        triple_counts[tri] -= 1
        
        return [blocks[i] for i in range(len(blocks)) if active[i]]
    
    def swap_optimize(blocks, max_swaps=100):
        """Try to replace a block with another block that doesn't increase the total count
        but may improve redundancy for future removals"""
        if not blocks:
            return blocks
        
        triple_counts = defaultdict(int)
        for b in blocks:
            for tri in block_to_triples[b]:
                triple_counts[tri] += 1
        
        improved = True
        swap_count = 0
        
        while improved and swap_count < max_swaps and time.time() < deadline - 3:
            improved = False
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                b_old = blocks[i]
                
                # Try replacing b_old with a different block
                # Find blocks that cover the same "essential" triples
                # A triple is essential for b_old if it's only covered by b_old
                essential_triples = set()
                for tri in block_to_triples[b_old]:
                    if triple_counts[tri] == 1:
                        essential_triples.add(tri)
                
                if not essential_triples:
                    # b_old is already redundant, skip
                    continue
                
                # Find candidate blocks that cover all essential triples
                # and are different from b_old
                candidates = set()
                for tri in essential_triples:
                    candidates &= triple_to_blocks[tri]
                candidates.discard(b_old)
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                b_new = rng.choice(list(candidates))
                
                # Check if this swap is beneficial
                # Remove b_old, add b_new, see if we can remove more blocks
                new_blocks = list(blocks)
                new_blocks[i] = b_new
                
                # Recount
                new_counts = defaultdict(int)
                for b in new_blocks:
                    for tri in block_to_triples[b]:
                        new_counts[tri] += 1
                
                # Check if all triples are still covered
                valid = True
                for tri in all_triples:
                    if new_counts[tri] < 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Try to remove redundant blocks from new_blocks
                temp_active = [True] * len(new_blocks)
                temp_changed = True
                while temp_changed:
                    temp_changed = False
                    for j in range(len(new_blocks)):
                        if not temp_active[j]:
                            continue
                        b_j = new_blocks[j]
                        can_remove = True
                        for tri in block_to_triples[b_j]:
                            if new_counts[tri] <= 1:
                                can_remove = False
                                break
                        if can_remove:
                            temp_active[j] = False
                            temp_changed = True
                            for tri in block_to_triples[b_j]:
                                new_counts[tri] -= 1
                
                new_block_count = sum(temp_active)
                old_block_count = len(blocks)
                
                if new_block_count < old_block_count:
                    blocks = [new_blocks[j] for j in range(len(new_blocks)) if temp_active[j]]
                    triple_counts = new_counts
                    improved = True
                    swap_count += 1
                    break  # Restart the loop with the new set of blocks
        
        return blocks
    
    # Main search loop
    best_blocks = None
    
    # First, do a quick greedy construction to get a baseline
    baseline = greedy_construct()
    baseline = remove_redundant(baseline)
    best_blocks = baseline
    
    if best_blocks is not None and len(best_blocks) <= 69:
        return best_blocks
    
    # Now try multiple rounds of greedy + optimization
    while time.time() < deadline - 5:
        # Diversified greedy construction
        cand = greedy_construct()
        cand = remove_redundant(cand)
        
        # Try swap optimization
        if time.time() < deadline - 5:
            cand = swap_optimize(cand, max_swaps=50)
        
        if best_blocks is None or len(cand) < len(best_blocks):
            best_blocks = cand
            
            if len(best_blocks) <= 69:
                break
        elif len(cand) == len(best_blocks) and time.time() < deadline - 10:
            # Same size, try to merge and optimize
            # Take union of both sets, then remove redundancies
            merged = list(set(best_blocks + cand))
            merged = remove_redundant(merged)
            if len(merged) < len(best_blocks):
                best_blocks = merged
                if len(best_blocks) <= 69:
                    break
    
    # Final cleanup
    if best_blocks is not None:
        best_blocks = remove_redundant(best_blocks)
    
    # Fallback: if we ran out of time and best_blocks is incomplete, finish it
    if best_blocks is not None:
        covered = set()
        for b in best_blocks:
            for tri in block_to_triples[b]:
                covered.add(tri)
        uncovered = triple_set - covered
        while uncovered and time.time() < deadline - 1:
            start_triple = rng.choice(list(uncovered))
            for b in triple_to_blocks[start_triple]:
                if b not in best_blocks:
                    best_blocks.append(b)
                    for tri in block_to_triples[b]:
                        uncovered.discard(tri)
                    break
    
    return best_blocks

if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Found {len(blocks)} blocks")
    
    # Verify
    covered = set()
    for b in blocks:
        for tri in itertools.combinations(b, 3):
            covered.add(tri)
    
    all_tris = set(itertools.combinations(range(24), 3))
    missing = all_tris - covered
    if missing:
        print(f"Invalid! Missing {len(missing)} triples")
    else:
        print("Valid covering!")
