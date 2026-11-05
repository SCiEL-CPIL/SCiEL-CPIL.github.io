"""Covering design C(24,7,3) using bitset-accelerated greedy + local search."""
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute: map each (t-1)-subset to the list of points that can extend it to a t-subset
    # Actually, we need to track uncovered t-subsets efficiently.
    # Use a set of frozensets or tuples for uncovered triples.
    
    # For v=24, k=7, t=3: C(24,3) = 2024 triples
    all_triples = list(__import__('itertools').combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Bitset representation of uncovered triples
    # Use numpy uint64 array: 2024 triples needs 32 uint64s
    n_words = (n_triples + 63) // 64
    uncovered = np.ones(n_words, dtype=np.uint64)
    
    def is_uncovered(triple):
        i = triple_to_idx[triple]
        word = i // 64
        bit = i % 64
        return (uncovered[word] >> bit) & 1
        
    def mark_covered(triple):
        i = triple_to_idx[triple]
        word = i // 64
        bit = i % 64
        uncovered[word] &= ~(1 << bit)
        
    def count_uncovered():
        return int(uncovered.sum() * 8)  # rough, actually use popcount
    
    # Better: count actual uncovered
    def uncovered_count():
        total = 0
        for w in uncovered:
            total += bin(int(w)).count('1')
        return total

    # For a partial block (list of points), compute gain of adding point x
    def gain(x, block):
        # Count how many currently-uncovered triples would be covered by adding x
        # A triple (a,b,x) where {a,b} is in block and a != b
        g = 0
        block_set = set(block)
        for i in range(len(block)):
            for j in range(i+1, len(block)):
                a, b = block[i], block[j]
                triple = tuple(sorted((a, b, x)))
                if triple in triple_to_idx:
                    idx = triple_to_idx[triple]
                    word = idx // 64
                    bit = idx % 64
                    if (uncovered[word] >> bit) & 1:
                        g += 1
        return g

    # Greedy construction
    def greedy_construct():
        unc = np.ones(n_words, dtype=np.uint64)
        blocks = []
        
        def is_unc(triple):
            i = triple_to_idx[triple]
            return (unc[i // 64] >> (i % 64)) & 1
            
        def mark_unc(triple):
            i = triple_to_idx[triple]
            unc[i // 64] &= ~(1 << (i % 64))
            
        def unc_count():
            total = 0
            for w in unc:
                total += bin(int(w)).count('1')
            return total
        
        # Start with all triples uncovered
        # Greedy: find a block that covers max uncovered triples
        # We need to build blocks one at a time
        
        # Precompute all k-subsets? C(24,7) = 346104 - too many to enumerate all
        # Instead: greedily build each block
        # For each uncovered triple, try to extend it to a full block
        
        while True:
            if unc_count() == 0:
                break
            
            # Find an uncovered triple to start
            # Pick the first uncovered triple
            found = False
            for i in range(n_triples):
                if (unc[i // 64] >> (i % 64)) & 1:
                    start_triple = all_triples[i]
                    found = True
                    break
            if not found:
                break
            
            block = list(start_triple)
            
            # Greedily add points to maximize uncovered coverage
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    g = 0
                    for a in block:
                        for b in block:
                            if a == b:
                                continue
                            triple = tuple(sorted((a, b, x)))
                            if triple in triple_to_idx:
                                idx = triple_to_idx[triple]
                                if (unc[idx // 64] >> (idx % 64)) & 1:
                                    g += 1
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Mark all triples in this block as covered
            from itertools import combinations
            for tr in combinations(block, t):
                mark_unc(tr)
            
            if time.time() > deadline - 5:
                break
        
        return blocks

    # Local search: try to remove blocks
    def local_search(blocks):
        if not blocks:
            return blocks
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try removing each block
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                # Check if removing block i still covers all triples
                # Build coverage count for all triples
                from collections import defaultdict
                cover_count = defaultdict(int)
                for j, block in enumerate(blocks):
                    if j == i:
                        continue
                    from itertools import combinations
                    for tr in combinations(block, t):
                        cover_count[tr] += 1
                
                # Check if any triple has cover_count 0
                all_covered = True
                for tr in all_triples:
                    if cover_count[tr] == 0:
                        all_covered = False
                        break
                
                if all_covered:
                    blocks.pop(i)
                    improved = True
                    break
        
        return blocks

    # Try to merge or swap to reduce block count further
    def local_search_swap(blocks):
        if len(blocks) < 2:
            return blocks
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try replacing one block with a different one
            # For each block, try random alternatives
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                
                # Generate a random alternative block
                from itertools import combinations
                # Try a few random blocks
                for attempt in range(20):
                    if time.time() > deadline - 5:
                        break
                    new_block = sorted(random.sample(range(v), k))
                    old_block = blocks[i]
                    
                    # Check if replacing improves coverage
                    # Build coverage with old block removed, check if new block helps
                    from collections import defaultdict
                    cover_count = defaultdict(int)
                    for j, block in enumerate(blocks):
                        if j == i:
                            continue
                        for tr in combinations(block, t):
                            cover_count[tr] += 1
                    
                    # Triples only covered by old block
                    old_only = []
                    for tr in combinations(old_block, t):
                        if cover_count[tr] == 0:
                            old_only.append(tr)
                    
                    if not old_only:
                        # Can just remove the block
                        blocks.pop(i)
                        improved = True
                        break
                    
                    # Check if new block covers all old_only triples
                    new_triples = set(combinations(new_block, t))
                    if all(tr in new_triples for tr in old_only):
                        blocks[i] = list(new_block)
                        improved = True
                        break
        
        return blocks

    # Main algorithm
    blocks = greedy_construct()
    
    if time.time() < deadline - 10:
        blocks = local_search(blocks)
    
    if time.time() < deadline - 5:
        blocks = local_search_swap(blocks)
    
    # Final cleanup: remove redundant blocks
    if time.time() < deadline - 2:
        blocks = local_search(blocks)
    
    return blocks
