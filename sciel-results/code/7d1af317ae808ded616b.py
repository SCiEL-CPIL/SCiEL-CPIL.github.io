import itertools
import random
import time
import sys

def construct(v, k, t):
    # Parameters are fixed by the task, but we keep them for interface compatibility
    # v=24, k=7, t=3
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute all triples and assign them indices
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute all blocks and their bitsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Bitset for each block: an integer where bit i is 1 if the block covers all_triples[i]
    block_bitsets = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        bset = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            bset |= (1 << idx)
        block_bitsets[i] = bset
    
    best_blocks = None
    best_count = float('inf')
    
    # We will perform multiple randomized greedy restarts
    # The greedy step: pick the block that covers the most currently uncovered triples.
    # To add randomness, we pick randomly from the top M blocks.
    
    M = 10  # Number of top candidates to consider
    
    while time.time() < deadline:
        uncovered = (1 << num_triples) - 1
        current_blocks = []
        used_blocks = set()
        
        while uncovered:
            # Find the best blocks
            # We compute the score for each unused block
            # Optimization: If we have many used blocks, we still scan all, but skip used ones.
            # Since num_blocks is ~350k, and we use ~76, skipping is negligible.
            
            scores = []
            # To avoid storing a list of 350k tuples, we can just track the top M
            # But sorting 350k items is fast enough in C (Python's sort is Timsort)
            # Let's just compute all scores and find top M.
            
            # This loop is the bottleneck.
            # local vars for speed
            bb = block_bitsets
            unc = uncovered
            ub = used_blocks
            
            # We want to find indices i such that i not in ub, and (bb[i] & unc).bit_count() is high.
            
            # A faster way: just iterate and keep a heap of size M?
            # Or just a list and sort?
            # Let's use a list of (score, index) and nlargest.
            
            # Actually, creating a list of 350k tuples is memory/time intensive.
            # Let's try to do it in a single pass with a max-heap of size M.
            
            import heapq
            heap = []
            for i in range(num_blocks):
                if i in ub:
                    continue
                s = (bb[i] & unc).bit_count()
                if s == 0:
                    continue
                # Maintain a min-heap of size M with the largest scores
                if len(heap) < M:
                    heapq.heappush(heap, (s, i))
                elif s > heap[0][0]:
                    heapq.heapreplace(heap, (s, i))
            
            if not heap:
                # Should not happen if uncovered is not 0 and we have blocks
                break
            
            # Pick one randomly from the heap
            # The heap contains the top M scores.
            # We can pick the max (heap[0] is min of the top M, so we need to find max in heap)
            # Or just pick randomly from the heap items?
            # The items in the heap are the top M. Picking randomly from them is a good strategy.
            s, idx = random.choice(heap)
            
            current_blocks.append(list(all_blocks[idx]))
            used_blocks.add(idx)
            uncovered &= ~bb[idx]
            
            # Check time inside the loop
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
        
        if uncovered == 0:
            count = len(current_blocks)
            if count < best_count:
                best_count = count
                best_blocks = current_blocks
                # If we reached the theoretical lower bound or known optimum, we could stop
                # Lower bound is 58, known is 76.
                # If we find 76, it's a tie with best known. If we find less, it's a breakthrough.
                # We should keep searching to see if we can go lower, but 76 is a strong target.
                # Given time limits, finding 76 is a win.
    
    if best_blocks is None:
        # Fallback: return a trivial covering (all blocks) if we ran out of time before finishing a single run?
        # This shouldn't happen as precomputation is fast and one run takes ~10-20s.
        # But to be safe, return an empty list or the last partial result?
        # The contract says "Valid when every t-subset ... lies in at least one block".
        # If we return an invalid covering, score is 0.
        # So we must return a valid covering.
        # If we didn't complete a single run, we can't guarantee validity.
        # However, precomputation + 1 run should fit in 120s easily.
        # If we are here, something went wrong.
        return []
        
    return best_blocks
