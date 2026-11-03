import itertools
import heapq
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Generate all triples and blocks
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # 2. Precompute incidence
    # triples_for_block[j] = list of triple indices covered by block j
    triples_for_block = [[] for _ in range(n_blocks)]
    # blocks_for_triple[i] = list of block indices covering triple i
    blocks_for_triple = [[] for _ in range(n_triples)]
    
    for j, block in enumerate(blocks):
        for triple in itertools.combinations(block, t):
            i = triple_to_idx[tuple(triple)]
            triples_for_block[j].append(i)
            blocks_for_triple[i].append(j)

    # 3. Greedy Set Cover
    # coverage[j] = number of uncovered triples covered by block j
    coverage = [35] * n_blocks  # C(k, t) = C(7,3) = 35
    uncovered = [True] * n_triples
    
    # Heap: (-coverage, block_id)
    heap = [(-35, j) for j in range(n_blocks)]
    heapq.heapify(heap)
    
    solution = []
    
    while True:
        if time.time() > deadline:
            break
            
        # Get best block
        valid_block = None
        while heap:
            neg_cov, j = heapq.heappop(heap)
            cov = -neg_cov
            if cov == coverage[j]:
                valid_block = j
                break
        
        if valid_block is None:
            break
            
        if coverage[valid_block] == 0:
            break
            
        solution.append(valid_block)
        
        # Update coverage
        for i in triples_for_block[valid_block]:
            if uncovered[i]:
                uncovered[i] = False
                for j in blocks_for_triple[i]:
                    coverage[j] -= 1
                    heapq.heappush(heap, (-coverage[j], j))
                    
        # Check if all covered
        if not any(uncovered):
            break

    # 4. Pruning
    # Count coverage in solution
    cover_count = [0] * n_triples
    for j in solution:
        for i in triples_for_block[j]:
            cover_count[i] += 1
            
    # Try to remove blocks
    # Iterate in reverse order of addition? Or random?
    # Reverse is often good as later blocks are more "specialized"
    new_solution = []
    # We need to be careful with indices if we remove.
    # Let's just mark for removal.
    to_remove = set()
    
    # Sort solution by some heuristic? 
    # Blocks that cover fewer unique triples are more likely to be removable.
    # Unique triples for block j: count of i in triples_for_block[j] where cover_count[i] == 1
    # But cover_count changes as we remove.
    
    # Simple iterative removal:
    # Repeat until no change
    changed = True
    current_solution = list(solution)
    while changed and time.time() < deadline:
        changed = False
        # Recalculate cover_count for current_solution
        cover_count = [0] * n_triples
        for j in current_solution:
            for i in triples_for_block[j]:
                cover_count[i] += 1
                
        next_solution = []
        for j in current_solution:
            # Check if j can be removed
            removable = True
            for i in triples_for_block[j]:
                if cover_count[i] == 1:
                    removable = False
                    break
            if removable:
                # Remove j
                for i in triples_for_block[j]:
                    cover_count[i] -= 1
                changed = True
            else:
                next_solution.append(j)
                
        current_solution = next_solution
        
    return [list(blocks[j]) for j in current_solution]
