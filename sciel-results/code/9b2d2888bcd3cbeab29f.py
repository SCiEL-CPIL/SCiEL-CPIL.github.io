import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    # Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_triples[b] = list of triple indices covered by block b
    block_triples = [[] for _ in range(num_blocks)]
    # blocks_containing_triple[t] = list of block indices that cover triple t
    blocks_containing_triple = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            block_triples[b_idx].append(idx)
            blocks_containing_triple[idx].append(b_idx)
            
    best_solution = None
    best_count = float('inf')
    
    # Run multiple randomized greedy + local search iterations
    for run in range(50):
        random.seed(run)
        
        # 1. Greedy Construction
        uncovered_set = set(range(num_triples))
        solution = []
        
        while uncovered_set:
            # Pick a random uncovered triple to anchor the search
            seed_triple = random.choice(list(uncovered_set))
            
            best_b_idx = -1
            max_covered = 0
            candidates = []
            
            # Look at all blocks containing this seed triple
            for b_idx in blocks_containing_triple[seed_triple]:
                # Count how many uncovered triples this block covers
                # Optimization: if the block covers the seed, it covers at least 1.
                # We check intersection with uncovered_set
                count = 0
                # Fast path: if we have many uncovered, checking all triples in block is fine
                # block size is C(7,3)=35 triples max.
                for t_idx in block_triples[b_idx]:
                    if t_idx in uncovered_set:
                        count += 1
                        
                if count > max_covered:
                    max_covered = count
                    candidates = [b_idx]
                elif count == max_covered:
                    candidates.append(b_idx)
            
            if not candidates:
                # Should not happen if uncovered_set is not empty
                break
                
            chosen_b_idx = random.choice(candidates)
            solution.append(chosen_b_idx)
            
            # Update uncovered_set
            for t_idx in block_triples[chosen_b_idx]:
                if t_idx in uncovered_set:
                    uncovered_set.remove(t_idx)
                    
        # 2. Local Search: Remove redundant blocks
        # Calculate coverage count for each triple
        cover_counts = [0] * num_triples
        for b_idx in solution:
            for t_idx in block_triples[b_idx]:
                cover_counts[t_idx] += 1
                
        # Iteratively remove blocks that are not needed
        # We do this by checking if a block's removal would leave any triple with count 0
        # i.e., a block is redundant if for all t in block, cover_counts[t] > 1
        
        changed = True
        while changed:
            changed = False
            # Shuffle solution to break symmetry/randomize removal order
            random.shuffle(solution)
            for i in range(len(solution)):
                b_idx = solution[i]
                is_redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_counts[t_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove block
                    for t_idx in block_triples[b_idx]:
                        cover_counts[t_idx] -= 1
                    solution[i] = None
                    changed = True
                    
            # Clean up None values
            solution = [b for b in solution if b is not None]
            
        if len(solution) < best_count:
            best_count = len(solution)
            best_solution = solution
            
    # Convert indices back to lists of points
    if best_solution is None:
        # Fallback: return a valid but potentially large covering if greedy failed completely
        # But greedy should always succeed
        return []
        
    return [list(all_blocks[b_idx]) for b_idx in best_solution]
