import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Shape: (n_blocks, C(k, t))
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Compute how many blocks cover each triple (for weighted greedy)
    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1
    # Weight: rare triples get higher weight
    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)
    
    def greedy_cover(random_tiebreak=False):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Weighted gain: sum of weights of uncovered triples in block
            gains = np.sum(uncovered[block_triples].astype(np.float64) * triple_weights[block_triples], axis=1)
            
            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        
        return selected
    
    def remove_redundant(block_indices):
        """Remove blocks that are not needed to cover all triples."""
        if len(block_indices) == 0:
            return block_indices
        
        # For each block, check if removing it leaves all triples covered
        # Use a coverage count approach
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                bi = result[i]
                # Check if any triple in this block would become uncovered
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                # Safe to remove
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break
        
        return result
    
    def local_search(block_indices, max_iterations=50):
        """Try to replace blocks with alternatives that maintain coverage."""
        if len(block_indices) == 0:
            return block_indices
        
        # Build coverage count
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                # Try to find a replacement block that covers all critical triples
                # Critical triples are those covered only by this block
                critical = block_triples[bi]
                critical_mask = coverage[critical] <= 1
                if not np.any(critical_mask):
                    continue
                
                # Find blocks that cover all critical triples
                # A block j is valid if for all critical triples, block j covers them
                # i.e., critical triples are subset of block_triples[j]
                
                # Efficient check: for each candidate block, see if it covers all critical triples
                # critical_triples = critical[critical_mask]
                critical_triples = critical[critical_mask]
                
                # For each block j != i, check if block_triples[j] contains all critical_triples
                # We can precompute this by checking set intersection
                # But that's expensive. Alternative: just try random candidates
                
                # Try a few random candidates
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    j = np.random.randint(n_blocks)
                    if j == bi:
                        continue
                    # Check if block j covers all critical triples
                    block_j_triples = block_triples[j]
                    # Check if all critical_triples are in block_j_triples
                    # Use set for speed
                    if len(set(critical_triples.tolist()).intersection(block_j_triples.tolist())) == len(critical_triples):
                        # Valid replacement
                        coverage[block_triples[bi]] -= 1
                        coverage[block_j_triples] += 1
                        current[i] = j
                        improved = True
                        break
                if improved:
                    break
        
        return current
    
    # First greedy pass
    selected = greedy_cover(random_tiebreak=False)
    selected = remove_redundant(selected)
    selected = local_search(selected, max_iterations=20)
    
    best_solution = [list(blocks[i]) for i in selected]
    best_size = len(best_solution)
    
    # Multiple random restarts
    num_restarts = 15
    for _ in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        selected = greedy_cover(random_tiebreak=True)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=10)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]
    
    return best_solution
