import random
import time
import itertools

N = 16
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute pairs for each triple
TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    pairs = list(itertools.combinations(t, 2))
    TRIPLE_PAIRS.append(pairs)

# Precompute conflicts: for each triple, which other triples share a pair with it?
# conflicts[i] is a set of indices j such that triple i and triple j share a pair.
CONFLICTS = [set() for _ in range(NUM_TRIPLES)]
# To speed up construction, map pair -> list of triples containing it
pair_to_triples = {}
for i, t in enumerate(ALL_TRIPLES):
    for p in TRIPLE_PAIRS[i]:
        if p not in pair_to_triples:
            pair_to_triples[p] = []
        pair_to_triples[p].append(i)

for i in range(NUM_TRIPLES):
    for p in TRIPLE_PAIRS[i]:
        for j in pair_to_triples[p]:
            if i != j:
                CONFLICTS[i].add(j)

def _compute_alpha(triples_indices):
    """Compute independence number for the hypergraph defined by triples_indices."""
    if not triples_indices:
        return N
    
    # Build the hypergraph edges as masks
    edge_masks = []
    for idx in triples_indices:
        t = ALL_TRIPLES[idx]
        mask = 0
        for v in t:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    n_edges = len(edge_masks)
    
    # Check subsets from largest to smallest
    # Optimization: If we find an independent set of size k, we return k.
    # We can use a recursive backtracking or simple iteration.
    # Given N=16, iteration over all subsets is 2^16 = 65536.
    # For each subset, checking against ~40 edges is fast.
    
    # We want the LARGEST subset with NO edge contained in it.
    # This is equivalent to: Find max k such that there exists a subset S of size k where no edge is a subset of S.
    
    # We can iterate k from N down to 1.
    for k in range(N, 0, -1):
        # Generate all subsets of size k
        for subset in itertools.combinations(range(N), k):
            s_mask = 0
            for v in subset:
                s_mask |= (1 << v)
            
            is_independent = True
            for em in edge_masks:
                if (s_mask & em) == em:
                    is_independent = False
                    break
            
            if is_independent:
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_alpha = N + 1
    best_triples = None
    
    def try_update(indices):
        nonlocal best_alpha, best_triples
        if not indices:
            return
        # Check linearity (should be guaranteed by construction, but just in case)
        # Actually, we construct it to be linear.
        alpha = _compute_alpha(indices)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(ALL_TRIPLES[i]) for i in indices]

    def greedy_init():
        indices = list(range(NUM_TRIPLES))
        random.shuffle(indices)
        current_set = set()
        used_pairs = set()
        
        for idx in indices:
            # Check if idx can be added
            can_add = True
            for p in TRIPLE_PAIRS[idx]:
                if p in used_pairs:
                    can_add = False
                    break
            
            if can_add:
                current_set.add(idx)
                for p in TRIPLE_PAIRS[idx]:
                    used_pairs.add(p)
        
        return list(current_set), used_pairs

    def local_search(indices, used_pairs, max_iters=500):
        current_set = set(indices)
        current_used = set(used_pairs)
        current_size = len(current_set)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = random.choice(['add', 'swap', 'kick'])
            
            if move_type == 'add':
                # Try to add a random node
                # Pick a few candidates
                for _ in range(10):
                    idx = random.randint(0, NUM_TRIPLES - 1)
                    if idx in current_set:
                        continue
                    
                    can_add = True
                    for p in TRIPLE_PAIRS[idx]:
                        if p in current_used:
                            can_add = False
                            break
                    
                    if can_add:
                        current_set.add(idx)
                        for p in TRIPLE_PAIRS[idx]:
                            current_used.add(p)
                        current_size += 1
                        break # Restart loop with new size? Or just continue
            
            elif move_type == 'swap' and current_size > 0:
                # Remove one, try to add another
                idx_remove = random.choice(list(current_set))
                # Tentatively remove
                for p in TRIPLE_PAIRS[idx_remove]:
                    current_used.discard(p)
                current_set.discard(idx_remove)
                
                added = False
                # Try to add a random node
                for _ in range(20):
                    idx_add = random.randint(0, NUM_TRIPLES - 1)
                    if idx_add in current_set:
                        continue
                    
                    can_add = True
                    for p in TRIPLE_PAIRS[idx_add]:
                        if p in current_used:
                            can_add = False
                            break
                    
                    if can_add:
                        current_set.add(idx_add)
                        for p in TRIPLE_PAIRS[idx_add]:
                            current_used.add(p)
                        added = True
                        break
                
                if not added:
                    # Revert
                    current_set.add(idx_remove)
                    for p in TRIPLE_PAIRS[idx_remove]:
                        current_used.add(p)
                    
            elif move_type == 'kick' and current_size > 5:
                # Remove a node and all its neighbors, then add the node
                idx_kick = random.choice(list(current_set))
                
                # Find all neighbors in current_set
                to_remove = {idx_kick}
                for neighbor in CONFLICTS[idx_kick]:
                    if neighbor in current_set:
                        to_remove.add(neighbor)
                
                # Remove them
                for r_idx in to_remove:
                    for p in TRIPLE_PAIRS[r_idx]:
                        current_used.discard(p)
                    current_set.discard(r_idx)
                
                # Add idx_kick back
                # It should be compatible now because we removed its conflicts
                current_set.add(idx_kick)
                for p in TRIPLE_PAIRS[idx_kick]:
                    current_used.add(p)
        
        return list(current_set), current_used, current_size

    # Run multiple restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        init_indices, init_used = greedy_init()
        
        # Local search
        search_indices, search_used, size = local_search(init_indices, init_used, max_iters=200)
        
        # Only evaluate alpha for reasonably large systems
        if size > 30: 
            try_update(search_indices)
        
        # Early stop if we found a very good solution
        if best_alpha <= 5:
            break

    if best_triples is None:
        # Fallback
        init_indices, _ = greedy_init()
        best_triples = [list(ALL_TRIPLES[i]) for i in init_indices]
        
    return best_triples
