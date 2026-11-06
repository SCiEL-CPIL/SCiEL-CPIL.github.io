import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute masks for all triples
    triple_masks = []
    for t in all_triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)
        
    num_triples = len(all_triples)
    
    # Precompute compatibility as bitsets for fast checking
    # compat[i] is a bitmask of triples compatible with triple i
    compat = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[i]
        mask = 0
        for j in range(num_triples):
            if i == j:
                continue
            mj = triple_masks[j]
            if (mi & mj).bit_count() <= 1:
                mask |= (1 << j)
        compat[i] = mask
        
    # Fast alpha computation using a greedy + verification approach
    def compute_alpha(system_indices):
        """Compute independence number of the hypergraph defined by system_indices."""
        if not system_indices:
            return 16
            
        sys_masks = [triple_masks[i] for i in system_indices]
        n = len(sys_masks)
        
        # For each point, find which triples contain it
        point_to_triples = [[] for _ in range(16)]
        for idx, m in enumerate(sys_masks):
            for p in range(16):
                if m & (1 << p):
                    point_to_triples[p].append(idx)
        
        # A subset S is independent if for every triple in system, 
        # not all 3 points of the triple are in S.
        # Equivalently, S is independent iff for every triple t, |S ∩ t| <= 2.
        
        # We want the largest S. Let's use a max-flow or ILP-like approach.
        # Since 16 points is small, we can use a branch-and-bound or 
        # simply check via maximum independent set on a derived graph.
        
        # Alternative: use the fact that alpha = 16 - minimum vertex cover of hypergraph
        # Minimum vertex cover of a 3-uniform hypergraph is NP-hard, but 16 points is tiny.
        
        # Let's use a simple approach: try all subsets from size 16 down,
        # but with early termination and pruning.
        
        # Better: use a recursive search with memoization on the bitmask of remaining points.
        # State: which points are still "available" to be in the independent set.
        # But we need to ensure no triple is fully contained.
        
        # Actually, let's just do the subset enumeration but more efficiently.
        # Precompute for each point bitmask the set of triples that are "satisfied" (fully contained).
        
        # For a candidate set S (as a 16-bit mask), S is independent iff
        # for all t in sys_masks: (S & t) != t
        
        # Let's enumerate subsets by size, starting from large.
        # C(16,8) = 12870, C(16,9) = 11440, etc.
        
        # Use a smarter approach: start with all 16 points, then remove points 
        # that are in "conflicting" triples.
        
        # Greedy heuristic: start with all points, while there's a triple fully in S,
        # remove a point from that triple (the one in the most such triples).
        
        S_mask = (1 << 16) - 1  # all 16 points
        
        # Count how many times each point appears in "active" triples (triples fully in S)
        while True:
            # Find all triples fully contained in S
            active = []
            for m in sys_masks:
                if (S_mask & m) == m:
                    active.append(m)
            
            if not active:
                break
            
            # For each point in S, count how many active triples it's in
            point_count = [0] * 16
            for m in active:
                for p in range(16):
                    if m & (1 << p):
                        point_count[p] += 1
            
            # Remove the point in S that's in the most active triples
            best_p = -1
            best_count = -1
            for p in range(16):
                if S_mask & (1 << p):
                    if point_count[p] > best_count:
                        best_count = point_count[p]
                        best_p = p
            
            if best_p >= 0:
                S_mask &= ~(1 << best_p)
        
        # S_mask is now an independent set. Return its size.
        return S_mask.bit_count()
    
    # Even faster: verify a specific size k
    def has_independent_set_of_size_k(system_indices, k):
        """Check if there's an independent set of size k."""
        if k > 16:
            return False
        if k == 16:
            # Check if any triple exists
            if not system_indices:
                return True
            return False
        
        sys_masks = [triple_masks[i] for i in system_indices]
        
        # Try all subsets of size k
        for subset in itertools.combinations(range(16), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            
            is_independent = True
            for sm in sys_masks:
                if (mask & sm) == sm:
                    is_independent = False
                    break
            
            if is_independent:
                return True
        return False
    
    def get_alpha_exact(system_indices):
        """Exact alpha by checking from 16 down."""
        if not system_indices:
            return 16
        for k in range(16, 0, -1):
            if has_independent_set_of_size_k(system_indices, k):
                return k
        return 0
    
    # Greedy construction with randomization
    def greedy_build():
        """Build a linear triple system using random greedy."""
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        current_system = []
        
        for idx in indices:
            # Check if idx is compatible with all in current_system
            compatible = True
            for existing_idx in current_system:
                if not (compat[existing_idx] & (1 << idx)):
                    compatible = False
                    break
            
            if compatible:
                current_system.append(idx)
        
        return current_system
    
    # Local search: try swapping triples
    def local_search(system, max_swaps=50):
        """Try to improve the system by swapping triples."""
        if not system:
            return system
        
        best_alpha = get_alpha_exact(system)
        current = system[:]
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            # Try removing a random triple and adding a different one
            if not current:
                break
                
            remove_idx = random.choice(current)
            remove_pos = current.index(remove_idx)
            
            # Find a triple that's compatible with all others in current (minus remove_idx)
            others = [x for i, x in enumerate(current) if i != remove_pos]
            
            candidates = []
            for j in range(num_triples):
                if j == remove_idx:
                    continue
                if j in others:
                    continue
                # Check compatibility with all others
                ok = True
                for o in others:
                    if not (compat[o] & (1 << j)):
                        ok = False
                        break
                if ok:
                    candidates.append(j)
            
            if not candidates:
                continue
            
            new_idx = random.choice(candidates)
            new_system = current[:]
            new_system[remove_pos] = new_idx
            
            new_alpha = get_alpha_exact(new_system)
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                current = new_system
            else:
                # Sometimes accept worse solutions for diversity (simulated annealing-like)
                if random.random() < 0.1:
                    current = new_system
        
        return current
    
    best_system = None
    best_alpha = 17
    
    # Multiple random restarts with local search
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Greedy build
        system = greedy_build()
        
        # Local search improvement
        system = local_search(system, max_swaps=30)
        
        alpha = get_alpha_exact(system)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = system[:]
            
            if best_alpha <= 6:
                # Good enough, but try to improve a bit more
                if best_alpha <= 5:
                    break
        
        # Periodic check
        if iterations % 100 == 0 and time.time() > deadline - 5:
            break
    
    if best_system is None:
        return []
        
    # Convert indices back to triples
    return [list(all_triples[i]) for i in best_system]
