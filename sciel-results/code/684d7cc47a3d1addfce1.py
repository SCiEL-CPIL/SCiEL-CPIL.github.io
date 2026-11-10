import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    n = 16
    all_points = list(range(n))
    
    # Precompute all triples
    triple_list = []
    mask_to_idx = {}
    idx_to_mask = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                idx = len(triple_list)
                triple_list.append(mask)
                mask_to_idx[mask] = idx
                idx_to_mask[idx] = mask
                
    num_triples = len(triple_list)
    
    # Precompute conflicts
    # conflicts[i] is a set of indices j such that triple i and j share >= 2 points
    conflicts = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        mi = triple_list[i]
        for j in range(i + 1, num_triples):
            mj = triple_list[j]
            if bin(mi & mj).count('1') >= 2:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Precompute masks by size for alpha calculation
    # We mainly care about sizes 3, 4, 5, 6
    masks_by_size = {k: [] for k in range(3, 7)}
    for mask in range(1 << n):
        k = bin(mask).count('1')
        if k in masks_by_size:
            masks_by_size[k].append(mask)
            
    # Shuffle masks for randomization in alpha check
    for k in masks_by_size:
        random.shuffle(masks_by_size[k])

    def is_independent(mask, edges_set):
        """
        Check if 'mask' (a set of points) is an independent set w.r.t edges_set.
        i.e., no edge in edges_set is a subset of mask.
        """
        for e_idx in edges_set:
            e_mask = idx_to_mask[e_idx]
            if (mask & e_mask) == e_mask:
                return False
        return True
        
    def calc_alpha(edges_set):
        if not edges_set:
            return n
            
        # Check size 6
        # If we find an independent set of size 6, alpha >= 6
        # We can limit the check for speed if we just want to know if it's <= 4 or 5
        # But let's be accurate.
        
        # Optimization: If we find an independent set of size K, we can stop checking smaller sizes for the "max" value?
        # No, we want the MAX size.
        # So we check large sizes first.
        
        # Check 6
        for m in masks_by_size[6]:
            if is_independent(m, edges_set):
                return 6
                
        # Check 5
        for m in masks_by_size[5]:
            if is_independent(m, edges_set):
                return 5
                
        # Check 4
        for m in masks_by_size[4]:
            if is_independent(m, edges_set):
                return 4
                
        # Check 3
        for m in masks_by_size[3]:
            if is_independent(m, edges_set):
                return 3
                
        return 2

    def is_valid_add(triple_idx, edges_set):
        # Check if triple_idx conflicts with any in edges_set
        for c in conflicts[triple_idx]:
            if c in edges_set:
                return False
        return True

    def greedy_pack():
        current_triples = set()
        available = list(range(num_triples))
        random.shuffle(available)
        
        for idx in available:
            if is_valid_add(idx, current_triples):
                current_triples.add(idx)
        
        return current_triples

    def random_swap(current_triples):
        """
        Perform a random swap: remove one edge, add one valid edge.
        """
        if not current_triples:
            return None
            
        to_remove = random.choice(list(current_triples))
        new_triples = set(current_triples)
        new_triples.remove(to_remove)
        
        # Find a valid add
        # Try random candidates
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            if idx in new_triples:
                continue
            if is_valid_add(idx, new_triples):
                new_triples.add(idx)
                return new_triples
        return None

    best_alpha = n
    best_triples = None
    
    # Initial Greedy
    current_triples = greedy_pack()
    current_alpha = calc_alpha(current_triples)
    
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_triples = set(current_triples)
        
    # Local Search with Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        # Calculate alpha
        current_alpha = calc_alpha(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = set(current_triples)
            
            # If we reach 3, it's very good. We can keep searching to ensure robustness or just stop if time is tight?
            # Let's keep searching to see if we can find 3 consistently or improve further (impossible if 3 is min).
            
        # Generate Move
        new_triples = random_swap(current_triples)
        
        if new_triples is None:
            # Stuck? Perturb
            # Remove 2 random edges
            if current_triples:
                for _ in range(2):
                    if current_triples:
                        current_triples.remove(random.choice(list(current_triples)))
            continue
            
        new_alpha = calc_alpha(new_triples)
        
        # Acceptance
        delta = new_alpha - current_alpha
        if delta <= 0:
            current_triples = new_triples
            current_alpha = new_alpha
            # Reset T if improved
            if delta < 0:
                T = max(T, 0.5)
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < (2.71828 ** (-delta / T)):
                current_triples = new_triples
                current_alpha = new_alpha
                
        T *= cooling_rate
        if T < min_T:
            T = min_T

    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    # Convert indices back to points
    result = []
    for idx in best_triples:
        mask = idx_to_mask[idx]
        pts = []
        for i in range(n):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result
