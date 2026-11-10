"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

P = {'n': 12, 'm': 40}

def _precompute_comparability(n):
    N = 1 << n
    C = [0] * N
    # subsets[i] is just i (bitmask)
    for i in range(N):
        for j in range(i + 1, N):
            # Check if i and j are comparable
            # i < j or j < i
            # (i & j) == i implies i subset j
            # (i & j) == j implies j subset i
            if (i & j) == i or (i & j) == j:
                C[i] |= (1 << j)
                C[j] |= (1 << i)
    return C

def _optimize_family(C, N, m, restarts=10, iterations_per_restart=20000):
    best_overall_score = -1
    best_overall_selected = None
    
    for r in range(restarts):
        rng = random.Random(1000 + r)
        # Initialize with random m subsets
        selected = set(rng.sample(range(N), m))
        selected_mask = 0
        for s in selected:
            selected_mask |= (1 << s)
        
        sel_list = list(selected)
        
        # Initial score
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if C[sel_list[i]] & (1 << sel_list[j]):
                    score += 1
        
        current_score = score
        current_selected = set(selected)
        current_mask = selected_mask
        current_list = sel_list
        
        for _ in range(iterations_per_restart):
            # Pick A from selected
            idx_a = rng.randrange(m)
            a = current_list[idx_a]
            
            # Pick B from unselected
            while True:
                b = rng.randrange(N)
                if b not in current_selected:
                    break
            
            # Calculate Delta
            # deg_a = number of neighbors of a in current_selected
            deg_a = (C[a] & current_mask).bit_count()
            
            # deg_b_new = number of neighbors of b in (current_selected \ {a} U {b})
            # Neighbors of b in current_selected:
            deg_b_in_current = (C[b] & current_mask).bit_count()
            # If a is a neighbor of b, we lose that connection
            is_neighbor = 1 if (C[b] & (1 << a)) else 0
            deg_b_new = deg_b_in_current - is_neighbor
            
            delta = deg_b_new - deg_a
            
            if delta > 0:
                # Swap
                current_selected.remove(a)
                current_selected.add(b)
                current_mask &= ~(1 << a)
                current_mask |= (1 << b)
                current_list[idx_a] = b
                current_score += delta
                
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_selected = set(current_selected)
                    
    return best_overall_selected, best_overall_score

def construct():
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute comparability matrix
    C = _precompute_comparability(n)
    
    # Optimize
    selected_mask, score = _optimize_family(C, N, m)
    
    # Convert masks to lists of integers
    result = []
    for mask in selected_mask:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
        
    # Ensure exactly m distinct subsets
    assert len(result) == m, f"Expected {m} subsets, got {len(result)}"
    assert len(set(tuple(s) for s in result)) == m, "Subsets are not distinct"
    
    return result
# EVOLVE-BLOCK-END
