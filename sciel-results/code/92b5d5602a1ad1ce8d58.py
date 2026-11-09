import time
import random
import sys

DEADLINE = 115.0  # safety margin before 120s kill

def _edge_index(i, j):
    """Map pair (i,j), 0<=i<j<16, to index in the 120-element list."""
    # Lexicographic order: (0,1), (0,2), ..., (0,15), (1,2), ...
    # Index = sum_{k=0}^{i-1} (15-k) + (j-1-i)
    #       = i*15 - i*(i-1)/2 + j - 1 - i
    #       = i*15 - i*(i-1)//2 + j - 1 - i
    return i * 15 - i * (i - 1) // 2 + j - 1 - i

def _precompute_k4s():
    """Return list of 6-tuples of edge indices for each K4."""
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    edges = (
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    )
                    k4s.append(edges)
    return k4s

_K4S = _precompute_k4s()

def _count_violations(colours):
    """Count number of K4s that span fewer than 5 distinct colours."""
    violations = 0
    for edges in _K4S:
        # Unpack for speed
        c0, c1, c2, c3, c4, c5 = (colours[e] for e in edges)
        # Check distinct count without creating a set if possible?
        # Set is fine for 6 elements.
        if len({c0, c1, c2, c3, c4, c5}) < 5:
            violations += 1
    return violations

def _num_colours_used(colours):
    return len(set(colours))

def _is_valid(colours):
    return _count_violations(colours) == 0

def _initial_colouring():
    """(i+j) mod 17 construction."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _try_merge(colours, c1, c2):
    """Try to merge colour c1 into c2. Returns new colours if valid, else None."""
    if c1 == c2:
        return None
    
    # Create a copy
    new_colours = list(colours)
    
    # Replace c1 with c2
    # Optimization: Only modify if c1 exists
    if c1 not in new_colours:
        return None
        
    for i in range(120):
        if new_colours[i] == c1:
            new_colours[i] = c2
            
    if _is_valid(new_colours):
        return new_colours
    return None

def _reduce_colours_by_merging(colours, deadline):
    """Iteratively try to merge colours to reduce count."""
    current = list(colours)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        used_colours = sorted(set(current))
        
        # Try all pairs of used colours
        for i in range(len(used_colours)):
            for j in range(i + 1, len(used_colours)):
                c1 = used_colours[i]
                c2 = used_colours[j]
                new_colours = _try_merge(current, c1, c2)
                if new_colours is not None:
                    current = new_colours
                    improved = True
                    # Restart the outer loop to pick up the new set of used colours
                    break
            if improved:
                break
    return current

def _simulated_annealing(colours, deadline):
    """
    SA to escape local minima in the merging process.
    Move: Change colour of a random edge to a random other used colour.
    Objective: violations * 1000 + num_colours
    """
    current = list(colours)
    current_violations = _count_violations(current)
    current_colours = _num_colours_used(current)
    current_obj = current_violations * 1000 + current_colours
    
    best = list(current)
    best_obj = current_obj
    
    T = 100.0
    T_min = 1.0
    cooling = 0.999
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Pick a random edge
        idx = rng.randrange(120)
        old_c = current[idx]
        
        # Pick a new colour from the currently used colours (excluding old_c)
        used = set(current)
        if len(used) <= 1:
            break
            
        candidates = [c for c in used if c != old_c]
        if not candidates:
            continue
            
        new_c = rng.choice(candidates)
        
        current[idx] = new_c
        
        # Calculate new objective
        new_violations = _count_violations(current)
        new_colours = _num_colours_used(current)
        new_obj = new_violations * 1000 + new_colours
        
        delta = new_obj - current_obj
        
        # Accept if better, or with probability exp(-delta/T)
        if delta <= 0 or (delta < 0 and False) or rng.random() < min(1.0, -delta / T if delta > 0 else 1.0):
             # Correction: Standard SA accepts if delta < 0 OR rng < exp(-delta/T)
             # If delta > 0, exp(-delta/T) is small.
             if delta <= 0:
                 accept = True
             else:
                 accept = rng.random() < (2.718281828 ** (-delta / T))
                 
             if accept:
                 current_obj = new_obj
                 current_violations = new_violations
                 current_colours = new_colours
                 
                 if new_obj < best_obj:
                     best_obj = new_obj
                     best = list(current)
             else:
                 current[idx] = old_c
        else:
            current[idx] = old_c
            
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Periodic merge attempt
        if rng.random() < 0.01: # 1% chance to try merging
            # Only try if we are valid
            if current_violations == 0:
                new_merged = _reduce_colours_by_merging(current, time.time() + 0.1)
                if new_merged is not None:
                    # Check if this improved the best
                    m_viol = _count_violations(new_merged)
                    m_col = _num_colours_used(new_merged)
                    m_obj = m_viol * 1000 + m_col
                    if m_obj < best_obj:
                        best_obj = m_obj
                        best = new_merged
                        current = list(best)
                        current_obj = best_obj
                        current_violations = m_viol
                        current_colours = m_col
                        T = 100.0 # Reheat

    return best, best_obj

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # Start from valid 17-colour construction
    colours = _initial_colouring()
    
    # Verify validity
    if _count_violations(colours) != 0:
        # Should not happen, but just in case
        pass
        
    best = list(colours)
    best_obj = _num_colours_used(colours)
    
    # Phase 1: Greedy merging
    # This is very fast and should reduce colours significantly
    merged = _reduce_colours_by_merging(colours, deadline)
    if merged is not None:
        m_obj = _num_colours_used(merged)
        if m_obj < best_obj:
            best = merged
            best_obj = m_obj
            
    # Phase 2: Simulated Annealing
    # SA can help escape local minima where no single merge works, 
    # but a sequence of moves might allow a merge later.
    if time.time() < deadline:
        sa_result, sa_obj = _simulated_annealing(best, deadline)
        # sa_obj includes violation penalty, so we need to check validity
        if _count_violations(sa_result) == 0:
            sa_colours = _num_colours_used(sa_result)
            if sa_colours < best_obj:
                best = sa_result
                best_obj = sa_colours
                
    # Final check
    if _count_violations(best) != 0:
        best = _initial_colouring()
        
    return best

# Pre-compute K4s at module load time for speed
_K4S = _precompute_k4s()
