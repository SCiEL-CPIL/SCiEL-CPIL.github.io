"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_structure(n):
    """
    Precompute the structure needed for fast local search.
    Returns:
    - num_triples
    - triples: list of (a,b,c)
    - triple_to_idx
    - quad_constraints: list of lists of triple indices for each 4-set
    - quint_constraints: list of lists of triple indices for each 5-set
    - triple_to_quads: for each triple index, list of quad indices containing it
    - triple_to_quints: for each triple index, list of quint indices containing it
    """
    triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    num_triples = len(triples)
    
    # Build quad and quint constraints
    quad_constraints = []
    for quad in combinations(range(n), 4):
        ts = [
            (quad[0], quad[1], quad[2]),
            (quad[0], quad[1], quad[3]),
            (quad[0], quad[2], quad[3]),
            (quad[1], quad[2], quad[3]),
        ]
        quad_constraints.append([triple_to_idx[t] for t in ts])
        
    quint_constraints = []
    for quint in combinations(range(n), 5):
        ts = []
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    ts.append((quint[i], quint[j], quint[k]))
        quint_constraints.append([triple_to_idx[t] for t in ts])

    # Build reverse indices: triple -> constraints
    triple_to_quads = [[] for _ in range(num_triples)]
    for q_idx, quad_triples in enumerate(quad_constraints):
        for t_idx in quad_triples:
            triple_to_quads[t_idx].append(q_idx)
            
    triple_to_quints = [[] for _ in range(num_triples)]
    for q_idx, quint_triples in enumerate(quint_constraints):
        for t_idx in quint_triples:
            triple_to_quints[t_idx].append(q_idx)

    return num_triples, triples, triple_to_idx, quad_constraints, quint_constraints, triple_to_quads, triple_to_quints

def _count_violations_fast(colors, quad_constraints, quint_constraints):
    """
    Initial count of violations.
    Red violation: Quad with 4 reds (count 4)
    Blue violation: Quint with 10 blues (count 10)
    """
    red_viol = 0
    for qt in quad_constraints:
        if all(colors[i] == 0 for i in qt):
            red_viol += 1
            
    blue_viol = 0
    for qt in quint_constraints:
        if all(colors[i] == 1 for i in qt):
            blue_viol += 1
            
    return red_viol + blue_viol

def _get_initial_counts(colors, quad_constraints, quint_constraints):
    """
    Calculate the number of red triples in each quad and blue triples in each quint.
    """
    quad_counts = [0] * len(quad_constraints)
    for q_idx, qt in enumerate(quad_constraints):
        for t_idx in qt:
            if colors[t_idx] == 0:
                quad_counts[q_idx] += 1
                
    quint_counts = [0] * len(quint_constraints)
    for q_idx, qt in enumerate(quint_constraints):
        for t_idx in qt:
            if colors[t_idx] == 1:
                quint_counts[q_idx] += 1
                
    return quad_counts, quint_counts

def _local_search_fast(n, max_iters, deadline, seed=42, init_colors=None, structure=None):
    """
    Fast local search using delta updates.
    """
    if structure is None:
        structure = _build_structure(n)
    num_triples, triples, triple_to_idx, quad_constraints, quint_constraints, triple_to_quads, triple_to_quints = structure
    
    rng = np.random.RandomState(seed)
    
    # Initialize colors
    if init_colors is not None:
        colors = np.array(init_colors, dtype=np.int8)
    else:
        # Biased initialization: slightly favor blue to avoid blue K5? 
        # Actually, random is fine, but we can try to balance.
        colors = rng.randint(0, 2, size=num_triples).astype(np.int8)
    
    # Get initial counts
    quad_counts, quint_counts = _get_initial_counts(colors, quad_constraints, quint_constraints)
    
    # Calculate initial violations
    red_viol = sum(1 for c in quad_counts if c == 4)
    blue_viol = sum(1 for c in quint_counts if c == 10)
    cur_viol = red_viol + blue_viol
    
    best_colors = colors.copy()
    best_viol = cur_viol
    
    T = 1.0
    T_min = 0.0001
    cooling = 0.99
    
    # Identify triples involved in violations for targeted search
    # We can recompute this periodically or just use random + greedy
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if cur_viol == 0:
            break
        
        # Strategy: Pick a triple involved in a violation if possible, else random
        # For efficiency, we'll just pick random but with a "kick" if stuck
        idx = rng.randint(0, num_triples)
        
        old_color = colors[idx]
        new_color = 1 - old_color
        
        # Calculate delta
        delta = 0
        
        # Update quad counts (Red count)
        # If old_color was 0 (Red), we are removing a Red.
        # If old_color was 1 (Blue), we are adding a Red.
        if old_color == 0:
            # Red -> Blue
            for q_idx in triple_to_quads[idx]:
                old_c = quad_counts[q_idx]
                new_c = old_c - 1
                # If old was 4, it was a violation. Now it's 3, so violation resolved.
                if old_c == 4:
                    delta -= 1
                # Can it create a new violation? No, reducing count doesn't create 'all red'.
                quad_counts[q_idx] = new_c
        else:
            # Blue -> Red
            for q_idx in triple_to_quads[idx]:
                old_c = quad_counts[q_idx]
                new_c = old_c + 1
                # If new is 4, it becomes a violation.
                if new_c == 4:
                    delta += 1
                quad_counts[q_idx] = new_c

        # Update quint counts (Blue count)
        # If old_color was 1 (Blue), we are removing a Blue.
        # If old_color was 0 (Red), we are adding a Blue.
        if old_color == 1:
            # Blue -> Red
            for q_idx in triple_to_quints[idx]:
                old_c = quint_counts[q_idx]
                new_c = old_c - 1
                # If old was 10, it was a violation. Now it's 9, so violation resolved.
                if old_c == 10:
                    delta -= 1
                # Can it create a new violation? No, reducing blue count doesn't create 'all blue'.
                quint_counts[q_idx] = new_c
        else:
            # Red -> Blue
            for q_idx in triple_to_quints[idx]:
                old_c = quint_counts[q_idx]
                new_c = old_c + 1
                # If new is 10, it becomes a violation.
                if new_c == 10:
                    delta += 1
                quint_counts[q_idx] = new_c

        # Apply Metropolis criterion
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept
            colors[idx] = new_color
            cur_viol += delta
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = colors.copy()
        else:
            # Reject: Revert counts
            if old_color == 0:
                # We did Red->Blue, so we reverted Blue->Red
                for q_idx in triple_to_quads[idx]:
                    quad_counts[q_idx] += 1
                for q_idx in triple_to_quints[idx]:
                    quint_counts[q_idx] -= 1
            else:
                # We did Blue->Red, so we reverted Red->Blue
                for q_idx in triple_to_quads[idx]:
                    quad_counts[q_idx] -= 1
                for q_idx in triple_to_quints[idx]:
                    quint_counts[q_idx] += 1
            # colors[idx] remains old_color
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_viol == 0:
        return best_colors.tolist()
    return None

def _algebraic_init(n):
    """
    Generate several initial colorings based on algebraic structures.
    """
    triples = list(combinations(range(n), 3))
    inits = []
    
    # 1. Mod 3 sum
    c1 = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    inits.append(c1)
    
    # 2. Mod 2 sum
    c2 = [0 if (a + b + c) % 2 == 0 else 1 for a, b, c in triples]
    inits.append(c2)
    
    # 3. Product mod 2 (even number of odds?)
    c3 = [0 if (a % 2 + b % 2 + c % 2) % 2 == 0 else 1 for a, b, c in triples]
    inits.append(c3)
    
    # 4. Random with fixed seed
    rng = np.random.RandomState(42)
    c4 = rng.randint(0, 2, size=len(triples)).tolist()
    inits.append(c4)
    
    # 5. Random with different seed
    rng2 = np.random.RandomState(123)
    c5 = rng2.randint(0, 2, size=len(triples)).tolist()
    inits.append(c5)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # We want to maximize n.
    # Try n=34 first, then 33, etc.
    # The published construction has 34 points.
    
    best_n = 0
    best_colors = []
    
    # Pre-compute structures for n=34, 33, 32 to save time if we need to fallback
    # But building structure is O(n^3) or O(n^4) for quints.
    # For n=34, quints are C(34,5) = 278256. Building lists is fast.
    
    # Try n from 34 down to 20
    # We allocate more time to higher n
    
    for n in range(34, 19, -1):
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Allocate time based on n
        # Give 34 more time
        if n == 34:
            alloc_time = min(time_left - 2, 60)
        elif n == 33:
            alloc_time = min(time_left - 2, 40)
        else:
            alloc_time = min(time_left - 2, 20)
            
        if alloc_time < 2:
            continue
            
        structure = _build_structure(n)
        
        found = False
        # Try different seeds and inits
        seeds = [42, 123, 999, 777, 555, 111, 222, 333]
        
        # Try algebraic inits first
        inits = _algebraic_init(n)
        
        # Check if any algebraic init is valid
        for init in inits:
            colors = np.array(init, dtype=np.int8)
            viol = _count_violations_fast(colors, structure[3], structure[4])
            if viol == 0:
                best_n = n
                best_colors = init
                return {'n': best_n, 'colors': best_colors}
                
        # Run local search with multiple restarts
        # Use the time budget
        search_deadline = time.time() + alloc_time
        
        # We can run multiple local searches in parallel? No, single thread.
        # We'll run sequential restarts.
        
        # Estimate max iters: 
        # Each iter is very fast (O(1) update). 
        # Let's just run until deadline.
        
        for seed in seeds:
            if time.time() > search_deadline:
                break
                
            # Use a specific init for this seed? Or random?
            # Random init is usually fine with good local search.
            # But we can try to start from the best algebraic init if it was close?
            # For simplicity, start from random, but use the seed.
            
            # Actually, let's try starting from the algebraic inits as well for local search
            # if they had low violations.
            
            # For now, just random init with different seeds
            colors = _local_search_fast(n, max_iters=1000000, deadline=search_deadline, seed=seed, init_colors=None, structure=structure)
            
            if colors is not None:
                best_n = n
                best_colors = colors
                return {'n': best_n, 'colors': best_colors}
        
        # If not found, continue to lower n
        
    # Fallback: return a small valid coloring if we failed to find large ones
    # n=9 is known to be valid with simple mod 3
    if best_n == 0:
        best_n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
        
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
