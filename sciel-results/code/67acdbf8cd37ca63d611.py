"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_triple_indices(n):
    """Build list of all triples and a mapping from triple to index."""
    triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    return triples, triple_to_idx

def _build_constraint_indices(n):
    """Build index arrays for quad constraints (4 triples each) and quint constraints (10 triples each)."""
    triples, triple_to_idx = _build_triple_indices(n)
    
    # Quad constraints: each 4-set has 4 triples
    quad_constraints = []
    for quad in combinations(range(n), 4):
        ts = [
            (quad[0], quad[1], quad[2]),
            (quad[0], quad[1], quad[3]),
            (quad[0], quad[2], quad[3]),
            (quad[1], quad[2], quad[3]),
        ]
        quad_constraints.append([triple_to_idx[t] for t in ts])
    
    # Quint constraints: each 5-set has 10 triples
    quint_constraints = []
    for quint in combinations(range(n), 5):
        ts = []
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    ts.append((quint[i], quint[j], quint[k]))
        quint_constraints.append([triple_to_idx[t] for t in ts])
    
    return np.array(quad_constraints, dtype=np.int32), np.array(quint_constraints, dtype=np.int32)

def _count_violations_vec(colors, quad_idx, quint_idx):
    """Count violations using vectorized NumPy operations.
    
    Red violation: a quad where all 4 triples are 0 (red)
    Blue violation: a quint where all 10 triples are 1 (blue)
    """
    # For quads: check if all 4 entries are 0
    quad_colors = colors[quad_idx]  # shape (num_quads, 4)
    red_viol = np.all(quad_colors == 0, axis=1).sum()
    
    # For quints: check if all 10 entries are 1
    quint_colors = colors[quint_idx]  # shape (num_quints, 10)
    blue_viol = np.all(quint_colors == 1, axis=1).sum()
    
    return int(red_viol + blue_viol)

def _local_search_vectorized(n, max_iters, deadline, seed=42, init_colors=None, quad_idx=None, quint_idx=None):
    """Vectorized local search to find a valid 2-coloring."""
    rng = np.random.RandomState(seed)
    
    num_triples = n * (n - 1) * (n - 2) // 6
    
    # Build constraint indices if not provided
    if quad_idx is None or quint_idx is None:
        quad_idx, quint_idx = _build_constraint_indices(n)
    
    # Initialize colors
    if init_colors is not None:
        colors = np.array(init_colors, dtype=np.int8)
    else:
        colors = rng.randint(0, 2, size=num_triples).astype(np.int8)
    
    cur_viol = _count_violations_vec(colors, quad_idx, quint_idx)
    best_colors = colors.copy()
    best_viol = cur_viol
    
    T = 1.0
    T_min = 0.0001
    cooling = 0.99
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if cur_viol == 0:
            break
        
        # Pick a random triple to flip
        idx = rng.randint(0, num_triples)
        old_color = colors[idx]
        colors[idx] = 1 - old_color
        
        new_viol = _count_violations_vec(colors, quad_idx, quint_idx)
        
        delta = new_viol - cur_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            cur_viol = new_viol
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = colors.copy()
        else:
            colors[idx] = old_color
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    if best_viol == 0:
        return best_colors.tolist()
    return None

def _algebraic_init(n):
    """Try algebraic initialization based on known constructions.
    
    One known approach: use vertices as elements of Z_m x Z_k, and color
    based on some algebraic property of the triple sum.
    
    For R(4,5;3) > 34, known constructions use specific algebraic structures.
    Let's try a few different algebraic colorings.
    """
    triples = list(combinations(range(n), 3))
    
    # Strategy 1: color based on (a+b+c) mod 3
    colors1 = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    
    # Strategy 2: color based on (a+b+c) mod 2
    colors2 = [0 if (a + b + c) % 2 == 0 else 1 for a, b, c in triples]
    
    # Strategy 3: color based on whether a,b,c form an arithmetic progression
    colors3 = []
    for a, b, c in triples:
        if 2 * b == a + c:
            colors3.append(0)
        else:
            colors3.append(1)
    
    # Strategy 4: use a hash-based approach with structured bias
    colors4 = []
    for a, b, c in triples:
        # Make red less likely to create K4
        val = (a * 7 + b * 13 + c * 19) % 5
        colors4.append(0 if val < 2 else 1)
    
    # Strategy 5: biased random with structure
    rng = np.random.RandomState(42)
    colors5 = []
    for a, b, c in triples:
        # Bias towards blue (1) to avoid blue K5, but not too much
        r = rng.random()
        if r < 0.45:
            colors5.append(0)
        else:
            colors5.append(1)
    
    return [colors1, colors2, colors3, colors4, colors5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with a small valid coloring
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try n from 34 down to 10
    # Prioritize higher n as they yield higher scores
    for n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break
        
        # Build indices once for this n
        quad_idx, quint_idx = _build_constraint_indices(n)
        
        # Try algebraic inits first
        inits = _algebraic_init(n)
        found = False
        
        for seed_idx, init in enumerate(inits):
            if time.time() > deadline - 2:
                break
            
            # Quick verification of algebraic init
            colors = np.array(init, dtype=np.int8)
            viol = _count_violations_vec(colors, quad_idx, quint_idx)
            
            if viol == 0:
                best_n = n
                best_colors = init
                found = True
                break
        
        if found:
            # Found a valid coloring for this n, return immediately
            return {'n': best_n, 'colors': best_colors}
        
        # If algebraic failed, try local search with limited budget
        # Use a small budget for each n to ensure we try multiple n values
        search_deadline = min(deadline - 2, time.time() + 5)
        
        for seed_idx in range(3): # Try 3 random restarts
            if time.time() > search_deadline:
                break
            seed = 1000 + n * 10 + seed_idx
            colors = _local_search_vectorized(n, max_iters=500, deadline=search_deadline, seed=seed, init_colors=None, quad_idx=quad_idx, quint_idx=quint_idx)
            if colors is not None:
                best_n = n
                best_colors = colors
                return {'n': best_n, 'colors': best_colors}
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
