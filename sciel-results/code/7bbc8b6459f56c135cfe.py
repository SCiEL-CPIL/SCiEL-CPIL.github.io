"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    deadline = time.time() + 55.0
    
    # Identify edge types
    # Prism graph: two m-cycles (0..m-1, m..2m-1) + matching (i, i+m)
    # Edges are ordered list of (u, v)
    # Let's classify edges
    cycle1_edges = []  # edges in first cycle (0..m-1)
    cycle2_edges = []  # edges in second cycle (m..2m-1)
    matching_edges = []  # matching edges (i, i+m)
    
    for idx, (u, v) in enumerate(edges):
        u_min = min(u, v)
        u_max = max(u, v)
        if u < m and v < m:
            cycle1_edges.append(idx)
        elif u >= m and v >= m:
            cycle2_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    best_signs = None
    best_radius = float('inf')
    
    def try_signs(signs):
        nonlocal best_signs, best_radius
        if time.time() > deadline:
            return
        try:
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        except Exception:
            pass
    
    def local_search(signs, max_flips=100):
        nonlocal best_signs, best_radius
        current = signs.copy()
        current_radius = best_radius if np.array_equal(current, best_signs) else float('inf')
        
        # Compute current radius if not known
        if current_radius == float('inf'):
            try:
                current_radius, _ = check_lift(graph, current)
            except Exception:
                return current
        
        for _ in range(max_flips):
            if time.time() > deadline:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                # Try flipping edge i
                trial = current.copy()
                trial[i] = -trial[i]
                try:
                    r, _ = check_lift(graph, trial)
                    if r < current_radius - 1e-12:
                        current = trial
                        current_radius = r
                        improved = True
                        break
                except Exception:
                    pass
            if not improved:
                break
        
        if np.array_equal(current, best_signs) or current_radius < best_radius:
            best_radius = current_radius
            best_signs = current.copy()
        return current
    
    # Strategy 1: Symmetry-reduced patterns
    # Pattern A: All cycle edges +1, matching edges vary
    # Pattern B: Alternating cycle signs, matching edges vary
    
    # Generate candidate sign vectors
    candidates = []
    
    # All +1
    candidates.append(np.ones(num_edges, dtype=int))
    
    # All -1
    candidates.append(-np.ones(num_edges, dtype=int))
    
    # Constant cycle signs, varying matching
    for c1 in [1, -1]:
        for c2 in [1, -1]:
            for cmatch in [1, -1]:
                signs = np.ones(num_edges, dtype=int)
                for idx in cycle1_edges:
                    signs[idx] = c1
                for idx in cycle2_edges:
                    signs[idx] = c2
                for idx in matching_edges:
                    signs[idx] = cmatch
                candidates.append(signs)
    
    # Alternating cycle signs (for even m)
    if m % 2 == 0:
        for c1_start in [1, -1]:
            for c2_start in [1, -1]:
                for cmatch in [1, -1]:
                    signs = np.ones(num_edges, dtype=int)
                    for j, idx in enumerate(cycle1_edges):
                        # Need to figure out the position in the cycle
                        # cycle1_edges is ordered, so j is the position
                        signs[idx] = c1_start * (1 if j % 2 == 0 else -1)
                    for j, idx in enumerate(cycle2_edges):
                        signs[idx] = c2_start * (1 if j % 2 == 0 else -1)
                    for idx in matching_edges:
                        signs[idx] = cmatch
                    candidates.append(signs)
    
    # Fourier harmonics: signs based on cos(2*pi*k*j/m)
    for k in range(1, m // 2 + 1):
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            signs = np.ones(num_edges, dtype=int)
            # Map edge index to position in cycle
            # cycle1_edges[j] corresponds to edge from j to (j+1)%m
            for j, idx in enumerate(cycle1_edges):
                val = np.cos(2 * np.pi * k * j / m + phase)
                signs[idx] = 1 if val >= 0 else -1
            for j, idx in enumerate(cycle2_edges):
                val = np.cos(2 * np.pi * k * j / m + phase)
                signs[idx] = 1 if val >= 0 else -1
            for idx in matching_edges:
                signs[idx] = 1  # default
            candidates.append(signs)
            
            # Also try with matching signs varying
            for cmatch in [-1]:
                signs2 = signs.copy()
                for idx in matching_edges:
                    signs2[idx] = cmatch
                candidates.append(signs2)
    
    # Try all candidates
    for signs in candidates:
        if time.time() > deadline:
            break
        try_signs(signs)
    
    # Now do local search from the best candidate found
    if best_signs is not None:
        local_search(best_signs, max_flips=200)
    
    # If we still have time, try more random starts
    if time.time() < deadline - 5:
        rng = np.random.default_rng(123)
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            try_signs(signs)
        
        # Local search from best
        if best_signs is not None:
            local_search(best_signs, max_flips=100)
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
