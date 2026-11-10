import numpy as np
import time
from itertools import product
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580  # Leave 20s safety margin
    
    # Strategy: Build a larger candidate pool with vectors of norm^2 = 4,
    # then find a large independent set in the incompatibility graph.
    # Two points p, q are compatible if ||p - q||^2 >= max(||p||^2, ||q||^2)
    # For all norm^2 = 4: compatible if ||p-q||^2 >= 4
    
    # Generate candidate vectors with norm^2 = 4
    # Types:
    # 1. (±2, 0, ..., 0): 22 vectors
    # 2. (±1, ±1, 0, ..., 0): 4 * C(11,2) = 220 vectors
    # 3. (±1, 0, ..., 0, ±1, 0, ..., ±1, 0, ..., ±1) with three 1s: 8 * C(11,3) = 8*165 = 1320
    #    Wait, that's norm^2=3, not 4.
    # Let me reconsider: norm^2=4 means sum of squares = 4
    # Options: (±2,0,...), (±1,±1,0,...), or (±1,0,...,±1,0,...,±1,0,...,±1) no that's 3
    # Actually (±1,±1,±1,0,...) has norm^2=3
    # (±2, ±1, 0,...) has norm^2=5
    # So for norm^2=4: only (±2,0,...) and (±1,±1,0,...)
    # Total: 22 + 220 = 242
    
    # But we can also use a MIX of different norms.
    # Key insight: if we use points with varying norms, the constraint is:
    # min over pairs ||pi-pj||^2 >= max over all points ||pk||^2
    # So all points must have the same max norm, and pairwise distances >= that norm.
    
    # Better approach: Use norm^2 = 4 vectors. There are 242 such vectors.
    # Find maximum independent set in the graph where edges connect incompatible pairs.
    
    candidates = []
    
    # Type 1: (±2, 0, ..., 0)
    for i in range(d):
        for sign in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * sign
            candidates.append(v)
    
    # Type 2: (±1, ±1, 0, ..., 0)
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    n = len(candidates)
    candidates = np.array(candidates, dtype=np.int64)
    
    # Build incompatibility graph: edge (i,j) if ||ci - cj||^2 < 4
    # (since all have norm^2 = 4, the constraint is ||ci-cj||^2 >= 4)
    
    # Precompute all pairwise squared distances
    # For n=242, that's ~29k pairs, very fast
    
    incompat = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = candidates[i] - candidates[j]
            dsq = np.dot(diff, diff)
            if dsq < 4:
                incompat[i, j] = True
                incompat[j, i] = True
    
    # Find maximum independent set using greedy + local search
    # Greedy: repeatedly pick vertex with fewest incompatibilities among remaining
    
    def greedy_mis(incompat, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        remaining = set(range(n))
        selected = []
        
        while remaining:
            # Pick vertex with minimum degree in induced subgraph
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and incompat[v, u])
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            
            selected.append(best_v)
            remaining.remove(best_v)
            # Remove all neighbors of best_v from remaining
            to_remove = set()
            for u in remaining:
                if incompat[best_v, u]:
                    to_remove.add(u)
            remaining -= to_remove
        
        return selected
    
    # Try multiple random orderings via weighted greedy
    best_set = None
    best_size = 0
    
    # First try pure greedy
    selected = greedy_mis(incompat, n)
    best_size = len(selected)
    best_set = selected.copy()
    
    # Try random restarts with perturbation
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Randomized greedy: at each step, pick from top-k lowest degree vertices
        remaining = set(range(n))
        selected = []
        
        while remaining:
            # Compute degrees
            degrees = {}
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and incompat[v, u])
                degrees[v] = deg
            
            min_deg = min(degrees.values())
            # Pick randomly from vertices with degree <= min_deg + 1
            candidates_v = [v for v in remaining if degrees[v] <= min_deg + 1]
            if len(candidates_v) == 1:
                best_v = candidates_v[0]
            else:
                best_v = np.random.choice(candidates_v)
            
            selected.append(best_v)
            remaining.remove(best_v)
            to_remove = set()
            for u in remaining:
                if incompat[best_v, u]:
                    to_remove.add(u)
            remaining -= to_remove
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected.copy()
    
    # Local search: try to add more points or swap
    # For now, just verify and return
    result_indices = best_set
    result = candidates[result_indices]
    
    # Final verification
    if len(result) >= 2:
        max_norm_sq = np.max(np.sum(result ** 2, axis=1))
        min_dist_sq = float('inf')
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                diff = result[i] - result[j]
                dsq = np.dot(diff, diff)
                if dsq < min_dist_sq:
                    min_dist_sq = dsq
        if min_dist_sq < max_norm_sq:
            # Fallback: return just 2 points that are far apart
            # Find any two with distance >= 4
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    diff = candidates[i] - candidates[j]
                    if np.dot(diff, diff) >= 4:
                        return np.array([candidates[i], candidates[j]], dtype=np.int64)
            return candidates[:2]
    
    return np.array(result, dtype=np.int64)
