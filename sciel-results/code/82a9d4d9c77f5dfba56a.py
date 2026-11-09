import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_K4 = 1820

# Pre-compute edge indices
EDGE_IDX = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        EDGE_IDX[(i, j)] = idx
        idx += 1

# Pre-compute K4s and their edge indices
K4_EDGES = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    edges = [
        EDGE_IDX[(a, b)], EDGE_IDX[(a, c)], EDGE_IDX[(a, d)],
        EDGE_IDX[(b, c)], EDGE_IDX[(b, d)], EDGE_IDX[(c, d)]
    ]
    K4_EDGES.append(tuple(edges))

# Pre-compute for each edge, which K4s it belongs to
EDGE_TO_K4 = [[] for _ in range(NUM_EDGES)]
for k_idx, edges in enumerate(K4_EDGES):
    for e in edges:
        EDGE_TO_K4[e].append(k_idx)

# Convert to numpy for faster access where possible, but lists are fine for small updates
K4_EDGES_LIST = K4_EDGES

def _compute_initial_state(colors, K):
    """
    Initialize the state of K4 constraints.
    Returns:
    k4_color_counts: list of dicts or arrays tracking color frequency in each K4
    k4_distinct: numpy array of distinct color counts for each K4
    total_violations: int
    """
    # For each K4, we track the count of each color (0..K-1)
    # Using a 2D array: [NUM_K4, K]
    k4_color_counts = np.zeros((NUM_K4, K), dtype=np.int32)
    
    for k_idx, edges in enumerate(K4_EDGES_LIST):
        for e in edges:
            c = int(colors[e])
            k4_color_counts[k_idx, c] += 1
        
    # Compute distinct counts
    k4_distinct = np.sum(k4_color_counts > 0, axis=1)
    total_violations = int(np.sum(k4_distinct < 5))
    
    return k4_color_counts, k4_distinct, total_violations

def _try_csp_search(K, deadline, max_restarts=50):
    """
    Perform a CSP-style local search with K colors.
    """
    best_solution = None
    best_violations = float('inf')
    
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
            
        # Initialize random assignment
        colors = np.random.randint(0, K, size=NUM_EDGES, dtype=np.int32)
        k4_color_counts, k4_distinct, total_violations = _compute_initial_state(colors, K)
        
        if total_violations == 0:
            return colors
        
        if total_violations < best_violations:
            best_violations = total_violations
            best_solution = colors.copy()
            best_state = (k4_color_counts.copy(), k4_distinct.copy())
            
        # Local Search Loop
        # We want to resolve violations.
        # Strategy: Pick a violated K4, pick an edge in it, try to change its color to reduce violations.
        
        no_improve_count = 0
        max_no_improve = 10000
        
        while time.time() < deadline - 2:
            if total_violations == 0:
                return colors
            
            # Pick a violated K4
            violated_k4s = np.where(k4_distinct < 5)[0]
            if len(violated_k4s) == 0:
                break
                
            k_idx = int(violated_k4s[random.randint(0, len(violated_k4s) - 1)])
            edges = K4_EDGES_LIST[k_idx]
            
            # Pick an edge in this K4 to modify
            # Heuristic: Pick the edge whose color change is most likely to help.
            # For simplicity, pick randomly, but we could prefer edges with 'unique' colors in this K4.
            e_idx = int(edges[random.randint(0, 5)])
            old_c = int(colors[e_idx])
            
            # Try all K colors
            best_delta = float('inf')
            best_new_c = old_c
            best_new_violations = total_violations
            
            # Pre-fetch affected K4s for this edge
            affected_k4s = EDGE_TO_K4[e_idx]
            
            for new_c in range(K):
                if new_c == old_c:
                    continue
                
                # Simulate the change
                # We need to calculate the change in distinct counts for all affected K4s
                delta_violations = 0
                
                for ak_idx in affected_k4s:
                    # Get current state
                    old_distinct = k4_distinct[ak_idx]
                    
                    # Update counts in a temporary way
                    # old color count decreases by 1
                    # new color count increases by 1
                    
                    # If old color count was 1, it becomes 0 -> distinct decreases by 1
                    # If new color count was 0, it becomes 1 -> distinct increases by 1
                    
                    # Note: It's possible old_c == new_c, but we skipped that.
                    # It's possible the edge is not in this K4? No, affected_k4s are derived from e_idx.
                    
                    old_count_old_c = k4_color_counts[ak_idx, old_c]
                    old_count_new_c = k4_color_counts[ak_idx, new_c]
                    
                    new_distinct = old_distinct
                    if old_count_old_c == 1:
                        new_distinct -= 1
                    if old_count_new_c == 0:
                        new_distinct += 1
                        
                    old_status = old_distinct < 5
                    new_status = new_distinct < 5
                    
                    if old_status and not new_status:
                        delta_violations -= 1
                    elif not old_status and new_status:
                        delta_violations += 1
                        # Early exit: if we are already worse, don't bother checking other K4s for this candidate
                        # Actually, we should check all to get accurate delta, but if it's already > best_delta, we can break
                        if delta_violations > best_delta:
                            break
                            
                if delta_violations < best_delta:
                    best_delta = delta_violations
                    best_new_c = new_c
                    best_new_violations = total_violations + delta_violations
                    if best_delta < 0:
                        # Found an improvement
                        break
                        # We could continue to find the best improvement, but any improvement is good
                        # To be faster, we break on first improvement found? 
                        # No, let's find the best one among K options (K is small, 13-15)
                        
            # Apply the best move found
            if best_new_c != old_c:
                # Update state
                colors[e_idx] = best_new_c
                
                for ak_idx in affected_k4s:
                    k4_color_counts[ak_idx, old_c] -= 1
                    k4_color_counts[ak_idx, best_new_c] += 1
                    k4_distinct[ak_idx] = np.sum(k4_color_counts[ak_idx] > 0)
                    
                total_violations = best_new_violations
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_solution = colors.copy()
                    best_state = (k4_color_counts.copy(), k4_distinct.copy())
                
                if total_violations == 0:
                    return colors
                
                if best_delta < 0:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # No move improved the situation
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                break
                
    # If we didn't find a solution with 0 violations, return the best one found (which might still have violations)
    # But the problem requires a VALID solution. If we return an invalid one, score is 0.
    # So we must only return if valid.
    if best_violations == 0:
        return best_solution
    return None

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Try to find a solution with 13 colors first
    # Then 14, then 15.
    
    # We will allocate time to each.
    # 13 colors is the hardest.
    
    # Random seed for reproducibility in search path, though result is deterministic per seed
    random.seed(42)
    np.random.seed(42)
    
    # Attempt 13 colors
    sol = _try_csp_search(13, deadline, max_restarts=20)
    if sol is not None:
        return [int(c) for c in sol]
        
    # Attempt 14 colors
    # We have less time now
    deadline_14 = time.time() + 60
    sol = _try_csp_search(14, deadline_14, max_restarts=20)
    if sol is not None:
        return [int(c) for c in sol]
        
    # Attempt 15 colors
    # Should be easier
    deadline_15 = time.time() + 20
    sol = _try_csp_search(15, deadline_15, max_restarts=50)
    if sol is not None:
        return [int(c) for c in sol]
        
    # Fallback: If CSP failed, try a simple random construction and hope for the best?
    # Or return a known valid 15-coloring?
    # A simple (i+j) % 17 uses 17 colors. We need <= 15.
    # Let's try a structured 15-coloring.
    # If all else fails, return a random 15-coloring and hope it's valid (unlikely).
    # Better: Return a hardcoded valid 15-coloring if we have one?
    # We don't have one hardcoded.
    
    # Last resort: Random 15-coloring. If it's invalid, score 0.
    # But CSP should find 15 easily.
    
    # If we get here, we failed. Return a dummy to avoid crash.
    # The evaluator will score 0.
    return [0] * NUM_EDGES

# EVOLVE-BLOCK-END
