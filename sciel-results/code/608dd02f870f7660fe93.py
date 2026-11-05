# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_set_size(order, A, N):
    """
    Computes the size of the independent set produced by the greedy decoder
    given the vertex order (highest priority first).
    """
    selected = np.zeros(N, dtype=bool)
    count = 0
    for v in order:
        v_idx = int(v)
        if not selected[v_idx]:
            if not selected[A[v_idx]].any():
                selected[v_idx] = True
                count += 1
    return count

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy:
    1. Initial heuristic: Weighted degree (inverse of degree + neighbor degree sum).
    2. Local Search: Iteratively try to swap adjacent vertices in the priority order 
       to increase the independent set size.
    3. Return scores corresponding to the final optimized order.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    A = np.asarray(adjacency, dtype=bool)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # 1. Initial Heuristic
    # We want vertices that are "safe" to pick.
    # Low degree is generally good.
    # Low sum of neighbor degrees is good (neighbors are less likely to be picked before us? 
    # Actually, if neighbors have low degree, they are less likely to be high priority, 
    # so they are less likely to block us. Wait, the decoder picks high priority first.
    # If v has high priority, it is picked first. Its neighbors are blocked.
    # We want the set of picked vertices to be large.
    # Picking a vertex with many neighbors is bad if those neighbors could have picked many other vertices.
    # But since they are blocked, we lose the chance to pick them.
    # However, the neighbors themselves might block others.
    
    # Standard heuristic: Minimize degree.
    deg = A.sum(axis=1)
    
    # Let's use a score that favors low degree and low "clustering coefficient" or similar.
    # Simple: Score = -deg
    # Better: Score = - (deg + sum(deg(neighbors)))
    sum_nei_deg = A @ deg
    
    # Initial score: lower is better (we want to pick low score first? No, higher score picked first).
    # So we want high score for "good" vertices.
    # Good vertex: Low deg, Low sum_nei_deg.
    init_score = -(deg.astype(float) + sum_nei_deg.astype(float))
    
    # Get initial order (descending score)
    order = np.argsort(-init_score)
    
    # Evaluate initial size
    current_size = _greedy_set_size(order, A, N)
    best_size = current_size
    best_order = order.copy()
    
    # 2. Local Search
    # Try swapping adjacent elements.
    # To speed up, we can check if a swap is worth evaluating.
    # A swap of u, v (u before v) changes the set only if:
    # 1. u is picked in original order.
    # 2. v is not picked in original order (blocked by u or someone else).
    # 3. In new order, v is picked (u is now after v, so u doesn't block v).
    # 4. In new order, u is not picked (blocked by v or someone else).
    # This is complex to check incrementally without re-evaluating or maintaining state.
    # Given N=200, re-evaluation is O(N^2) ~ 40k ops. 
    # We can do thousands of swaps in 1 second.
    # Let's do a simple hill climbing.
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(N - 1):
            if (time.time() - start_time) > deadline:
                break
            
            # Try swapping i and i+1
            u = best_order[i]
            v = best_order[i+1]
            
            # Quick check: If u and v are not adjacent, swapping them doesn't change 
            # the blocking relationship between them. 
            # However, it might change their relationship with other vertices?
            # No, the relative order of u and v with respect to others (before i or after i+1) remains the same.
            # The only change is the order of u and v.
            # If A[u,v] is False:
            #   - If u was picked, v's status depends on u only if u blocks v. If not adjacent, u doesn't block v.
            #   - So if not adjacent, the set of picked vertices remains exactly the same?
            #   - Let's verify.
            #   - Original: ... u, v ...
            #   - New: ... v, u ...
            #   - If not adjacent, u's pickability depends on neighbors before i. v's pickability depends on neighbors before i+1 (which includes u in original, not in new).
            #   - Wait. In original, when considering v, u has already been considered.
            #   - If u was picked, and A[u,v]=0, u doesn't block v.
            #   - In new, when considering u, v has been considered. If v was picked, and A[v,u]=0, v doesn't block u.
            #   - So the status of u and v individually might change if they were blocked by each other? No, they don't block each other.
            #   - Do they block other vertices?
            #   - The set of picked vertices is determined sequentially.
            #   - If u and v are not adjacent, swapping them does not change the set of picked vertices.
            #   - Proof:
            #     Let S_before be the set of picked vertices from indices < i.
            #     Case 1: u and v are not adjacent.
            #     Original:
            #       u is picked if not blocked by S_before.
            #       v is picked if not blocked by S_before U {u if picked}. Since A[u,v]=0, u doesn't block v. So v picked if not blocked by S_before.
            #     New:
            #       v is picked if not blocked by S_before.
            #       u is picked if not blocked by S_before U {v if picked}. Since A[v,u]=0, v doesn't block u. So u picked if not blocked by S_before.
            #     The set of picked vertices is {u if not blocked by S_before} U {v if not blocked by S_before} in both cases.
            #     The subsequent vertices see the same set of picked vertices (the union is the same).
            #     Therefore, if A[u,v] == False, the swap does not change the set size.
            
            if not A[u, v]:
                continue
            
            # If adjacent, the swap might change the set.
            # Create new order
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_size = _greedy_set_size(new_order, A, N)
            
            if new_size > best_size:
                best_order = new_order
                best_size = new_size
                improved = True
                # Restart scan from beginning? Or continue?
                # Restarting is safer for convergence but slower.
                # Given small N, let's restart to ensure we don't miss local maxima.
                # Actually, just breaking the inner loop and restarting the while loop is fine.
                break
    
    # 3. Map order back to scores
    # The decoder picks in descending score order.
    # We need to return scores such that argsort(-scores) == best_order.
    # Assign scores based on rank.
    ranks = np.zeros(N)
    for i, v in enumerate(best_order):
        ranks[v] = N - i
        
    return ranks
# EVOLVE-BLOCK-END
