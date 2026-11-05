import numpy as np
import time
import sys

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0 # 5s safety margin before 150s kill

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbor lists for faster access
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    
    # Features
    D = A.sum(axis=1)
    
    # Spectral Centrality: Principal eigenvector
    # Use power iteration for speed
    v = np.ones(N) / N
    for _ in range(100): # Enough for convergence for N=200
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm == 0:
            break
        v = v_new / norm
    Spectral = v
    
    # Normalize features
    def norm(x):
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)
        
    D_n = norm(D)
    Spec_n = norm(Spectral)
    
    # Initial Score: Low Degree, Low Spectral
    # We want to pick nodes that are not central.
    scores = -D_n - 0.5 * Spec_n
    
    # Greedy Decoder
    def greedy_decode(scores):
        indices = np.argsort(-scores, kind='stable')
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                is_picked[neighbors[v]] = True
        return count, indices, is_picked

    # Evaluate a specific set of picked nodes? 
    # We will work with the set directly.
    
    # Initial Greedy
    current_size, _, is_picked = greedy_decode(scores)
    S = set(np.where(is_picked)[0].tolist())
    
    # Local Search: Swap
    # We try to remove a node u from S and replace it with a set of nodes from its neighbors
    # that are not blocked by other nodes in S.
    
    # Helper to get available nodes given a set S
    def get_available(S_set):
        blocked = np.zeros(N, dtype=bool)
        for u in S_set:
            blocked[neighbors[u]] = True
        blocked[list(S_set)] = True
        avail = np.where(~blocked)[0]
        return avail

    # Helper to greedily pick MIS from a subset of nodes
    def greedy_mis_size(avail_nodes, scores_subset):
        if len(avail_nodes) == 0:
            return 0, []
        
        # Sort by score (higher score = better to pick first? 
        # Actually, for MIS, we want to pick nodes that block few others.
        # Low degree is good.
        # Let's use degree within the subgraph? Or just global degree.
        # For simplicity, use global score (which favors low degree).
        # But we need to return the nodes too.
        
        # Re-rank avail_nodes by score
        # scores is the global score array. Higher score = picked earlier in greedy.
        # So we want to pick high score nodes? 
        # Wait, my greedy_decode picks highest score first.
        # My scores are -D - Spec. So low D and low Spec get high scores.
        # So high score = good candidate.
        
        sorted_avail = sorted(avail_nodes, key=lambda x: -scores[x])
        
        picked = []
        blocked_local = np.zeros(N, dtype=bool)
        for v in sorted_avail:
            if not blocked_local[v]:
                picked.append(v)
                blocked_local[v] = True
                blocked_local[neighbors[v]] = True
                if len(picked) > 1: # We only care if we can add > 1 to improve
                    # Actually we can keep going to maximize
                    pass
        return len(picked), picked

    improved = True
    while improved and time.time() < DEADLINE - 5.0:
        improved = False
        # Try to swap each node in S
        for u in list(S):
            if time.time() > DEADLINE - 5.0:
                break
            
            S_temp = S - {u}
            avail = get_available(S_temp)
            
            # Filter avail to only neighbors of u? 
            # No, any node not blocked by S_temp is available.
            # But to "swap" u, we should ideally replace it with nodes that were previously blocked by u.
            # However, removing u might unblock nodes that were blocked by u AND other nodes? No, if they were blocked by others, they are still blocked.
            # So avail contains nodes that are NOT blocked by S_temp.
            # This includes neighbors of u that are not neighbors of any other node in S_temp.
            
            if len(avail) == 0:
                continue
                
            # Try to find an independent set in avail
            # We want size > 1 to improve total size (remove 1, add k, net +k-1)
            k, picked_nodes = greedy_mis_size(avail, scores)
            
            if k > 1:
                # Improvement!
                S = S_temp | set(picked_nodes)
                current_size += (k - 1)
                improved = True
            elif k == 1:
                # Tie in size. We can swap to explore different configurations.
                # This helps escape local minima.
                # Only do this occasionally or if it leads to better "potential"?
                # Let's do it to mix the set.
                # But careful: it might lead to a worse configuration later?
                # Since we are maximizing size, a tie is neutral.
                # Let's accept the swap to diversify.
                S = S_temp | set(picked_nodes)
                # current_size remains same
                # Mark as improved to continue loop? 
                # If we mark improved, we might loop forever on ties.
                # Let's only mark improved if size increased.
                # But we want to explore. 
                # Let's add a small random tie-breaker or just accept and continue.
                # To avoid infinite loops on ties, we can limit tie-swaps.
                # For now, let's just do size-increasing swaps.
                pass

    # Final Score Calculation for Return
    # The function must return "one score per vertex".
    # The evaluator uses these scores to run ITS OWN greedy decoder.
    # So we must return scores that, when used in a greedy decoder, yield a large IS.
    # Our local search found a good SET S.
    # How do we convert a good SET S into SCORES that reproduce S (or something similar) via greedy?
    # We can set score[v] = 0 if v in S, and score[v] = -1 (or very low) if v not in S.
    # But greedy picks highest score first.
    # If we set S nodes to 1 and others to 0, greedy will pick all S nodes first (since they are not adjacent to each other).
    # Then it will pick others? No, if it picks all S nodes, it blocks their neighbors.
    # The remaining nodes are those not in S and not neighbors of S.
    # Greedy will then pick from the remaining.
    # This might add more nodes to S, which is fine (increases size).
    # Or it might pick nodes that are adjacent to S? No, they are blocked.
    # So, returning scores that prioritize S is a good strategy.
    
    # Let's construct scores:
    # High score for nodes in S.
    # Medium score for nodes that are "good" candidates (low degree) but not in S.
    # Low score for nodes that are bad.
    
    final_scores = np.zeros(N)
    for v in S:
        final_scores[v] = 10.0 # High priority
        
    # For others, use the original heuristic
    # This ensures that if the greedy decoder picks nodes from S first, it effectively reconstructs our good set.
    # If it picks other nodes first, it might interfere, but 10.0 is high enough to ensure S is picked first.
    
    # Wait, if we pick all nodes in S, we block their neighbors.
    # The greedy decoder will then look at the remaining unblocked nodes.
    # It will pick the best ones among them.
    # This is effectively: IS = S U Greedy(IS of G - N[S]).
    # This is likely to be large.
    
    # Let's refine: 
    # If we just return scores based on S, the greedy decoder will pick S.
    # Then it picks the best from the rest.
    # This is equivalent to our Local Search result plus a greedy fill of the remainder.
    # This is a valid and strong strategy.
    
    # Apply original scores to non-S nodes to guide the "fill"
    for v in range(N):
        if v not in S:
            final_scores[v] = scores[v]
            
    return final_scores
