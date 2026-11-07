import numpy as np

def heuristics(prizes, distance_matrix, max_len):
        # Ensure float
        prizes = np.asarray(prizes, dtype=float)
        d = np.asarray(distance_matrix, dtype=float)
        
        n = len(prizes)
        eps = 1e-9
        
        # d[:, 0] is the return cost for each node j
        # d[i, j] is the travel cost from i to j
        # We want H[i, j]
        
        # Strategy: "Regret" / Opportunity Cost
        # Attractiveness of j is high if P_j is high.
        # Penalty is high if the move i->j is long OR if the return j->0 is long.
        # The "Regret" of not picking the best node is related to the difference in prizes.
        # A static approximation of "regret" is normalizing the prize by the global max prize,
        # and penalizing by the total path extension.
        
        # Let's try: H_ij = (P_j / Max_P) / (d_ij + d_j0)
        # This is similar to the 0.92 score child.
        
        # Let's try a different structural family: "Greedy Lookahead"
        # Approximate the value of visiting j as P_j.
        # The cost is d_ij + d_j0.
        # But also consider that visiting j might block other nodes?
        # Hard to do statically.
        
        # Let's try: H_ij = P_j * exp(-beta * (d_ij + d_j0))
        # With max_len=3, typical d is 0.1-0.5.
        # If beta=2, exp(-1) = 0.37.
        
        # Let's try: H_ij = P_j / (d_ij + d_j0 + P_j)
        # This is a smooth ratio.
        
        # Let's go with the "Regret" hint:
        # "inversely proportional to the maximum prize value of any other unvisited node k..."
        # This implies comparing P_j to the best alternative.
        # Static approximation:
        # For each j, let M_j = max(P_k for k != j).
        # Regret of choosing j is roughly M_j - P_j.
        # We want to MINIMIZE regret, so MAXIMIZE (P_j - Regret)?
        # Or simply, attractiveness is P_j / (1 + M_j)?
        # If M_j is large, P_j is likely smaller (unless j is the max).
        # This would penalize nodes when there is a very good alternative available.
        # This encourages picking the "best" node first.
        
        # Let's implement:
        # 1. Compute global max prize for each j (excluding j itself? or just global max?).
        #    If j is the global max, M_j is the second max.
        # 2. H_ij = (P_j / (M_j + eps)) / (d_ij + d_j0 + eps)
        
        # This makes the "best" node (where P_j is close to M_j) have a ratio close to 1.
        # Other nodes have ratio < 1.
        # This biases strongly towards the highest prize nodes.
        
        max_prize = np.max(prizes)
        # For each j, we want the max prize of OTHERS.
        # If we just use global max, the node with global max has ratio 1.
        # Others have P_j / GlobalMax.
        # This is just normalization.
        
        # Let's try: H_ij = P_j / (d_ij + d_j0) * (P_j / max_prize)
        # This squares the effect of high prize.
        
        # Let's try the "Regret" definition literally from the prompt:
        # "attractiveness ... is inversely proportional to the maximum prize value of any other unvisited node k"
        # Attractiveness ~ 1 / MaxPrizeOfOthers?
        # This would make ALL nodes less attractive if there is a high prize node available.
        # That doesn't make sense for selecting the high prize node.
        # Maybe: Attractiveness ~ P_j / MaxPrizeOfOthers?
        # If P_j IS the max, MaxPrizeOfOthers is the second max. Ratio > 1.
        # If P_j is small, MaxPrizeOfOthers is the global max. Ratio < 1.
        # This creates a strong bias for the top prize nodes.
        
        # Implementation:
        # Sort prizes descending.
        # For rank 0 (max prize), MaxOther is rank 1.
        # For rank 1, MaxOther is rank 0.
        # For rank k > 0, MaxOther is rank 0.
        
        # This seems like a good "Regret" metric. It says: "How good is this choice relative to the best alternative I'm giving up?"
        # If I pick the best, I give up the second best. Ratio is high.
        # If I pick a bad one, I give up the best. Ratio is low.
        
        # Let's code this.
        sorted_prizes = np.sort(prizes)[::-1]
        # Create an array max_other_prize for each node j
        # If j is the index of the max prize, max_other is sorted_prizes[1]
        # If j is any other index, max_other is sorted_prizes[0]
        
        max_prize_val = sorted_prizes[0]
        second_max_prize_val = sorted_prizes[1] if n > 1 else 0.0
        
        max_other = np.full(n, max_prize_val)
        # Find indices where prizes are the max
        # There might be multiple maxes.
        # If P_j == max_prize_val, then max_other should be the max of the rest.
        # If there are multiple maxes, the "other" max is still max_prize_val.
        
        # Simple approach:
        # For each j, max_other_j = max(prizes[k] for k != j)
        # This is O(N^2) but N=50, so it's fine.
        
        max_other_prizes = np.zeros(n)
        for j in range(n):
            # Mask out j
            masked_prizes = prizes.copy()
            masked_prizes[j] = 0.0 # Or -inf
            max_other_prizes[j] = np.max(masked_prizes)
        
        # Now, Regret Ratio R_j = P_j / (max_other_prizes[j] + eps)
        # If P_j is the max, R_j = Max / SecondMax (or Max/Max if tie) >= 1.
        # If P_j is low, R_j = Low / Max < 1.
        
        # Heuristic H_ij = R_j * P_j / (d_ij + d_j0 + eps)
        # Or just H_ij = R_j / (d_ij + d_j0 + eps)?
        # The prompt says "attractiveness ... inversely proportional to max prize of other".
        # So Attractiveness ~ 1 / MaxOther?
        # That would mean if there is a huge prize available, ALL moves are less attractive?
        # That seems wrong for ACO. We want to pick the huge prize.
        # Perhaps the prompt implies:
        # Attractiveness of j is high if P_j is high AND MaxOther is low?
        # No, MaxOther is determined by the set of nodes.
        
        # Let's stick to the interpretation:
        # Bias towards nodes that have a high "Regret Ratio" (P_j / MaxOther).
        # And low distance.
        
        eps = 1e-9
        return_cost = d[:, 0]
        total_cost = d + return_cost[None, :]
        
        # R_j shape (n,)
        R = prizes / (max_other_prizes + eps)
        
        # H_ij = R_j / total_cost[i, j]
        # Or H_ij = R_j * P_j / total_cost?
        # Let's use R_j * P_j to emphasize the prize itself.
        
        heuristic = (R[None, :] * prizes[None, :]) / (total_cost + eps)
        
        # Ensure positive
        heuristic = np.maximum(heuristic, 1e-9)
        
        return heuristic
