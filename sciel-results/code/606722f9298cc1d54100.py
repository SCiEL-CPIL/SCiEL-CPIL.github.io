    def dfs(idx, mask, weight):
        nonlocal best_score, best_mask
        
        # If we can't beat best_score with all remaining, return
        if weight + suffix_sum[idx] <= best_score:
            return
            
        if idx == num_count:
            if weight > best_score:
                best_score = weight
                best_mask = mask
            return
        
        node = order[idx]
        
        # Option 1: Include node
        if (mask & conflict_mask[node]) == 0:
            new_weight = weight + weights[node]
            # Check if this branch can possibly beat best_score
            # Max possible gain from remaining nodes after idx is suffix_sum[idx+1]
            if new_weight + suffix_sum[idx+1] > best_score:
                dfs(idx + 1, mask | (1 << node), new_weight)
            
        # Option 2: Exclude node
        # Max possible gain from remaining nodes after idx is suffix_sum[idx+1]
        if weight + suffix_sum[idx+1] > best_score:
            dfs(idx + 1, mask, weight)
