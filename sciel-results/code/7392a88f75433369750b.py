import math
import time

def construct():
        N = 511
        weights = [0.0] * (N + 1)
        for i in range(1, N + 1):
            weights[i] = 1.0 / (512 - i)
        
        # Precompute conflicts
        conflicts = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                if math.gcd(i, j) > 1:
                    conflicts[i] |= (1 << j)
                    conflicts[j] |= (1 << i)
        
        # Greedy Init
        in_set = 0
        selected = []
        for a in range(N, 0, -1):
            if (conflicts[a] & in_set) == 0:
                in_set |= (1 << a)
                selected.append(a)
        
        score = sum(weights[a] for a in selected)
        
        start_time = time.time()
        deadline = start_time + 140.0
        
        # Local Search
        # To avoid infinite loops or slow convergence, we can use a simple hill climbing.
        # We try to find any improving move.
        
        # Optimization: Precompute list of indices for faster iteration?
        # Using bitsets is fast.
        
        while time.time() < deadline:
            improved = False
            
            # 1-Swap
            for u in selected:
                in_set_minus_u = in_set ^ (1 << u)
                # Find best x
                best_x = -1
                best_gain = 0.0
                # Iterate through all x not in set
                # To make this faster, we can just check all x
                for x in range(1, N + 1):
                    if not (in_set & (1 << x)):
                        if (conflicts[x] & in_set_minus_u) == 0:
                            gain = weights[x] - weights[u]
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                
                if best_x != -1 and best_gain > 1e-9:
                    # Apply swap
                    in_set = in_set_minus_u | (1 << best_x)
                    selected.remove(u)
                    selected.append(best_x)
                    score += best_gain
                    improved = True
                    break # Restart search from beginning after change
            
            if improved:
                continue
                
            # 2-Swap
            # This is more expensive. Only do if 1-swap fails.
            # To speed up, we can limit the search or use random sampling if N is large.
            # But N=511 is small enough for a structured search if optimized.
            
            # Try pairs of u, v in selected
            # We can optimize by only considering u, v that are "weak" (small weight)
            
            found_2swap = False
            # Iterate u
            for i_idx, u in enumerate(selected):
                in_set_minus_u = in_set ^ (1 << u)
                # Iterate v > u
                for j_idx in range(i_idx + 1, len(selected)):
                    v = selected[j_idx]
                    if u == v: continue
                    in_set_minus_uv = in_set_minus_u ^ (1 << v)
                    
                    # Find valid x
                    # x must not be in set, and coprime to set \ {u,v}
                    valid_xs = []
                    for x in range(1, N + 1):
                        if not (in_set & (1 << x)):
                            if (conflicts[x] & in_set_minus_uv) == 0:
                                valid_xs.append(x)
                    
                    if len(valid_xs) < 2:
                        continue
                        
                    # Find pair x, y in valid_xs that are coprime to each other
                    # and maximize weight[x] + weight[y]
                    best_pair = None
                    best_pair_weight = 0.0
                    current_weight_uv = weights[u] + weights[v]
                    
                    # Sort valid_xs by weight desc to find good pairs early?
                    valid_xs.sort(key=lambda z: weights[z], reverse=True)
                    
                    for k_idx, x in enumerate(valid_xs):
                        # If even the best possible sum is not better, break
                        # Max possible sum is weights[x] + max(weights[valid_xs[k_idx+1:]])
                        # Approximation:
                        if weights[x] + weights[valid_xs[k_idx+1]] <= current_weight_uv + 1e-9:
                            break
                            
                        for l_idx in range(k_idx + 1, len(valid_xs)):
                            y = valid_xs[l_idx]
                            if (conflicts[x] & (1 << y)) == 0:
                                pair_weight = weights[x] + weights[y]
                                if pair_weight > current_weight_uv + 1e-9:
                                    if best_pair is None or pair_weight > best_pair_weight:
                                        best_pair = (x, y)
                                        best_pair_weight = pair_weight
                                # Since sorted, if this y doesn't work, maybe a smaller y does?
                                # No, we want max weight. If we found a valid pair, we can keep looking for better?
                                # Or just take the first valid one since sorted?
                                # The first valid one is the highest weight pair?
                                # Not necessarily, x is fixed, y is highest coprime.
                                # But a different x might yield better sum.
                                pass
                    
                    if best_pair:
                        x, y = best_pair
                        # Apply
                        in_set = in_set_minus_uv | (1 << x) | (1 << y)
                        selected.remove(u)
                        selected.remove(v)
                        selected.append(x)
                        selected.append(y)
                        score += best_pair_weight - current_weight_uv
                        improved = True
                        found_2swap = True
                        break
                
                if found_2swap:
                    break
            
            if not improved:
                # No improving move found.
                # We can try random restarts or accept current.
                # Given the structure, greedy is likely optimal or near-optimal.
                # Break to save time.
                break

        return selected
