import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 256
    
    A = set()
    remaining = set(range(1, N + 1))
    
    # Precompute 4-AP conflicts: for each x, which pairs in A would make a 4-AP with x
    # 4-AP: a, a+d, a+2d, a+3d. x participates if it's one of these four positions.
    
    def is_valid_add(x, A):
        # Check 4-AP
        # x is 1st: need x+d, x+2d, x+3d all in A
        for d in range(1, (N - x) // 3 + 1):
            if x + d in A and x + 2 * d in A and x + 3 * d in A:
                return False
        # x is 2nd: need x-d, x+d, x+2d all in A
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if x - d in A and x + d in A and x + 2 * d in A:
                return False
        # x is 3rd: need x-2d, x-d, x+d all in A
        for d in range(1, min(x // 2, N - x) + 1):
            if x - 2 * d in A and x - d in A and x + d in A:
                return False
        # x is 4th: need x-3d, x-2d, x-d all in A
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3 * d in A and x - 2 * d in A and x - d in A:
                return False
        return True

    def count_3ap_gain(x, A):
        gain = 0
        # x is 1st: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if x + d in A and x + 2 * d in A:
                gain += 1
        # x is 2nd: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in A and x + d in A:
                gain += 1
        # x is 3rd: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if x - 2 * d in A and x - d in A:
                gain += 1
        return gain

    def count_total_3ap(A):
        count = 0
        for b in A:
            for d in range(1, min(b - 1, N - b) + 1):
                if b - d in A and b + d in A:
                    count += 1
        return count

    def compute_loss(x, A):
        loss = 0
        for d in range(1, (N - x) // 2 + 1):
            if x + d in A and x + 2 * d in A:
                loss += 1
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in A and x + d in A:
                loss += 1
        for d in range(1, x // 2 + 1):
            if x - 2 * d in A and x - d in A:
                loss += 1
        return loss

    # Improved greedy: use a weighted score that balances 3-AP gain with "future potential"
    # Future potential = number of elements in remaining that could still be added without conflict
    def greedy_score(x, A, remaining):
        gain = count_3ap_gain(x, A)
        if gain < 0:
            return -1
        
        # Count how many elements in remaining would become invalid if we add x
        # This is a proxy for "structural cost"
        conflict_count = 0
        for y in remaining:
            if y == x:
                continue
            # Check if x and y together would create a 4-AP with some z in A
            # Simplified: just check if adding x makes y invalid
            # This is expensive, so only do it for a sample
            pass
        
        # Use a simpler heuristic: prefer elements that complete more 3-APs
        # and are "central" (more neighbors in remaining)
        centrality = 0
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in remaining:
                centrality += 1
            if x + d in remaining:
                centrality += 1
        
        # Weighted score: gain * 2 + centrality * 0.1
        return gain * 2 + centrality * 0.1

    # Greedy Phase
    while remaining and time.time() < deadline - 40:
        best_x = -1
        best_score = -1
        for x in remaining:
            if is_valid_add(x, A):
                score = greedy_score(x, A, remaining)
                if score > best_score:
                    best_score = score
                    best_x = x
        if best_x == -1:
            break
        A.add(best_x)
        remaining.remove(best_x)
    
    current_score = count_total_3ap(A)
    
    # Local Search Phase: 1-swap moves with improved evaluation
    improved = True
    max_passes = 50
    pass_count = 0
    
    while improved and pass_count < max_passes and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Compute loss for each element in A
        A_losses = {}
        for a in A:
            A_losses[a] = compute_loss(a, A)
        
        # Compute gain for each element in remaining (if valid to add to A)
        R_gains = {}
        for b in remaining:
            if is_valid_add(b, A):
                R_gains[b] = count_3ap_gain(b, A)
            else:
                R_gains[b] = -1
        
        # Sort A by loss ascending (remove low-value elements first)
        sorted_A = sorted(A_losses.keys(), key=lambda x: A_losses[x])
        # Sort R by gain descending (add high-value elements first)
        sorted_R = sorted(R_gains.keys(), key=lambda x: R_gains[x], reverse=True)
        
        # Try swaps: for each a in sorted_A (up to 15), for each b in sorted_R (up to 15)
        max_a_tries = 15
        max_b_tries = 15
        found_improvement = False
        
        for a in sorted_A[:max_a_tries]:
            if found_improvement:
                break
            if time.time() > deadline:
                break
                
            for b in sorted_R[:max_b_tries]:
                if time.time() > deadline:
                    break
                    
                if R_gains[b] < 0:
                    continue
                    
                # Check if b is valid to add to A - {a}
                A.remove(a)
                
                valid_b = is_valid_add(b, A)
                
                if valid_b:
                    gain_b = count_3ap_gain(b, A)
                    new_score = current_score - A_losses[a] + gain_b
                    
                    if new_score > current_score:
                        A.add(b)
                        remaining.remove(b)
                        current_score = new_score
                        improved = True
                        found_improvement = True
                        break
                else:
                    A.add(a)
            
            if a not in A:
                A.add(a)
        
        # Try 2-swap moves: remove two elements, add two new ones
        if not found_improvement and time.time() < deadline - 5:
            # Sample pairs from low-loss A and high-gain R
            sample_a = sorted_A[:5]
            sample_b = sorted_R[:5]
            
            for i, a1 in enumerate(sample_a):
                if time.time() > deadline:
                    break
                for a2 in sample_a[i+1:]:
                    if time.time() > deadline:
                        break
                    for b1 in sample_b:
                        if time.time() > deadline:
                            break
                        for b2 in sample_b:
                            if b2 <= b1:
                                continue
                            if R_gains[b1] < 0 or R_gains[b2] < 0:
                                continue
                            
                            A.remove(a1)
                            A.remove(a2)
                            
                            valid_b1 = is_valid_add(b1, A)
                            if valid_b1:
                                A.add(b1)
                                valid_b2 = is_valid_add(b2, A)
                                A.remove(b1)
                            else:
                                valid_b2 = False
                            
                            if valid_b2:
                                gain_b1 = count_3ap_gain(b1, A)
                                A.add(b1)
                                gain_b2 = count_3ap_gain(b2, A)
                                A.remove(b1)
                                
                                new_score = current_score - A_losses[a1] - A_losses[a2] + gain_b1 + gain_b2
                                
                                if new_score > current_score:
                                    A.add(b1)
                                    A.add(b2)
                                    remaining.remove(b1)
                                    remaining.remove(b2)
                                    current_score = new_score
                                    improved = True
                                    found_improvement = True
                                    break
                            else:
                                pass
                            
                            A.add(a1)
                            A.add(a2)
                        
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            
            # Restore if we broke out without accepting
            if not found_improvement:
                # Ensure A is restored (it should be, since we add back in the loop)
                pass
    
    return sorted(A)
