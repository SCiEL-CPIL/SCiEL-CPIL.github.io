import time

def construct():
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248
    # Max denominator: 248
    best_solution = [8, 31, 248]
    best_max = 248
    
    target_num = 5
    target_den = 31
    
    deadline = time.time() + 50.0

    # Helper to check if a list of denominators sums to target
    # Not needed for the search logic below, which is constructive.

    # Search for k-term solutions with all denominators < limit_max
    # We want to minimize the largest denominator, so we search for max_den < current_best_max
    
    def search_k_terms(k, limit_max):
        """
        Finds a solution of exactly k terms where all denominators are < limit_max.
        Returns the solution with the smallest maximum denominator if found.
        """
        best_local = None
        best_local_max = limit_max
        
        # Use recursion/DFS
        # rem_num / rem_den is the remaining fraction to represent
        # min_d is the minimum denominator for the next term
        # terms is the current list of denominators
        
        def dfs(rem_num, rem_den, min_d, current, terms_left):
            nonlocal best_local, best_local_max
            
            if time.time() > deadline:
                return False
                
            if rem_num == 0:
                if terms_left == 0 and current:
                    m = max(current)
                    if m < best_local_max:
                        best_local_max = m
                        best_local = current[:]
                        return True
                return False
                
            if terms_left == 0:
                return False
                
            # If we have 1 term left, it must be exactly rem
            if terms_left == 1:
                if rem_num == 1:
                    d = rem_den
                    if min_d <= d < limit_max:
                        current.append(d)
                        # Check if this improves our best
                        m = max(current)
                        if m < best_local_max:
                            best_local_max = m
                            best_local = current[:]
                        current.pop()
                        return True
                return False

            # Pruning:
            # The sum of the remaining 'terms_left' reciprocals must be able to form rem_num/rem_den.
            # Max sum: 1/min_d + 1/(min_d+1) + ...
            # Min sum: 1/(limit_max-1) + ...
            
            # Simple bound: 1/min_d * terms_left >= rem_num/rem_den
            # And 1/limit_max * terms_left <= rem_num/rem_den (approx)
            
            # Better bound for d_i:
            # 1/d_i <= rem_num/rem_den  => d_i >= rem_den/rem_num
            # Also d_i >= min_d
            
            d_start = max(min_d, (rem_den + rem_num - 1) // rem_num)
            
            # d_i must be such that rem_num/rem_den - 1/d_i > 0
            # and rem_num/rem_den - 1/d_i can be represented by terms_left-1 terms < limit_max.
            
            # Upper bound for d_i:
            # The remaining sum rem_num/rem_den - 1/d_i must be <= (terms_left-1) * (1/min_d_next) ? No.
            # The remaining sum must be >= (terms_left-1) * (1/(limit_max-1)).
            # And <= (terms_left-1) * (1/(d_i+1)) (using smallest possible next denominators)
            
            # Let's just iterate d from d_start upwards.
            # We can stop when 1/d is too small to help even with the smallest remaining terms.
            # Actually, if 1/d is small, d is large.
            # Constraint: rem_num/rem_den - 1/d >= (terms_left-1) / (limit_max-1)
            # 1/d <= rem_num/rem_den - (terms_left-1)/(limit_max-1)
            # d >= 1 / (rem_num/rem_den - (terms_left-1)/(limit_max-1))
            
            # And constraint: rem_num/rem_den - 1/d <= (terms_left-1) * (1/(d+1)) is not quite right for min sum of next terms.
            # The max sum of next terms is 1/(d+1) + 1/(d+2) + ...
            # So we need rem_num/rem_den - 1/d <= sum_{j=1}^{terms_left-1} 1/(d+j)
            
            # To keep it simple and fast for small limit_max (247):
            # We iterate d. The range is small.
            
            # Optimization: 
            # d must be < limit_max.
            # Also, rem_num/rem_den - 1/d must be > 0.
            
            # Calculate upper bound for d to prune the loop end.
            # If terms_left == 2, we can solve directly.
            if terms_left == 2:
                # 1/d + 1/e = rem
                # e = d * rem_den / (rem_num * d - rem_den)
                # Let x = rem_num * d - rem_den. e = d * rem_den / x.
                # x must divide d * rem_den.
                # Also e > d and e < limit_max.
                
                # d starts from d_start
                # d < limit_max
                # e = d * rem_den / x. 
                # x > 0 => rem_num * d > rem_den => d > rem_den/rem_num.
                
                # Also e > d => d * rem_den / x > d => rem_den > x => rem_den > rem_num * d - rem_den => 2*rem_den > rem_num * d => d < 2*rem_den/rem_num.
                
                d_end = min(limit_max, (2 * rem_den) // rem_num)
                
                for d in range(d_start, d_end + 1):
                    x = rem_num * d - rem_den
                    if x <= 0:
                        continue
                    if (d * rem_den) % x != 0:
                        continue
                    e = (d * rem_den) // x
                    if e <= d:
                        continue
                    if e >= limit_max:
                        continue
                    # Found a solution
                    current.append(d)
                    current.append(e)
                    m = max(current)
                    if m < best_local_max:
                        best_local_max = m
                        best_local = current[:]
                    current.pop()
                    current.pop()
                    
                return best_local is not None

            # For terms_left > 2, iterate d
            # d_start to limit_max - 1
            # Prune upper end: 
            # If 1/d is too small, the remaining sum is too large to be formed by terms_left-1 terms? No.
            # If d is large, 1/d is small. rem - 1/d is large.
            # The max sum of terms_left-1 terms with denominators > d is bounded by sum 1/(d+i).
            # We need rem - 1/d <= sum_{i=1}^{terms_left-1} 1/(d+i)
            
            # Since limit_max is small (<= 247), we can just iterate up to limit_max.
            # The number of iterations is small.
            
            for d in range(d_start, limit_max):
                if time.time() > deadline:
                    return False
                    
                # Check if rem - 1/d is positive
                # rem_num/rem_den - 1/d = (rem_num*d - rem_den) / (rem_den*d)
                num_rem = rem_num * d - rem_den
                if num_rem <= 0:
                    continue
                
                den_rem = rem_den * d
                
                # Simplify fraction? Not strictly necessary for integer check in next steps, but good for bounds.
                # We pass num_rem/den_rem to next DFS.
                
                # Pruning: Can num_rem/den_rem be represented by terms_left-1 terms > d?
                # Min sum of terms_left-1 terms > d is (terms_left-1)/(limit_max-1)
                # If num_rem/den_rem < (terms_left-1)/(limit_max-1), return False for this d? No, just skip this d?
                # Actually, if the required sum is too small, we can't form it with distinct denominators > d and < limit_max?
                # Yes, if min_possible_sum > target, it's impossible.
                
                # Max sum of terms_left-1 terms > d is sum 1/(d+i).
                # If num_rem/den_rem > max_possible_sum, skip.
                
                # For small terms_left and limit_max, these checks are cheap.
                
                # Heuristic: 
                # 1/d < rem  (already ensured by num_rem > 0)
                # Also 1/d should not be too small such that the rest is "too big"?
                # The rest is num_rem/den_rem.
                # The max sum of next terms is approx (terms_left-1)/d.
                # So num_rem/den_rem <= (terms_left-1)/d  => num_rem*d <= (terms_left-1)*den_rem
                # num_rem*d <= (terms_left-1)*rem_den*d
                # rem_num*d - rem_den <= (terms_left-1)*rem_den
                # rem_num*d <= terms_left*rem_den
                # d <= terms_left * rem_den / rem_num
                
                d_bound = (terms_left * rem_den) // rem_num
                if d_bound < d_start:
                    continue
                # We are already in a loop, but we can break if d exceeds this bound
                # Actually, the loop goes to limit_max. We can just check if d > d_bound, then break.
                if d > d_bound:
                    break

                # Additional pruning: 
                # The minimum sum of the next terms is (terms_left-1)/(limit_max-1).
                # If num_rem/den_rem < (terms_left-1)/(limit_max-1), then it's too small to form with that many large terms.
                # num_rem * (limit_max-1) < (terms_left-1) * den_rem
                if num_rem * (limit_max - 1) < (terms_left - 1) * den_rem:
                    continue

                # Recursive step
                current.append(d)
                if dfs(num_rem, den_rem, d + 1, current, terms_left - 1):
                    current.pop()
                    return True # Found a solution, can stop searching for this branch? 
                               # We want the GLOBAL best (min max). 
                               # The DFS finds solutions. We update best_local.
                               # We should probably continue searching to find a better one?
                               # Or does the structure of the search (increasing d) ensure we find smaller maxes first?
                               # No. Smaller maxes might be found later.
                               # However, if we find a solution with max M, any subsequent solution must have max < M to be better.
                               # We can prune branches where the current max is already >= best_local_max.
                current.pop()
                
                # Prune if current max is already >= best_local_max?
                # We don't know the final max yet. But if d >= best_local_max, and all future denominators are > d, then max will be > d >= best_local_max.
                if d >= best_local_max:
                    break

            return best_local is not None

        dfs(target_num, target_den, 1, [], k)
        
        if best_local:
            return best_local
        return None

    # Search 2 terms
    sol = search_k_terms(2, best_max)
    if sol:
        return sol

    # Search 3 terms
    sol = search_k_terms(3, best_max)
    if sol:
        return sol

    # Search 4 terms
    sol = search_k_terms(4, best_max)
    if sol:
        return sol

    # Search 5 terms (unlikely to be needed if 3 or 4 works, but just in case)
    # The space grows, so we add a time check in the DFS.
    sol = search_k_terms(5, best_max)
    if sol:
        return sol

    # If no better solution found, return the known one
    return best_solution

if __name__ == "__main__":
    print(construct())
