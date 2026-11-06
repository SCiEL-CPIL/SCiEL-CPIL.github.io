"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter to only divisors <= target
    divisors = [d for d in divisors if d <= target]
    # Sort in descending order for better pruning (larger numbers first)
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # Precompute prefix sums for max sum pruning
    # prefix_sums[i] = sum of divisors[0:i]
    # But we need sum of top k from index i.
    # Let's just compute sums on the fly or use a helper.
    # Given small N (60), on-the-fly sum is fine, but precomputation is safer.
    
    # For min sum pruning, we need the sum of the smallest k divisors available.
    # Since list is descending, the smallest are at the end.
    # If we are at index i, we can pick from i to end.
    # The smallest k divisors in divisors[i:] are divisors[-k:] 
    # provided that len(divisors) - i >= k.
    
    start_time = time.time()
    deadline = start_time + 50
    
    def search(start_idx, remaining, depth, current_path):
        if time.time() > deadline:
            return False
            
        if depth == 0:
            return remaining == 0
            
        if remaining < 0:
            return False
            
        # Not enough divisors left
        if num_divs - start_idx < depth:
            return False
            
        # Max possible sum with 'depth' divisors starting from start_idx
        # These are the largest 'depth' divisors available
        max_sum = sum(divisors[start_idx : start_idx + depth])
        if remaining > max_sum:
            return False
            
        # Min possible sum with 'depth' divisors starting from start_idx
        # These are the smallest 'depth' divisors in the whole list (which are in the suffix)
        # Note: divisors is sorted descending.
        # The smallest 'depth' divisors are divisors[-depth:]
        # They are available if start_idx <= num_divs - depth
        min_sum = sum(divisors[-depth:])
        if remaining < min_sum:
            return False
            
        # Additional pruning: if remaining is small, we can limit the search
        # divisors are sorted descending, so if divisors[start_idx] > remaining, we can't use it?
        # No, we can skip it. But if divisors[start_idx] < remaining, we must use larger ones?
        # Actually, since we are picking 'depth' items, and they are distinct.
        
        for i in range(start_idx, num_divs):
            d = divisors[i]
            if d > remaining:
                continue  # Since sorted descending, all subsequent are smaller, so we can break?
                # Wait, if d > remaining, we can't use d. But we can use smaller ones.
                # So we should NOT break, just continue.
                # However, if d <= remaining, we proceed.
                # If d > remaining, we skip.
                # But since sorted descending, once d <= remaining, all subsequent are <= remaining.
                # So we can break out of the "skip" logic? No, we just start checking from the first valid one.
                # Actually, if d > remaining, we skip.
                # If we find d <= remaining, we try it.
                # We can optimize by finding the first index j such that divisors[j] <= remaining.
                # But the loop handles it.
            
            # Heuristic: If d is too small, we might not reach the sum with remaining depth-1 items?
            # Max sum of remaining depth-1 items from i+1 is sum(divisors[i+1 : i+1+depth-1])
            # If d + max_sum_rest < remaining, then we can break?
            # No, if d is small, we need larger d. But we are iterating i increasing (smaller d).
            # So if we pick a small d, the remaining sum is large.
            # The max sum we can get from remaining depth-1 items is sum(divisors[i+1 : i+1+depth-1]).
            # If remaining - d > that max sum, then we can't reach the target.
            # Since d is decreasing as i increases, remaining-d is increasing? No.
            # d decreases, so remaining-d increases.
            # So if remaining - d > max_sum_rest, we can break?
            # Yes, because for any j > i, d_j < d_i, so remaining - d_j > remaining - d_i.
            # And max_sum_rest for j will be smaller or equal.
            # So we can break.
            
            if depth > 1:
                max_rest = sum(divisors[i+1 : i+1+depth-1])
                if remaining - d > max_rest:
                    break
            
            current_path.append(d)
            if search(i + 1, remaining - d, depth - 1, current_path):
                return True
            current_path.pop()
            
        return False

    # Iterative deepening
    for k in range(1, num_divs + 1):
        path = []
        if search(0, target, k, path):
            return path
            
    # Fallback (should not happen)
    return []

# EVOLVE-BLOCK-END
