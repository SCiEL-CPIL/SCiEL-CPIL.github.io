import time

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    deadline = start_time + 85.0

    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # Precompute suffix sums for max possible sum from index i
    # suffix_sum[i] = sum(divisors[i:])
    suffix_sum = [0] * (num_divs + 1)
    for i in range(num_divs - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
        
    # Precompute prefix sums? Or just use suffix sums for max.
    # For min sum of 'k' items from 'divisors[i:]', it is sum of the last 'k' items of the whole array,
    # provided 'num_divs - k >= i'.
    
    def find_subset_of_size_k(k):
        """
        Returns a list of k divisors summing to target, or None.
        """
        if k == 0:
            return [] if target == 0 else None
        
        # If k is 1, just check if target is in divisors
        if k == 1:
            if target in divisors:
                return [target]
            return None

        # Precompute min_sum_k: sum of the k smallest divisors
        # Since divisors is sorted descending, the k smallest are divisors[-k:]
        min_sum_k = sum(divisors[-k:])
        
        # If even the k smallest sum to more than target, no solution
        if min_sum_k > target:
            return None
            
        # If the k largest sum to less than target, no solution
        max_sum_k = sum(divisors[:k])
        if max_sum_k < target:
            return None

        # Recursive search
        # State: index (start index in divisors), remaining (count to pick), current_sum
        # We want to pick 'remaining' items from divisors[index:]
        
        def backtrack(index, remaining, current_sum, path):
            if remaining == 0:
                if current_sum == target:
                    return path.copy()
                return None
            
            # We need to pick 'remaining' items from divisors[index:]
            # The last possible index to start picking is num_divs - remaining
            # because we need 'remaining' items.
            
            # Pruning:
            # 1. Max sum we can add with 'remaining' items starting from 'index'
            #    is sum(divisors[index : index+remaining])
            #    If current_sum + this < target, we can't reach target with this index or any higher index.
            #    Since divisors are sorted descending, sum(divisors[j : j+remaining]) decreases as j increases.
            #    So if it's too small here, it will be too small for all j >= index.
            if index > num_divs - remaining:
                return None
                
            max_add = sum(divisors[index : index + remaining])
            if current_sum + max_add < target:
                return None # Break/Return, as further indices will have even smaller max_add

            # 2. Min sum we can add with 'remaining' items.
            #    The smallest 'remaining' items in the entire array are divisors[-remaining:].
            #    As long as index <= num_divs - remaining, the suffix divisors[index:] contains divisors[-remaining:].
            #    So min_add is constant: sum(divisors[-remaining:])
            min_add = sum(divisors[-remaining:])
            if current_sum + min_add > target:
                # Even the smallest possible sum is too large.
                # Since we are iterating index upwards, the available set shrinks but the smallest elements remain the same (the global smallest).
                # Wait, if we move index up, we might exclude some small elements? No, we exclude LARGE elements.
                # The smallest elements are at the end. As long as we can pick 'remaining' items, we can pick the global smallest 'remaining'.
                # So if current_sum + min_add > target, then ANY choice of 'remaining' items from divisors[index:] will sum to > target.
                # So we can return None immediately for this branch.
                return None

            # Iterate possible next divisors
            # We must leave enough divisors for the remaining picks.
            # So i can go from index to num_divs - remaining.
            for i in range(index, num_divs - remaining + 1):
                d = divisors[i]
                new_sum = current_sum + d
                
                # If new_sum > target, since d is descending, subsequent d's are smaller.
                # But we are not breaking because smaller d might lead to valid sum with other picks?
                # No, if we pick d, the rest must sum to target - new_sum.
                # If new_sum > target, we can't use d.
                # Can we break? No, because a smaller d might be <= target.
                
                # Check if it's possible to reach target with this d
                if new_sum > target:
                    continue
                    
                # Check bounds for the remaining (remaining - 1) picks
                if remaining - 1 > 0:
                    # Min sum of (remaining-1) smallest divisors
                    min_rest = sum(divisors[-(remaining-1):])
                    if new_sum + min_rest > target:
                        # Even the smallest rest is too big.
                        # Since min_rest is constant (global smallest), and we are just picking from a subset,
                        # this condition holds for any completion. So skip this d.
                        continue
                        
                    # Max sum of (remaining-1) largest divisors from divisors[i+1:]
                    # This is sum(divisors[i+1 : i+1 + remaining - 1])
                    max_rest = sum(divisors[i+1 : i+1 + remaining - 1])
                    if new_sum + max_rest < target:
                        # Even the largest rest is not enough.
                        # For any j > i, max_rest will be smaller.
                        # So we can break the loop.
                        break
                else:
                    # remaining - 1 == 0
                    if new_sum == target:
                        return path + [d]
                    # If new_sum != target, no solution for this branch
                    continue

                path.append(d)
                res = backtrack(i + 1, remaining - 1, new_sum, path)
                if res is not None:
                    return res
                path.pop()
                
            return None

        return backtrack(0, k, 0, [])

    # Try k from 1 to num_divs
    for k in range(1, num_divs + 1):
        if time.time() > deadline:
            break
        result = find_subset_of_size_k(k)
        if result is not None:
            return result
            
    # If no solution found (should not happen)
    return []

# Test locally
if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
    print(sum(res))
