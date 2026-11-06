import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, leaving 10s safety margin
    
    # 1. Generate 2-3-5 smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
            
    N = len(smooth_numbers)
    smooth_set = set(smooth_numbers)
    smooth_index = {v: i for i, v in enumerate(smooth_numbers)}
    
    # Precompute incompatibility matrix
    # incompat[i][j] is True if smooth_numbers[i] and smooth_numbers[j] cannot be in the same antichain
    incompat = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            a, b = smooth_numbers[i], smooth_numbers[j]
            if b % a == 0 or a % b == 0:
                incompat[i][j] = True
                incompat[j][i] = True
                
    answers = [None] * 129  # Index 0 unused, 1..128
    
    def time_check():
        return time.time() > deadline

    # Helper to find a representation for a specific target T with exactly k terms
    # Returns list of values or None
    def find_rep(T, k):
        # Recursive or iterative? Iterative with pruning is better for small k
        # We will use iterative nesting for k up to 5
        # For efficiency, we pass the list of indices chosen so far and the sum so far
        
        # Base case: if k == 1
        if k == 1:
            if T in smooth_set:
                idx = smooth_index[T]
                return [smooth_numbers[idx]]
            return None
        
        # For k > 1, we iterate
        # To keep it simple and fast for k<=5, we can unroll or use a helper
        # But unrolling k=3,4,5 is verbose.
        # Let's use a recursive helper with pruning.
        
        def dfs(start_idx, remaining_sum, terms_left, current_path):
            if terms_left == 0:
                return current_path if remaining_sum == 0 else None
            if remaining_sum < 0:
                return None
            if terms_left == 1:
                if remaining_sum in smooth_set:
                    idx = smooth_index[remaining_sum]
                    if idx > start_idx:
                        # Check compatibility with all in current_path
                        for p_idx in current_path:
                            if incompat[p_idx][idx]:
                                return None
                        return current_path + [smooth_numbers[idx]]
                return None
            
            # Pruning: if the sum of the smallest 'terms_left' smooth numbers starting from start_idx
            # is greater than remaining_sum, no solution.
            # Since smooth numbers are sorted, the smallest available are at start_idx, start_idx+1...
            # We need to pick 'terms_left' distinct indices > start_idx.
            # The minimum sum is sum(smooth_numbers[start_idx : start_idx + terms_left])
            # But we must ensure compatibility.
            # Simple bound: if sum of first 'terms_left' available (ignoring compat) > remaining_sum, prune.
            min_possible_sum = 0
            for i in range(start_idx, N):
                min_possible_sum += smooth_numbers[i]
                if len(range(start_idx, i+1)) == terms_left:
                    break
            if min_possible_sum > remaining_sum:
                return None
                
            for i in range(start_idx, N):
                # Prune if even adding this number exceeds remaining_sum
                if smooth_numbers[i] > remaining_sum:
                    break
                new_remaining = remaining_sum - smooth_numbers[i]
                # Check compatibility with current_path
                compatible = True
                for p_idx in current_path:
                    if incompat[p_idx][i]:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                res = dfs(i + 1, new_remaining, terms_left - 1, current_path + [i])
                if res is not None:
                    return [smooth_numbers[idx] for idx in res]
            return None

        return dfs(0, T, k, [])

    # k=1
    for i in range(N):
        s = smooth_numbers[i]
        if s <= 128 and answers[s] is None:
            answers[s] = [smooth_numbers[i]]
            
    # k=2
    # Target-driven for k=2 is also fast
    for t in range(1, 129):
        if answers[t] is None:
            if time_check():
                break
            # Try to find pair
            for i in range(N):
                a = smooth_numbers[i]
                if a >= t: break
                b = t - a
                if b in smooth_set:
                    j = smooth_index[b]
                    if j > i and not incompat[i][j]:
                        answers[t] = [a, b]
                        break
    
    # k=3
    for t in range(1, 129):
        if answers[t] is None:
            if time_check():
                break
            rep = find_rep(t, 3)
            if rep:
                answers[t] = rep

    # k=4
    for t in range(1, 129):
        if answers[t] is None:
            if time_check():
                break
            rep = find_rep(t, 4)
            if rep:
                answers[t] = rep

    # k=5
    for t in range(1, 129):
        if answers[t] is None:
            if time_check():
                break
            rep = find_rep(t, 5)
            if rep:
                answers[t] = rep

    # Fallback for any remaining (should be rare if minimal k <= 5)
    for t in range(1, 129):
        if answers[t] is None:
            for k in range(6, 10):
                if time_check():
                    break
                rep = find_rep(t, k)
                if rep:
                    answers[t] = rep
                    break

    return answers[1:]
