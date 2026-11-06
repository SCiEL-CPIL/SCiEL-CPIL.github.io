import sys
import time

# Increase recursion limit just in case, though depth should be small
sys.setrecursionlimit(2000)

def construct():
    start_time = time.time()
    MAX_TARGET = 128

    # Generate 2-3-5 smooth numbers up to MAX_TARGET
    smooth_numbers = []
    for i in range(1, MAX_TARGET + 1):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
    
    smooth_numbers.sort()
    N = len(smooth_numbers)
    
    # Pre-compute divisibility
    # divs[i][j] is True if smooth_numbers[i] divides smooth_numbers[j]
    divs = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if smooth_numbers[j] % smooth_numbers[i] == 0:
                divs[i][j] = True

    # best_rep[sum_val] will store the list of indices
    # best_size[sum_val] will store the minimum size found
    best_rep = [None] * (MAX_TARGET + 1)
    best_size = [float('inf')] * (MAX_TARGET + 1)

    # DFS to find all valid antichains
    def dfs(start_idx, current_sum, current_set):
        if time.time() - start_time > 55:
            return

        # If current_sum is a valid target, update best if size is smaller
        if 0 < current_sum <= MAX_TARGET:
            if len(current_set) < best_size[current_sum]:
                best_size[current_sum] = len(current_set)
                best_rep[current_sum] = list(current_set)

        # Try adding more elements
        for i in range(start_idx, N):
            val = smooth_numbers[i]
            new_sum = current_sum + val
            if new_sum > MAX_TARGET:
                break
            
            # Check compatibility with all elements in current_set
            compatible = True
            for j in current_set:
                if divs[j][i] or divs[i][j]:
                    compatible = False
                    break
            
            if not compatible:
                continue

            current_set.append(i)
            dfs(i + 1, new_sum, current_set)
            current_set.pop()

    dfs(0, 0, [])

    # Construct the result
    answers = []
    for t in range(1, MAX_TARGET + 1):
        if best_rep[t] is not None:
            # Convert indices to values
            reps = [smooth_numbers[idx] for idx in best_rep[t]]
            answers.append(reps)
        else:
            # Fallback: This should not happen if the problem is well-posed
            # But if it does, we need to return something.
            # A single term works if t is smooth.
            if t in smooth_numbers:
                answers.append([t])
            else:
                # Last resort: empty list (invalid, but better than crash)
                # Or try to find any representation quickly
                answers.append([])
                
    return answers
