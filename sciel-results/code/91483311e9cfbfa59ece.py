import time
import random
import itertools
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # For n=2,3,4,5,6,7 we can use the TSP approach
    # The problem is to order the permutations of 1..n-1 to maximize overlap
    
    # Get all permutations of 1..n-1
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    M = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # k can be 0 to n-2
    overlap = [[0] * M for _ in range(M)]
    
    for i in range(M):
        p1 = perms[i]
        for j in range(M):
            p2 = perms[j]
            # Max possible overlap is n-2
            # Check from n-2 down to 0
            max_k = n - 2
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap[i][j] = 0
                    break
                if p1[-k:] == p2[:k]:
                    overlap[i][j] = k
                    break
    
    # TSP Heuristic
    # We want to maximize sum of overlaps.
    # This is equivalent to minimizing sum of (max_overlap - overlap).
    # Let cost[i][j] = (n-2) - overlap[i][j].
    # We want to minimize total cost.
    
    start_time = time.time()
    time_limit = 145  # Safety margin for 150s limit
    
    best_order = None
    best_cost = float('inf')
    
    # 1. Multiple Nearest Neighbor searches
    # To speed up, we can use a "greedy" step that picks the neighbor with min cost.
    # We do this from multiple random starts.
    
    def nearest_neighbor(start):
        visited = [False] * M
        visited[start] = True
        current = start
        order = [start]
        
        for _ in range(M - 1):
            best_next = -1
            min_c = float('inf')
            # Find neighbor with min cost
            for next_idx in range(M):
                if not visited[next_idx]:
                    c = (n - 2) - overlap[current][next_idx]
                    if c < min_c:
                        min_c = c
                        best_next = next_idx
            if best_next == -1:
                break
            visited[best_next] = True
            order.append(best_next)
            current = best_next
        return order
    
    # Run NN from a few starts
    num_nn_runs = 20
    for s in range(num_nn_runs):
        if time.time() - start_time > time_limit - 20:
            break
        start = random.randint(0, M - 1)
        order = nearest_neighbor(start)
        cost = sum((n - 2) - overlap[order[i]][order[i+1]] for i in range(M - 1))
        if cost < best_cost:
            best_cost = cost
            best_order = order
    
    # 2. 2-Opt Improvement
    # Try to improve the best_order found by NN
    
    def two_opt(order):
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            N = len(order)
            # To make it faster, we can iterate i, j in a random order or limited range
            # Standard 2-opt is O(N^2) per pass. N=720. 720^2 = 500k.
            # In Python, 500k ops is fast (~0.1s).
            # We can do several passes.
            
            for i in range(N - 1):
                if time.time() - start_time > time_limit - 5:
                    break
                for j in range(i + 1, N):
                    # Calculate cost change
                    # Old: cost(order[i-1], order[i]) + cost(order[i], order[i+1]) ... 
                    # Actually, 2-opt reverses segment [i, j].
                    # Affected edges:
                    # (i-1, i) and (j, j+1) are boundaries.
                    # Internal edges (k, k+1) for k in [i, j-1] are reversed.
                    
                    old_cost = 0
                    new_cost = 0
                    
                    # Boundary i-1 -> i
                    if i > 0:
                        old_cost += (n - 2) - overlap[order[i-1]][order[i]]
                        new_cost += (n - 2) - overlap[order[i-1]][order[j]]
                    
                    # Boundary j -> j+1
                    if j < N - 1:
                        old_cost += (n - 2) - overlap[order[j]][order[j+1]]
                        new_cost += (n - 2) - overlap[order[j+1]][order[i]]
                    
                    # Internal
                    for k in range(i, j):
                        old_cost += (n - 2) - overlap[order[k]][order[k+1]]
                        new_cost += (n - 2) - overlap[order[k+1]][order[k]]
                    
                    if new_cost < old_cost:
                        order[i:j+1] = order[i:j+1][::-1]
                        improved = True
        
        return order
    
    if best_order:
        # Do a few 2-opt passes
        # Since 2-opt is expensive, we limit the number of full passes or iterations
        # We'll just run the while loop in two_opt which has a time check.
        best_order = two_opt(best_order)
        
        # Recalculate final cost
        final_cost = sum((n - 2) - overlap[best_order[i]][best_order[i+1]] for i in range(M - 1))
        if final_cost < best_cost:
            best_cost = final_cost

    # If TSP failed or didn't run, fallback to standard order (lexicographic or similar)
    if not best_order:
        best_order = list(range(M))

    # Construct the word
    # W = P_sigma(1)
    # For k=2..M: W += [n] + P_sigma(k)[overlap:]
    
    word = []
    first_perm = perms[best_order[0]]
    word.extend(first_perm)
    
    for k in range(1, M):
        prev_perm = perms[best_order[k-1]]
        curr_perm = perms[best_order[k]]
        
        # Find overlap
        ov = overlap[best_order[k-1]][best_order[k]]
        
        word.append(n)
        # Append curr_perm[ov:]
        word.extend(curr_perm[ov:])
        
    return "".join(map(str, word))

if __name__ == "__main__":
    # Test for n=7
    # w = construct(7)
    # print(len(w))
    pass
