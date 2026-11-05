import time
import math
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    
    start_time = time.time()
    time_limit = 115 # Leave 5s buffer
    
    # 1. Generate Permutations
    perms = list(permutations(range(1, n+1)))
    N = len(perms)
    
    # 2. Build Cost Matrix
    # Cost[i][j] = n - overlap(perms[i], perms[j])
    # We want to minimize sum of costs.
    # Initialize with n (max cost)
    cost = [[n] * N for _ in range(N)]
    
    # Fill using grouping
    # k from n-1 down to 1
    for k in range(n-1, 0, -1):
        # Map suffix of length k to list of indices
        suffix_map = {}
        for i, p in enumerate(perms):
            s = p[-k:]
            if s not in suffix_map:
                suffix_map[s] = []
            suffix_map[s].append(i)
            
        # Map prefix of length k to list of indices
        prefix_map = {}
        for j, p in enumerate(perms):
            s = p[:k]
            if s not in prefix_map:
                prefix_map[s] = []
            prefix_map[s].append(j)
            
        # Update costs
        # For each key s in suffix_map:
        #   i_list = suffix_map[s]
        #   j_list = prefix_map.get(s, [])
        #   for i in i_list:
        #     for j in j_list:
        #       cost[i][j] = min(cost[i][j], n - k)
        
        for s, i_list in suffix_map.items():
            j_list = prefix_map.get(s, [])
            if not j_list:
                continue
            new_cost = n - k
            for i in i_list:
                row = cost[i]
                for j in j_list:
                    if row[j] > new_cost:
                        row[j] = new_cost

    # 3. Initial Solution: Greedy Nearest Neighbor
    visited = [False] * N
    tour = []
    
    # Start with a random permutation or 0
    current = 0
    visited[0] = True
    tour.append(0)
    
    for _ in range(N - 1):
        best_next = -1
        best_cost = float('inf')
        curr_cost_row = cost[current]
        for j in range(N):
            if not visited[j]:
                c = curr_cost_row[j]
                if c < best_cost:
                    best_cost = c
                    best_next = j
        if best_next == -1:
            break
        visited[best_next] = True
        tour.append(best_next)
        current = best_next
        
    # 4. Local Search
    def calc_tour_cost(t):
        total = 0
        for i in range(len(t) - 1):
            total += cost[t[i]][t[i+1]]
        return total
        
    best_tour = tour[:]
    best_cost_val = calc_tour_cost(best_tour)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 2-Opt
        # Try reversing segments
        for i in range(N - 1):
            if time.time() - start_time > time_limit: break
            for j in range(i + 1, N):
                # Reverse best_tour[i:j+1]
                # Edges: (i-1, i) and (j, j+1) removed
                # Edges: (i-1, j) and (i, j+1) added
                # Handle boundaries
                prev_i = i - 1
                next_j = j + 1
                
                if prev_i < 0: prev_i = N - 1
                if next_j >= N: next_j = 0
                
                old_cost = cost[best_tour[prev_i]][best_tour[i]] + cost[best_tour[j]][best_tour[next_j]]
                new_cost = cost[best_tour[prev_i]][best_tour[j]] + cost[best_tour[i]][best_tour[next_j]]
                
                if new_cost < old_cost:
                    best_tour[i:j+1] = best_tour[i:j+1][::-1]
                    best_cost_val += new_cost - old_cost
                    improved = True
        
        # Or-Opt (Move 1 element)
        if time.time() - start_time < time_limit:
            for i in range(N):
                if time.time() - start_time > time_limit: break
                elem = best_tour[i]
                # Remove i
                # Be careful with indices
                # Cost change:
                # Remove edges (i-1, i) and (i, i+1)
                # Add edge (i-1, i+1)
                # Insert at j:
                # Remove edge (j-1, j)
                # Add edges (j-1, elem) and (elem, j)
                
                # This is complex to do in-place efficiently without errors.
                # Given time, maybe skip Or-Opt if 2-opt is fast enough?
                # 2-opt is $O(N^2)$.
                # Let's just run 2-opt repeatedly.
                pass

    # 5. Reconstruct
    word = list(perms[best_tour[0]])
    for i in range(1, N):
        prev_p = perms[best_tour[i-1]]
        curr_p = perms[best_tour[i]]
        
        # Find max overlap
        # We know cost = n - overlap
        # overlap = n - cost
        overlap = n - cost[best_tour[i-1]][best_tour[i]]
        
        # Append curr_p[overlap:]
        word.extend(curr_p[overlap:])
        
    return "".join(map(str, word))
