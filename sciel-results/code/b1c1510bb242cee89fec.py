import itertools
import random
import time
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix
    # M[i][j] is the largest k such that suffix of perms[i] of length k == prefix of perms[j] of length k
    # k ranges from 0 to n-1
    # We want to maximize sum of M[path[i]][path[i+1]]
    M = [[0] * N for _ in range(N)]
    
    # Optimization: Use prefix/suffix maps for faster calculation?
    # For N=5040, N^2 = 25M entries. 
    # Calculating overlap is O(n). 25M * 7 = 175M ops. Might be slow in Python.
    # Let's optimize overlap calculation.
    
    # Better: Precompute for each permutation, its suffixes and prefixes.
    # Actually, just iterating k from n-1 down to 0 is fine if we break early.
    # But 25M iterations in Python is ~1-2 seconds. Acceptable.
    
    # Let's try to speed it up.
    # For each i, for each j:
    #   for k in range(n-1, -1, -1):
    #      if perms[i][n-k:] == perms[j][:k]:
    #         M[i][j] = k
    #         break
         
    # This is O(N^2 * n).
    # We can do better by using the fact that perms are permutations.
    # But let's stick to the simple version first, maybe with some numpy or just efficient loops.
    
    # Actually, we can compute M row by row.
    # For a fixed i, we want to find for all j the overlap.
    
    # Let's just do it. 5040^2 * 7 is ~176 million simple ops. In Python, this might take 10-20 seconds.
    # We have 120s. This is feasible.
    
    start_time = time.time()
    deadline = start_time + 100
    
    # Precompute M
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            max_k = 0
            # Check overlaps from n-1 down to 0
            # We can use string slicing or tuple slicing
            # pi is a tuple
            for k in range(n - 1, -1, -1):
                if pi[n-k:] == pj[:k]:
                    max_k = k
                    break
            M[i][j] = max_k
            
    if time.time() > deadline:
        # Fallback if precomputation is too slow (unlikely)
        return "".join(map(str, perms[0])) # Invalid but fast exit
        
    # Initial Path: Standard Recursive Ordering
    # The standard recursive superpermutation S_n is constructed as follows:
    # S_1 = [1]
    # S_n is formed by taking S_{n-1}, and for each permutation p in S_{n-1},
    # we insert n after p? No.
    
    # The order of permutations in the standard superpermutation is:
    # 1. Start with S_{n-1}.
    # 2. The sequence of permutations in S_n is:
    #    p_1, p_1 with n appended? No.
    
    # Let's use the known property:
    # The permutations appear in the order:
    # 123...n, 123...n1, ...
    # Actually, let's just use a greedy construction to get a good initial path, 
    # or use the "recursive" order which is known to be close to optimal.
    
    # A simple greedy nearest neighbor (max overlap) is a good start too.
    # But the recursive order is better.
    
    # Let's generate the recursive order.
    # S_1 = [1]
    # S_n: 
    #   Take S_{n-1}.
    #   For each permutation p in S_{n-1}, the new permutations are not just p+n.
    #   The structure is:
    #   S_n = p_1, p_1 n, p_2, p_2 n, ..., p_m, p_m n, p_m, p_{m-1} n, ...
    #   This is getting complex to implement correctly from scratch.
    
    # Alternative: Just use Greedy Nearest Neighbor to get a decent start, then SA.
    # Greedy NN:
    # Start with random perm.
    # Next perm is the unvisited one with max overlap.
    
    # Let's try to get the recursive order properly.
    # Reference: "The shortest superpermutation problem"
    # S_n is built from S_{n-1} by:
    # 1. Append n to every permutation in S_{n-1} to get a set of perms ending in n.
    # 2. The sequence is:
    #    p_1, p_1 n, p_2, p_2 n, ..., p_{(n-1)!}, p_{(n-1)!} n, p_{(n-1)!}, p_{(n-1)!-1} n, ...
    #    Wait, the standard construction is:
    #    S_n = concat( p_i n p_{i+1} n ... )
    
    # Let's use a simpler heuristic start: Random Restart Greedy.
    # We will run a few greedy constructions to find a good start, or just one.
    
    # Greedy Construction:
    # 1. Pick random start.
    # 2. At each step, pick the unvisited permutation with the maximum overlap with the current one.
    # 3. If tie, pick randomly.
    
    path = []
    visited = [False] * N
    current = random.randint(0, N - 1)
    path.append(current)
    visited[current] = True
    
    for _ in range(N - 1):
        max_overlap = -1
        best_next = -1
        # Optimization: We can pre-sort candidates? No, N is small enough.
        for j in range(N):
            if not visited[j]:
                if M[current][j] > max_overlap:
                    max_overlap = M[current][j]
                    best_next = j
                    # If we find max possible (n-1), we can break?
                    # Max overlap is n-1.
                    if max_overlap == n - 1:
                        break
        if best_next == -1:
            # Fallback: pick any unvisited
            for j in range(N):
                if not visited[j]:
                    best_next = j
                    break
        path.append(best_next)
        visited[best_next] = True
        current = best_next
        
    # Calculate initial cost (sum of overlaps)
    def calc_cost(path):
        cost = 0
        for i in range(len(path) - 1):
            cost += M[path[i]][path[i+1]]
        return cost
        
    current_cost = calc_cost(path)
    
    # Simulated Annealing
    # T_start, T_end, steps
    # We want to maximize cost.
    
    T = 10.0
    T_min = 0.1
    alpha = 0.9995 # Cooling rate
    # We have ~100s left.
    # Each step is O(1) or O(n) for 2-opt.
    # 2-opt: Reverse segment [i, j].
    # Cost change can be calculated in O(1) if we store the path.
    
    # Let's use 2-opt moves.
    # Move: Select i < j. Reverse path[i:j+1].
    # This changes overlaps at (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1).
    # Specifically, the internal overlaps of the segment are reversed and their values change.
    # M[a][b] != M[b][a].
    
    # For simplicity and speed, let's use **Swap** moves (swap two elements) or **Insert** moves.
    # Swap i, j:
    # Affected edges: (i-1, i), (i, i+1), (j-1, j), (j, j+1) and the edge between i and j if they were adjacent?
    # If i and j are not adjacent:
    # Old edges: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
    # New edges: (i-1, j), (j, i+1), (j-1, i), (i, j+1)
    # Wait, if we swap path[i] and path[j]:
    # Old: ... A, P_i, B ... C, P_j, D ...
    # New: ... A, P_j, B ... C, P_i, D ...
    # Changed edges:
    # 1. (A, P_i) -> (A, P_j)
    # 2. (P_i, B) -> (P_j, B)
    # 3. (C, P_j) -> (C, P_i)
    # 4. (P_j, D) -> (P_i, D)
    # If i and j are adjacent, it's simpler.
    
    # Let's implement Swap.
    
    best_path = path[:]
    best_cost = current_cost
    
    steps = 0
    max_steps = 2000000 # Estimate
    
    while time.time() < deadline and steps < max_steps:
        if T < T_min:
            break
            
        # Pick random i, j
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
            
        # Ensure i < j for consistency? Not necessary for swap.
        
        # Calculate delta
        # Indices: i-1, i, i+1, j-1, j, j+1
        # Handle boundaries
        
        # Get values
        pi = path[i]
        pj = path[j]
        
        # Old overlaps
        old_cost = 0
        
        # Edge (i-1, i)
        if i > 0:
            old_cost += M[path[i-1]][pi]
        # Edge (i, i+1)
        if i < N - 1:
            old_cost += M[pi][path[i+1]]
            
        # Edge (j-1, j)
        if j > 0:
            old_cost += M[path[j-1]][pj]
        # Edge (j, j+1)
        if j < N - 1:
            old_cost += M[pj][path[j+1]]
            
        # New overlaps
        new_cost = 0
        
        # Edge (i-1, j) -> path[i] becomes pj
        if i > 0:
            new_cost += M[path[i-1]][pj]
        # Edge (j, i+1) -> path[i] is pj, next is path[i+1]
        if i < N - 1:
            new_cost += M[pj][path[i+1]]
            
        # Edge (j-1, i) -> path[j] becomes pi
        if j > 0:
            new_cost += M[path[j-1]][pi]
        # Edge (i, j+1) -> path[j] is pi, next is path[j+1]
        if j < N - 1:
            new_cost += M[pi][path[j+1]]
            
        delta = new_cost - old_cost
        
        # Accept if delta > 0 or with probability exp(delta/T)
        if delta > 0 or random.random() < math.exp(delta / T):
            # Perform swap
            path[i], path[j] = path[j], path[i]
            current_cost += delta
            
            if current_cost > best_cost:
                best_cost = current_cost
                best_path = path[:]
                
        T *= alpha
        steps += 1
        
    # Convert best_path to string
    word = []
    for idx in best_path:
        p = perms[idx]
        if not word:
            word.extend(p)
        else:
            # Overlap with previous
            prev_idx = best_path[best_path.index(idx) - 1] # This is O(N) and wrong if duplicates? No, unique.
            # Better: iterate with index
            pass
            
    # Let's rebuild properly
    final_word = []
    for k in range(N):
        idx = best_path[k]
        p = perms[idx]
        if k == 0:
            final_word.extend(p)
        else:
            prev_idx = best_path[k-1]
            # Find overlap
            # We have M[prev_idx][idx]
            k_overlap = M[prev_idx][idx]
            final_word.extend(p[k_overlap:])
            
    return "".join(map(str, final_word))
