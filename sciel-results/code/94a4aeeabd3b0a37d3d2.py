import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute cost matrix
    # C[i][j] = cost to go from perms[i] to perms[j]
    # Cost is n - overlap_length
    # overlap_length is max k such that perms[i][n-k:] == perms[j][:k]
    
    # To speed up, we can precompute suffixes and prefixes?
    # Or just loop.
    
    # Let's try to optimize the inner loop.
    # For each i, we have suffixes: s_1 = p[-1:], s_2 = p[-2:], etc.
    # For each j, we have prefixes: p_1 = q[:1], p_2 = q[:2], etc.
    
    # We can create a dictionary for each i? No.
    
    # Let's just run the double loop.
    # 25M iterations. In Python, ~1-2s per 10M simple ops? Maybe 5-10s.
    
    C = [[0] * N for _ in range(N)]
    
    for i in range(N):
        p = perms[i]
        # Precompute suffixes for p
        suffixes = [p[n-k:] for k in range(1, n+1)] # suffixes[k-1] is last k chars
        for j in range(N):
            q = perms[j]
            # Find max k
            # Check from largest k to smallest
            for k in range(n, 0, -1):
                if suffixes[k-1] == q[:k]:
                    C[i][j] = n - k
                    break
            else:
                C[i][j] = n # no overlap, append all n
        
    # Check time
    if time.time() > deadline - 30:
         # If precomputation took too long, fall back to classic
         return _classic_construct(n)

    # Nearest Neighbor to get initial path
    # Start with a random permutation
    start = random.randint(0, N - 1)
    visited = [False] * N
    path = [start]
    visited[start] = True
    current = start
    
    for _ in range(N - 1):
        best_next = -1
        best_cost = float('inf')
        row = C[current]
        # Find unvisited j with min cost
        for j in range(N):
            if not visited[j]:
                c = row[j]
                if c < best_cost:
                    best_cost = c
                    best_next = j
        path.append(best_next)
        visited[best_next] = True
        current = best_next
        
    # Calculate initial length
    def calc_length(path):
        length = n
        for i in range(len(path) - 1):
            length += C[path[i]][path[i+1]]
        return length

    current_len = calc_length(path)
    
    # 2-Opt Local Search
    # We want to minimize length.
    # Reversing a segment (i, j) changes cost at boundaries.
    # Old cost: C[path[i-1]][path[i]] + C[path[j]][path[j+1]]
    # New cost: C[path[i-1]][path[j]] + C[path[i]][path[j+1]]
    # Note: The internal costs of the reversed segment remain the same?
    # No, the edges inside the segment are reversed.
    # Edge (u, v) becomes (v, u). Cost C[u][v] vs C[v][u].
    # In superpermutations, C[u][v] is not necessarily equal to C[v][u].
    # So we must recalculate the whole segment cost or maintain a running sum.
    
    # Efficient 2-opt for asymmetric TSP:
    # Delta = (C[prev[i]][path[j]] + C[path[i]][next[j]]) - (C[prev[i]][path[i]] + C[path[j]][next[j]])
    # PLUS the change in internal edges.
    # This is complex.
    
    # Simpler: Just try all pairs (i, j) and calculate the new total length.
    # O(N^2) per pass. N=5040. N^2 = 25M.
    # 25M operations per pass.
    # If we do 10 passes, 250M ops. Too slow in Python.
    
    # We need a faster 2-opt or fewer passes.
    # Or use a "fast" 2-opt that only checks promising moves.
    
    # Alternative: Or-opt (move a single node) or 3-opt.
    # Let's try a simple 2-opt but break early if no improvement in a full sweep.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(1, N - 2):
            if time.time() > deadline - 5:
                break
            prev_i = path[i-1]
            curr_i = path[i]
            # We can optimize by only checking j > i
            for j in range(i + 1, N - 1):
                next_j = path[j+1]
                curr_j = path[j]
                
                # Calculate delta
                # Old edges: (prev_i, curr_i) and (curr_j, next_j)
                # New edges: (prev_i, curr_j) and (curr_i, next_j)
                # Internal edges reversed: (path[i+1], path[i+2]) ... (path[j-1], path[j])
                # Old internal: sum(C[path[k]][path[k+1]] for k in i..j-1)
                # New internal: sum(C[path[k+1]][path[k]] for k in i..j-1)
                
                # This is O(N) per pair, making it O(N^3). Too slow.
                
                # We need to precompute prefix sums of costs?
                # But costs change when we reverse? No, the set of edges is the same, just direction reversed.
                # Let's maintain an array `costs` where costs[k] = C[path[k]][path[k+1]].
                # Then sum of internal costs is sum(costs[i:j]).
                # We can use prefix sums of costs.
                
                # But updating prefix sums after a move is O(N).
                # So each 2-opt move is O(N).
                # Checking all pairs is O(N^2) moves * O(N) update = O(N^3).
                
                # This is too slow for N=5040.
                
                # Let's use a different heuristic: **Greedy Repair**.
                # Or just accept the Nearest Neighbor result?
                # NN is usually close to optimal for these structured graphs.
                
                # Let's try to do a few random restarts of NN and pick the best.
                # NN is O(N^2). 10 restarts = 250M ops. Still slow.
                
                # Let's stick to one NN path and try a limited 2-opt with early termination.
                # Or, just return the NN path.
                
                pass # Skip detailed 2-opt for now to ensure we finish.
        
    # Convert path to string
    result = list(perms[path[0]])
    for i in range(1, len(path)):
        prev = perms[path[i-1]]
        curr = perms[path[i]]
        # Find overlap
        for k in range(n, 0, -1):
            if prev[n-k:] == curr[:k]:
                result.extend(curr[k:])
                break
        else:
            result.extend(curr)
            
    return "".join(map(str, result))

def _classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
