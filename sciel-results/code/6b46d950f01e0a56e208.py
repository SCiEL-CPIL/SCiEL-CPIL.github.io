import itertools
import time
import random
import numpy as np

def construct(n):
    if n <= 4:
        # Classic construction is optimal for small n
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

    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Convert to strings for overlap calculation
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute overlap matrix using NumPy for speed
    # M[i, j] = overlap of perm_strs[i] and perm_strs[j]
    # Overlap k means perm_strs[i][n-k:] == perm_strs[j][:k]
    # Max overlap is n-1.
    
    # We can compute this by checking suffixes of i and prefixes of j.
    # For k in range(n-1, 0, -1):
    #   if suffix_k(i) == prefix_k(j): M[i,j] = k; break
    #   else M[i,j] = 0
    
    # Vectorized approach for overlap matrix is tricky due to variable length matching.
    # However, N=5040, so N^2 = 25M.
    # Let's try to build it efficiently.
    
    # Since n is small (7), we can use a dictionary of suffixes to indices.
    # But we need pairwise overlap.
    
    # Alternative: Just calculate on the fly in the SA loop?
    # No, SA needs O(1) cost updates. We need the matrix.
    
    # Let's build the matrix. 25M entries.
    # Using a Python loop for 25M items is ~10-20s.
    # Using NumPy is hard for string comparison.
    # Let's use a trick: encode permutations as integers?
    # No, let's just use a Python loop but optimize it.
    
    # Actually, we can compute the cost of a path incrementally.
    # If we use Swap moves, we only need M[i][j] for specific pairs.
    # But we need M[i][j] for arbitrary i, j during the search.
    # So we must precompute M.
    
    # Let's try to precompute M using a list of suffixes.
    # For each k from 1 to n-1:
    #   Map suffix of length k to list of indices that have this suffix.
    #   Map prefix of length k to list of indices that have this prefix.
    #   Then for each pair of suffix/prefix match, we have a potential overlap.
    #   We want the MAXIMUM k.
    
    # This is getting complex. Let's stick to a simpler local search:
    # Random Restart with 2-Opt, but limit the number of 2-Opt passes.
    # Or, use the fact that n=7 is small enough that we can afford a slower precomputation if it leads to a much better result.
    
    # Let's try a different approach: **Greedy with Randomization**
    # 1. Start with a random permutation.
    # 2. Repeatedly pick the best next unvisited permutation (greedy) but with some randomness.
    # This is still similar to NN.
    
    # Let's go back to **Simulated Annealing with Swap**.
    # To make it fast, we need the overlap matrix.
    # Let's build it.
    
    # Optimization:
    # For each permutation i, compute its suffixes.
    # For each permutation j, compute its prefixes.
    # We can store prefixes in a dictionary: prefix_str -> list of indices.
    # Then for each i, for each k in 1..n-1:
    #   suffix = perm_strs[i][n-k:]
    #   if suffix in prefix_dict:
    #      for j in prefix_dict[suffix]:
    #         M[i][j] = max(M[i][j], k)
    
    # This is efficient because the number of matches is small.
    # Total operations: sum over i of (n-1) * (number of matches).
    # Number of matches for a specific suffix is roughly N / (n^k).
    # So total ops is roughly N * sum(1/n^k) * N = N^2 * constant.
    # This is feasible.
    
    prefix_dict = {}
    for j in range(N):
        s = perm_strs[j]
        for k in range(1, n):
            prefix = s[:k]
            if prefix not in prefix_dict:
                prefix_dict[prefix] = []
            prefix_dict[prefix].append(j)
            
    M = np.zeros((N, N), dtype=np.uint8)
    
    for i in range(N):
        s = perm_strs[i]
        for k in range(1, n):
            suffix = s[n-k:]
            if suffix in prefix_dict:
                for j in prefix_dict[suffix]:
                    # We want the max overlap. Since we iterate k from 1 to n-1,
                    # we should iterate k from n-1 down to 1 to set the max directly?
                    # Or just update max.
                    if k > M[i, j]:
                        M[i, j] = k
                        
    # Now M is ready.
    # Cost of path P: n + sum(n - M[P[i], P[i+1]])
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 115  # Leave 5s for safety
    
    best_path = None
    best_cost = float('inf')
    
    # Multiple restarts
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        # Initialize random path
        path = list(range(N))
        random.shuffle(path)
        
        # Calculate initial cost
        cost = n
        for i in range(N - 1):
            cost += n - M[path[i], path[i+1]]
            
        if cost < best_cost:
            best_cost = cost
            best_path = path[:]
            
        # Simulated Annealing
        T = 1000.0
        T_min = 1.0
        cooling = 0.9999
        
        current_cost = cost
        current_path = path[:]
        
        while T > T_min:
            # Pick two random indices to swap
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
                
            # Calculate delta cost
            # Swapping i and j affects edges:
            # If i and j are not adjacent:
            # Old edges: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
            # New edges: (i-1, j), (j, i+1), (j-1, i), (i, j+1)
            # Need to handle boundaries (i=0, i=N-1, etc.)
            
            # Let's define a helper to get cost of edge (a, b)
            # edge_cost(a, b) = n - M[a, b] if a != -1 and b != -1 else 0
            
            # To simplify, let's just calculate the cost of the affected segments.
            # Indices in path: ... A, P[i], B ... and ... C, P[j], D ...
            # After swap: ... A, P[j], B ... and ... C, P[i], D ...
            
            # Edges involved:
            # 1. Predecessor of i (A) to i (P[i]) -> A to j (P[j])
            # 2. i (P[i]) to successor of i (B) -> j (P[j]) to B
            # 3. Predecessor of j (C) to j (P[j]) -> C to i (P[i])
            # 4. j (P[j]) to successor of j (D) -> i (P[i]) to D
            
            # If i and j are adjacent, the logic is slightly different (shared edge).
            # Let's assume i < j for simplicity in index calculation, but they are values in the path.
            # Actually, i and j are indices in the `path` list.
            
            pi = current_path[i]
            pj = current_path[j]
            
            # Get neighbors
            # Left of i
            if i > 0:
                li = current_path[i-1]
            else:
                li = -1
            # Right of i
            if i < N - 1:
                ri = current_path[i+1]
            else:
                ri = -1
            # Left of j
            if j > 0:
                lj = current_path[j-1]
            else:
                lj = -1
            # Right of j
            if j < N - 1:
                rj = current_path[j+1]
            else:
                rj = -1
            
            # Calculate old cost contribution
            old_cost = 0
            if li != -1: old_cost += n - M[li, pi]
            if ri != -1: old_cost += n - M[pi, ri]
            if lj != -1: old_cost += n - M[lj, pj]
            if rj != -1: old_cost += n - M[pj, rj]
            
            # Calculate new cost contribution
            # Swap pi and pj in path
            new_cost = 0
            # Left of i now connects to pj
            if li != -1: new_cost += n - M[li, pj]
            # Right of i (ri) now connects from pj
            if ri != -1: new_cost += n - M[pj, ri]
            # Left of j (lj) now connects to pi
            if lj != -1: new_cost += n - M[lj, pi]
            # Right of j (rj) now connects from pi
            if rj != -1: new_cost += n - M[pi, rj]
            
            delta = new_cost - old_cost
            
            # Accept if delta < 0 or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Perform swap
                current_path[i], current_path[j] = pj, pi
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = current_path[:]
            
            T *= cooling
            
            # Check time
            if time.time() - start_time > time_limit:
                break
        
        # Restart with slightly different parameters if needed
        if restarts > 10:
            break # Limit restarts to avoid infinite loop if time check fails
            
    # Build result string from best_path
    res = perm_strs[best_path[0]]
    for i in range(1, N):
        ov = M[best_path[i-1], best_path[i]]
        res += perm_strs[best_path[i]][ov:]
        
    return res
