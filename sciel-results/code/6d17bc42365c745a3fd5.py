import time
import itertools
import sys

def construct(n):
    start_time = time.time()
    deadline = 150.0 # Seconds
    
    if n <= 4:
        # For small n, we can potentially find optimal or near-optimal easily
        # Standard is optimal for n<=4
        return _standard_construct(n)
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute transition costs
    # C[i][j] is cost to go from perms[i] to perms[j]
    # Cost is n - k where k is longest suffix of i that is prefix of j
    C = [[0] * N for _ in range(N)]
    
    # Optimization: Group permutations by prefix/suffix?
    # For n=7, N=5040. N^2 = 25M.
    # Inner loop: check k from n down to 0.
    # 25M * 7 = 175M ops. Might be slow in pure Python.
    # Let's try to speed it up or use a different heuristic if too slow.
    
    # Alternative: Just run the standard construct first as a baseline.
    # Then try to improve.
    
    baseline_word = _standard_construct(n)
    baseline_len = len(baseline_word)
    
    # If n is large, TSP might be too slow to even compute matrix.
    # But n=7 is the target.
    
    # Let's compute C efficiently.
    # Map (perm) to index
    # We can precompute for each perm, its suffixes.
    
    # Actually, let's just try the NN heuristic on the fly or with a precomputed matrix.
    # If precomputing C takes too long, we abort.
    
    # Time check
    if time.time() - start_time > 10.0:
         return baseline_word
         
    # Compute C
    for i in range(N):
        p_i = perms[i]
        for j in range(N):
            p_j = perms[j]
            # Find max k such that p_i[-k:] == p_j[:k]
            # Max possible k is n-1 (if k=n, they are same perm, but we visit each once, so i!=j)
            # Actually if i==j, cost is n (reset), but we don't visit same node twice.
            k = 0
            # Check k from n-1 down to 0
            # p_i is tuple of length n
            # p_j is tuple of length n
            # We want p_i[n-k:n] == p_j[0:k]
            for k in range(n - 1, -1, -1):
                if k == 0:
                    C[i][j] = n
                    break
                if p_i[n-k:] == p_j[:k]:
                    C[i][j] = n - k
                    break
            # This inner loop is O(n). Total O(N^2 n).
            
    if time.time() - start_time > 30.0:
         return baseline_word
         
    # Now find a path
    # Nearest Neighbor
    visited = [False] * N
    path = []
    current = 0
    visited[0] = True
    path.append(0)
    total_cost = 0
    
    for _ in range(N - 1):
        best_next = -1
        best_cost = float('inf')
        for j in range(N):
            if not visited[j]:
                cost = C[current][j]
                if cost < best_cost:
                    best_cost = cost
                    best_next = j
        if best_next == -1:
            break
        visited[best_next] = True
        path.append(best_next)
        total_cost += best_cost
        current = best_next
        if time.time() - start_time > 100.0:
             break
             
    # If we didn't finish the path, we can't use it as a valid superpermutation
    # unless we complete it. But if we broke early, it's invalid.
    # So we must ensure we visit all nodes.
    # NN is O(N^2) per step? No, O(N) per step, total O(N^2).
    # 25M steps. In Python, ~1-2 seconds.
    
    # Try to improve with 2-opt?
    # 2-opt on 5040 nodes might be expensive.
    # Let's just return the NN path if it's better than baseline.
    
    # Reconstruct word
    if len(path) == N:
        # Construct word
        word = list(perms[path[0]])
        for i in range(1, N):
            prev_p = perms[path[i-1]]
            curr_p = perms[path[i]]
            # Append necessary symbols
            # We know cost = C[path[i-1]][path[i]]
            # We need to find the overlap k.
            # Since we computed C, we can re-derive k or store it.
            # To save memory, let's just re-calculate the overlap for the specific pair in the path.
            # Or, simpler: just append the whole curr_p and trim? No, that's not how it works.
            # We need to append the suffix of curr_p that is not covered by the prefix match with prev_p's suffix.
            
            # Let's find k again for this specific pair
            k = 0
            for k in range(n - 1, -1, -1):
                if k == 0:
                    break
                if prev_p[n-k:] == curr_p[:k]:
                    break
            # Append curr_p[k:]
            word.extend(curr_p[k:])
        
        # Validate
        if _is_valid_superpermutation(word, n):
            if len(word) < baseline_len:
                return "".join(map(str, word))
                
    return baseline_word

def _standard_construct(n):
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

def _is_valid_superpermutation(word, n):
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) != n:
            continue
        seen.add(window)
    if len(seen) == n ** 0: # Placeholder
        pass
    # Check if all perms are seen
    # This is expensive for n=7 (5040 perms).
    # But we only do it once.
    all_perms = set(itertools.permutations(range(1, n+1)))
    return all_perms.issubset(seen)
