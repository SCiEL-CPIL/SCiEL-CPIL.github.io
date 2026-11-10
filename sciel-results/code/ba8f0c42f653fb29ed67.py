"""Structural change: Model superpermutation as TSP on permutation overlap graph."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(n):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    if n <= 1:
        return "1"
    
    if n < 7:
        return _standard_superperm(n)

    # For n=7, use Nearest Neighbor heuristic on the permutation overlap graph
    perms_list = list(itertools.permutations(range(1, n + 1)))
    N = len(perms_list)
    
    # Nearest Neighbor
    visited = [False] * N
    path = [0]
    visited[0] = True
    
    # Precompute suffixes for speed? 
    # For each perm, precompute suffixes of length 1..n-1?
    # Or just calculate on the fly.
    
    for _ in range(N - 1):
        curr = path[-1]
        curr_perm = perms_list[curr]
        
        best_idx = -1
        best_cost = n + 1
        
        # Optimize: Iterate through unvisited
        for j in range(N):
            if not visited[j]:
                target_perm = perms_list[j]
                # Calculate overlap
                # curr_perm suffix vs target_perm prefix
                # Max possible overlap is n-1
                for k in range(n - 1, -1, -1):
                    if k == 0 or curr_perm[-k:] == target_perm[:k]:
                        c = n - k
                        if c < best_cost:
                            best_cost = c
                            best_idx = j
                        break
                
        if best_idx == -1:
            # Fallback if something went wrong
            for j in range(N):
                if not visited[j]:
                    best_idx = j
                    break
            
        if best_idx != -1:
            path.append(best_idx)
            visited[best_idx] = True
            
        if time.time() > deadline - 5:
            # If we run out of time, break and return what we have?
            # But we need a valid superpermutation.
            # If we break early, we have a partial path.
            # We must append the rest in arbitrary order to ensure validity.
            if len(path) < N:
                for j in range(N):
                    if not visited[j]:
                        path.append(j)
                        visited[j] = True
            break

    # Reconstruct string
    if not path:
        return _standard_superperm(n)
        
    word = list(perms_list[path[0]])
    for i in range(1, len(path)):
        prev_perm = perms_list[path[i-1]]
        curr_perm = perms_list[path[i]]
        # Find overlap
        for k in range(n - 1, -1, -1):
            if k == 0 or prev_perm[-k:] == curr_perm[:k]:
                word.extend(curr_perm[k:])
                break
    
    return "".join(map(str, word))

def _standard_superperm(n):
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
