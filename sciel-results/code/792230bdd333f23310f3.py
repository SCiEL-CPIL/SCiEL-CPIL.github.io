import time
import random
import itertools

def compute_overlap_matrix(perms):
    """
    Compute overlap matrix for a list of permutations.
    overlap[i][j] is the maximum k such that suffix of perms[i] of length k 
    equals prefix of perms[j] of length k.
    """
    n = len(perms)
    overlap = [[0] * n for _ in range(n)]
    for i in range(n):
        pi = perms[i]
        for j in range(n):
            pj = perms[j]
            # Max possible overlap is len(pi)
            max_k = len(pi)
            best = 0
            for k in range(1, max_k + 1):
                if pi[-k:] == pj[:k]:
                    best = k
            overlap[i][j] = best
    return overlap

def greedy_path(perms, overlap_matrix, n_restarts=10):
    """
    Find a good ordering of perms to maximize total overlap.
    Returns the list of indices of the optimal path found.
    """
    n = len(perms)
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    best_path = None
    best_score = -1
    
    for r in range(n_restarts):
        # Random start
        start = random.randint(0, n - 1)
        path = [start]
        visited = [False] * n
        visited[start] = True
        current = start
        current_score = 0
        
        for _ in range(n - 1):
            # Find next node with max overlap
            best_next = -1
            best_ov = -1
            # Optimization: if n is large, this O(N) scan is fine for N=720
            for j in range(n):
                if not visited[j]:
                    ov = overlap_matrix[current][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = j
            
            if best_next == -1:
                # Should not happen if graph is connected, but fallback to any unvisited
                for j in range(n):
                    if not visited[j]:
                        best_next = j
                        break
            
            visited[best_next] = True
            path.append(best_next)
            current_score += best_ov
            current = best_next
        
        if current_score > best_score:
            best_score = current_score
            best_path = path
            
    return best_path

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract distinct (m-1) permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # If few permutations, use standard order (or greedy is fast enough)
        # For m >= 5, use optimized greedy
        if len(perms) > 10:
            # Check time
            if time.time() > deadline:
                break
            
            # Compute overlap matrix
            # This can be expensive for large m, but m=7 is 720 perms
            overlap_matrix = compute_overlap_matrix(perms)
            
            # Determine restarts based on size
            # N=720 -> 10 restarts
            # N=120 -> 20 restarts
            restarts = 5 if len(perms) > 500 else 20
            
            path = greedy_path(perms, overlap_matrix, n_restarts=restarts)
            ordered_perms = [perms[i] for i in path]
        else:
            # For small m, standard order is likely fine, but let's just use the list
            # Actually, even for small m, we can optimize
            ordered_perms = perms

        new = []
        for idx, p in enumerate(ordered_perms):
            block = list(p) + [m] + list(p)
            if idx == 0:
                new += block
            else:
                # Find overlap
                # new ends with previous p
                # block starts with current p
                # We can use the overlap matrix if we kept it, 
                # but for simplicity and consistency with the "model", 
                # we re-calculate or use the stored matrix if available.
                # For efficiency, let's just do the slice comparison.
                # However, doing this for 720 blocks * 6 length is fast.
                max_ov = min(len(new), len(block))
                ov = 0
                # Optimization: check from max possible down to 1
                for k in range(max_ov, 0, -1):
                    if new[-k:] == block[:k]:
                        ov = k
                        break
                new += block[ov:]
        
        word = new
        
        if time.time() > deadline:
            break

    return "".join(map(str, word))
