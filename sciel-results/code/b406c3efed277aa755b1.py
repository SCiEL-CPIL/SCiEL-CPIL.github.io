import time
import itertools
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Base case
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate standard superpermutation for n-1 to get the initial order of permutations
    # We need the order of permutations of 1..n-1 as they appear in S_{n-1}
    # We can just run the standard construction recursively to get the word, then extract order
    # Or we can just generate all permutations and use the standard order?
    # The standard order is NOT arbitrary. It's specific.
    # Let's generate the standard word for n-1 first.
    
    def standard_superperm(k):
        if k == 1:
            return [1]
        word = standard_superperm(k-1)
        perms, seen = [], set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [k] + list(p)
            overlap = 0
            # Max possible overlap is len(block) but limited by len(new)
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new

    # Get the standard word for n-1
    # For n=7, n-1=6. S_6 is 873 long. Fast.
    word_prev = standard_superperm(n-1)
    
    # Extract the order of permutations of 1..n-1
    perms_order = []
    seen = set()
    for i in range(len(word_prev) - n + 2):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_order.append(window)
            
    # perms_order contains all (n-1)! permutations in the standard order.
    # Map them to indices 0..N-1
    N = len(perms_order)
    perm_to_idx = {p: i for i, p in enumerate(perms_order)}
    
    # Precompute overlap matrix
    # M[i][j] = overlap between perms_order[i] (suffix) and perms_order[j] (prefix)
    # Overlap is max k such that perms_order[i][-k:] == perms_order[j][:k]
    if time.time() > deadline - 30:
        # Not enough time, just return the standard construction
        word = list(perms_order[0]) + [n] + list(perms_order[0])
        for idx in range(1, N):
            p = perms_order[idx]
            block = list(p) + [n] + list(p)
            ov = 0
            prev = perms_order[idx-1]
            for k in range(n-1, 0, -1):
                if prev[-k:] == p[:k]:
                    ov = k
                    break
            word += block[ov:]
        return "".join(map(str, word))
    
    M = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        p_i = perms_order[i]
        for j in range(N):
            p_j = perms_order[j]
            # Max overlap is n-1
            max_k = n - 1
            for k in range(max_k, 0, -1):
                if p_i[-k:] == p_j[:k]:
                    M[i][j] = k
                    break
    
    # Initial path is 0, 1, ..., N-1
    path = list(range(N))
    
    # Calculate initial score (sum of overlaps)
    def calc_score(path):
        score = 0
        for i in range(len(path) - 1):
            score += M[path[i]][path[i+1]]
        return score
    
    current_score = calc_score(path)
    
    # Local Search: 2-opt
    # We want to maximize score.
    # A 2-opt move reverses a segment of the path.
    # This changes edges (u, v) and (x, y) to (u, y) and (v, x) if we reverse v..x?
    # Actually, reversing segment [i, j] in path:
    # Path: ... u, v, ..., x, y ...
    # Becomes: ... u, x, ..., v, y ...
    # Edges removed: (u, v) and (x, y)
    # Edges added: (u, x) and (v, y)
    # Delta = M[u][x] + M[v][y] - M[u][v] - M[x][y]
    
    # Local Search: 2-opt with first-improvement strategy
    # Path is a sequence of N nodes. Edges are (path[k], path[k+1]) for k=0..N-2.
    # Reversing segment path[i+1..j] (inclusive) changes:
    #   Removed edges: (path[i], path[i+1]) and (path[j], path[j+1]) if j+1 < N
    #   Added edges: (path[i], path[j]) and (path[i+1], path[j+1]) if j+1 < N
    # For j == N-1: only remove (path[i], path[i+1]), add (path[i], path[N-1])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline:
                break
            u = path[i]
            v = path[i+1]
            for j in range(i + 1, N):
                if time.time() > deadline:
                    break
                x = path[j]
                
                if j == N - 1:
                    # Reversing path[i+1..N-1]: remove edge (u,v), add edge (u,x)
                    delta = M[u][x] - M[u][v]
                else:
                    y = path[j+1]
                    # Reversing path[i+1..j]: remove (u,v) and (x,y), add (u,x) and (v,y)
                    delta = M[u][x] + M[v][y] - M[u][v] - M[x][y]
                
                if delta > 0:
                    # Perform the reversal
                    path[i+1:j+1] = reversed(path[i+1:j+1])
                    current_score += delta
                    improved = True
                    break
            if improved:
                break
        
    # Construct the final word
    # Start with the first block
    first_p = perms_order[path[0]]
    word = list(first_p) + [n] + list(first_p)
    
    for idx in range(1, N):
        p = perms_order[path[idx]]
        block = list(p) + [n] + list(p)
        # Overlap is M[path[idx-1]][path[idx]]
        ov = M[path[idx-1]][path[idx]]
        word += block[ov:]
        
    return "".join(map(str, word))
