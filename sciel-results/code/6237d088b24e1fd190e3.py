"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

N = 12
MAX_M = 1000000
MAX_L = 144
P = 27720  # lcm(1..12)

def construct():
    deadline = time.time() + 85  # 85s safety margin for 90s limit
    
    # Precompute masks for one period
    # mask[d] for d in 1..P, where bit i-1 is set if i divides d
    masks = [0] * (P + 1)
    for d in range(1, P + 1):
        m = 0
        for i in range(1, N + 1):
            if d % i == 0:
                m |= (1 << (i - 1))
        masks[d] = m

    best_r = 0
    best_f = 0

    # Function to check if a perfect matching exists for a given window (r, r+L]
    # We check if there is a bijection between indices 1..N and distinct numbers in (r, r+L]
    # such that i divides the assigned number.
    def check_matching(r, L):
        # Get masks for x = r+1, ..., r+L
        # Note: masks are 1-indexed in our precomputation, but r can be 0.
        # If r=0, x are 1..L. If r>0, x are r+1..r+L.
        # The mask for x is masks[x % P] if x%P != 0 else masks[P].
        # Actually, simpler: just compute on the fly or use circular logic.
        # Since N is small, we can just build the adjacency list.
        
        # adj[i] = list of indices j (0..L-1) such that (i+1) divides (r+1+j)
        adj = [[] for _ in range(N)]
        for j in range(L):
            x = r + 1 + j
            # Calculate mask for x
            # x mod P
            xm = x % P
            if xm == 0:
                mx = masks[P]
            else:
                mx = masks[xm]
            
            for i in range(N):
                if mx & (1 << i):
                    adj[i].append(j)
        
        # Hopcroft-Karp or simple DFS matching
        # Since N=12 and L<=144, simple DFS is fast enough.
        match_to_num = [-1] * L  # match_to_num[j] = i if number j is matched to index i
        match_to_idx = [-1] * N  # match_to_idx[i] = j if index i is matched to number j
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to_num[j] == -1 or dfs(match_to_num[j], visited):
                    match_to_num[j] = i
                    match_to_idx[i] = j
                    return True
            return False
        
        matching_size = 0
        for i in range(N):
            visited = [False] * L
            if dfs(i, visited):
                matching_size += 1
        
        return matching_size == N

    # Iterate over all residues
    for r in range(P):
        if time.time() > deadline:
            break
            
        # Binary search or linear search for max L
        # Max possible L is 144.
        # We want the LEAST L such that matching exists.
        # Wait, the task asks to MAXIMIZE f(12, m).
        # f(12, m) is the LEAST L such that matching exists.
        # So we want to find m such that this "least L" is as large as possible.
        
        # Let's find the least L for this r.
        current_f = 0
        for L in range(1, MAX_L + 1):
            if check_matching(r, L):
                current_f = L
                break
        
        if current_f > best_f:
            best_f = current_f
            best_r = r

    # The best m is best_r (since best_r < P <= MAX_M)
    return int(best_r)

# EVOLVE-BLOCK-END
