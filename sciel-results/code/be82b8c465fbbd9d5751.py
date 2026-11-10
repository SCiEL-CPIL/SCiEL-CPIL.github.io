import time
import random
from collections import deque

def construct():
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 3.0
    
    P = 101
    N = 10
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, -1, P)
    
    # Precompute all subsets
    num_subsets = (1 << N) - 1
    subset_sum = [0] * (1 << N)
    subset_len = [0] * (1 << N)
    subset_denoms = [None] * (1 << N)
    
    for mask in range(1, 1 << N):
        length = bin(mask).count('1')
        s = 0
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                d = i + 1
                denoms.append(d)
                s = (s + inv[d]) % P
        subset_sum[mask] = s
        subset_len[mask] = length
        subset_denoms[mask] = denoms
    
    # For each residue, list of subset masks that sum to it, sorted by length
    masks_by_residue = [[] for _ in range(P)]
    for mask in range(1, 1 << N):
        masks_by_residue[subset_sum[mask]].append(mask)
    for r in range(P):
        masks_by_residue[r].sort(key=lambda m: subset_len[m])
    
    def find_matching_k(k, max_iters=200):
        """Find perfect matching using only subsets of size <= k."""
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        adj = [[] for _ in range(P)]
        for r in range(P):
            for mask in masks_by_residue[r]:
                if subset_len[mask] <= k:
                    adj[r].append(mask - 1)
        
        # Check if perfect matching is possible (necessary condition)
        for r in range(P):
            if not adj[r]:
                return None
        
        pair_u = [-1] * P
        pair_v = [-1] * num_subsets
        dist = [0] * P
        work = [0] * P
        
        def bfs():
            queue = deque()
            for u in range(P):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == float('inf'):
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found
        
        def dfs(u):
            if work[u] >= len(adj[u]):
                return False
            for i in range(work[u], len(adj[u])):
                work[u] = i
                v = adj[u][i]
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = 0
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                return None
            if not bfs():
                break
            work = [0] * P
            for u in range(P):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        if matching != P:
            return None
        
        assignment = [None] * P
        for u in range(P):
            v = pair_u[u]
            mask = v + 1
            assignment[u] = subset_denoms[mask]
        
        return assignment
    
    # Try k=1,2,...,N and find the smallest k with a perfect matching
    best_assignment = None
    best_k = N + 1
    for k in range(1, N + 1):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        result = find_matching_k(k)
        if result is not None:
            best_assignment = result
            best_k = k
            break
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: local search to find any valid assignment
    # Start with a random assignment
    random.seed(42)
    assignment = [None] * P
    for r in range(P):
        if masks_by_residue[r]:
            # Pick a random subset for this residue
            mask = random.choice(masks_by_residue[r])
            assignment[r] = subset_denoms[mask]
        else:
            assignment[r] = [1]
    
    # Try to improve via local search
    max_local_iters = 500
    for it in range(max_local_iters):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Check if current assignment is valid
        valid = True
        for r in range(P):
            if assignment[r] is None:
                valid = False
                break
            s = 0
            seen = set()
            for d in assignment[r]:
                if d in seen:
                    valid = False
                    break
                seen.add(d)
                s = (s + inv[d]) % P
            if s != r:
                valid = False
                break
        if valid:
            return assignment
        
        # Find a residue to fix
        fix_r = None
        for r in range(P):
            if assignment[r] is None:
                fix_r = r
                break
            s = 0
            seen = set()
            for d in assignment[r]:
                if d in seen:
                    fix_r = r
                    break
                seen.add(d)
                s = (s + inv[d]) % P
            if s != r:
                fix_r = r
                break
        
        if fix_r is None:
            break
        
        # Try all possible subsets for this residue
        found_fix = False
        for mask in masks_by_residue[fix_r]:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            new_denoms = subset_denoms[mask]
            old_denoms = assignment[fix_r]
            assignment[fix_r] = new_denoms
            
            # Check if still valid
            s = 0
            seen = set()
            valid_here = True
            for d in new_denoms:
                if d in seen:
                    valid_here = False
                    break
                seen.add(d)
                s = (s + inv[d]) % P
            if valid_here and s == fix_r:
                found_fix = True
                break
            else:
                assignment[fix_r] = old_denoms
        
        if not found_fix:
            # Try a random restart
            random.seed(random.randint(0, 1000000))
            for r in range(P):
                if masks_by_residue[r]:
                    mask = random.choice(masks_by_residue[r])
                    assignment[r] = subset_denoms[mask]
    
    # Final fallback: greedy
    best_assignment = [None] * P
    for r in range(P):
        if masks_by_residue[r]:
            best_mask = masks_by_residue[r][0]
            best_assignment[r] = subset_denoms[best_mask]
        else:
            best_assignment[r] = [1]
    
    return best_assignment
