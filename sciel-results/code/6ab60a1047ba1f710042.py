import time
import sys

# Global cache
_CACHE = {}

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    MOD = 101
    N_DENOMS = 10
    
    # Precompute subset info if not cached
    if 'subset_info' not in _CACHE:
        invs = [0] * (N_DENOMS + 1)
        for x in range(1, N_DENOMS + 1):
            invs[x] = pow(x, MOD - 2, MOD)
        
        subset_info = []
        # mask 1 to (1<<10)-1
        for mask in range(1, 1 << N_DENOMS):
            s = 0
            size = 0
            denoms = []
            for x in range(1, N_DENOMS + 1):
                if mask & (1 << (x - 1)):
                    denoms.append(x)
                    s = (s + invs[x]) % MOD
                    size += 1
            subset_info.append((s, size, denoms))
        
        _CACHE['subset_info'] = subset_info
        _CACHE['n_subsets'] = len(subset_info)
        
        # Precompute adjacency lists for each residue, sorted by size
        # adj_by_res[r] = list of (size, subset_index)
        adj_by_res = [[] for _ in range(MOD)]
        for i, (res, sz, denoms) in enumerate(subset_info):
            adj_by_res[res].append((sz, i))
        
        # Sort by size to prioritize smaller subsets if needed, 
        # though for matching existence it doesn't strictly matter.
        for r in range(MOD):
            adj_by_res[r].sort(key=lambda x: x[0])
            
        _CACHE['adj_by_res'] = adj_by_res
        
    subset_info = _CACHE['subset_info']
    n_subsets = _CACHE['n_subsets']
    adj_by_res = _CACHE['adj_by_res']
    n_residues = MOD

    # Kuhn's Algorithm (DFS-based bipartite matching)
    # Left: Residues 0..100
    # Right: Subsets 0..1022
    
    def find_matching(max_k):
        """
        Finds a perfect matching using only subsets of size <= max_k.
        Returns list of subset indices for each residue, or None if no perfect matching.
        """
        # Build adjacency list of subset indices only
        adj = [[] for _ in range(n_residues)]
        for r in range(n_residues):
            for sz, idx in adj_by_res[r]:
                if sz <= max_k:
                    adj[r].append(idx)
                else:
                    break # Since sorted by size
            
        # Check if any residue has no options
        for r in range(n_residues):
            if not adj[r]:
                return None

        match_r = [-1] * n_subsets
        match_l = [-1] * n_residues
        
        # Iterative DFS to avoid recursion limits, though depth is small
        # We'll use a standard recursive DFS first, as depth <= 101 is safe.
        sys.setrecursionlimit(20000)
        
        def dfs(u):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_r[v] == -1 or dfs(match_r[v]):
                    match_r[v] = u
                    match_l[u] = v
                    return True
            return False

        matching_size = 0
        for u in range(n_residues):
            if time.time() - start_time > DEADLINE - 5:
                return None
            visited = [False] * n_subsets
            if dfs(u):
                matching_size += 1
                
        if matching_size == n_residues:
            return match_l
        return None

    # Try increasing max_k
    # We know k=1,2 are impossible. Start from 3.
    # But for safety, start from 1.
    for max_k in range(1, N_DENOMS + 1):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        match_l = find_matching(max_k)
        if match_l:
            result = []
            for r in range(n_residues):
                subset_idx = match_l[r]
                result.append(subset_info[subset_idx][2])
            return result
    
    # Fallback: Greedy (should not happen if matching exists for K<=10)
    # But if time runs out, return best effort
    if time.time() - start_time > DEADLINE - 5:
        result = []
        used_subsets = set()
        # Try to assign smallest subsets
        # Sort residues by number of available subsets (most constrained first)
        # For greedy, just pick first available smallest
        for r in range(n_residues):
            best_si = -1
            best_size = N_DENOMS + 1
            for sz, idx in adj_by_res[r]:
                if idx not in used_subsets and sz < best_size:
                    best_size = sz
                    best_si = idx
            if best_si != -1:
                result.append(subset_info[best_si][2])
                used_subsets.add(best_si)
            else:
                # Last resort: any unused
                found = False
                for sz, idx in adj_by_res[r]:
                    if idx not in used_subsets:
                        result.append(subset_info[idx][2])
                        used_subsets.add(idx)
                        found = True
                        break
                if not found:
                    # Should not happen
                    result.append([1])
        return result

    return []
