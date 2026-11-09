import time
import sys

def _precompute_divisibility():
    """
    Precompute divisibility masks.
    P = lcm(1..12) = 27720.
    div_mask[r][j] is a 12-bit integer where bit k is set if (k+1) divides (r + 1 + j).
    j ranges from 0 to 143 (max L is 144).
    """
    P = 27720
    MAX_L = 144
    div_mask = [[0] * MAX_L for _ in range(P)]
    
    # Optimization: Iterate j first, then r? Or r first?
    # r + 1 + j. 
    # For a fixed j, r+1+j mod i == 0.
    # It's probably faster to just compute directly given the small size.
    
    for r in range(P):
        base = r + 1
        for j in range(MAX_L):
            val = base + j
            mask = 0
            # Check divisibility by 1..12
            # 1 always divides
            mask |= 1 # bit 0 for i=1
            if val % 2 == 0: mask |= (1 << 1)
            if val % 3 == 0: mask |= (1 << 2)
            if val % 4 == 0: mask |= (1 << 3)
            if val % 5 == 0: mask |= (1 << 4)
            if val % 6 == 0: mask |= (1 << 5)
            if val % 7 == 0: mask |= (1 << 6)
            if val % 8 == 0: mask |= (1 << 7)
            if val % 9 == 0: mask |= (1 << 8)
            if val % 10 == 0: mask |= (1 << 9)
            if val % 11 == 0: mask |= (1 << 10)
            if val % 12 == 0: mask |= (1 << 11)
            div_mask[r][j] = mask
    return div_mask

def _has_matching(mask_row, L):
    """
    Check if there is a matching of size 12 in the window of length L.
    mask_row[j] has bit i-1 set if i divides position j+1 (relative to m).
    We need to assign each i in 1..12 to a distinct j in 0..L-1.
    """
    if L < 12:
        return False
    
    # Build adjacency lists for left nodes 1..12
    # adj[i] is a list of j's such that i divides position j+1
    # Using bitmasks for positions might be faster for iteration?
    # Let's use lists of indices for DFS.
    
    adj = [[] for _ in range(12)] # 0-indexed for i=1..12
    for j in range(L):
        mask = mask_row[j]
        for i in range(12):
            if mask & (1 << i):
                adj[i].append(j)
                
    # Heuristic: Sort left nodes by degree (ascending) to prune early
    # We need to match all 12. 
    # We can just iterate 0..11, but sorting helps.
    # However, the "left" nodes are fixed 1..12. 
    # We can permute the order in which we try to match them.
    
    # Standard DFS augmenting path
    # match_to_pos[j] = i matched to j, or -1
    match_to_pos = [-1] * L
    
    # visited array for the current DFS path
    visited = [False] * L
    
    # To speed up, we can try nodes with fewer options first.
    # But the nodes are 1..12. 
    # Let's just use the standard algorithm.
    
    def dfs(i):
        for j in adj[i]:
            if not visited[j]:
                visited[j] = True
                if match_to_pos[j] == -1 or dfs(match_to_pos[j]):
                    match_to_pos[j] = i
                    return True
        return False
    
    count = 0
    for i in range(12):
        # Reset visited for each new root
        for j in range(L):
            visited[j] = False
        if dfs(i):
            count += 1
        else:
            return False
            
    return count == 12

def construct():
    start_time = time.time()
    DEADLINE = 85.0 # Safety margin
    
    div_mask = _precompute_divisibility()
    
    best_m = 0
    best_f = 12 # Minimum possible is at least 12? No, could be less? 
                # Actually, distinct integers b_i. If L=1, only 1 number. 
                # We need 12 distinct numbers. So L must be at least 12.
                # Wait, b_i are distinct integers in (m, m+L]. 
                # There are L integers in the interval. We need to pick 12 distinct ones.
                # So L >= 12 is a hard lower bound.
                
    for r in range(27720):
        if time.time() - start_time > DEADLINE:
            break
            
        # Binary search for minimal L
        lo, hi = 12, 144
        found = False
        while lo <= hi:
            mid = (lo + hi) // 2
            if _has_matching(div_mask[r], mid):
                best_f_candidate = mid
                found = True
                hi = mid - 1
            else:
                lo = mid + 1
                
        if found:
            if best_f_candidate > best_f:
                best_f = best_f_candidate
                best_m = r
                
    return int(best_m)

if __name__ == "__main__":
    # Quick test
    m = construct()
    print(f"Best m: {m}, f: {best_f}")
