"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    n = 12
    period = 27720  # lcm(1,2,...,12)
    max_L = 144
    
    # Precompute: for each offset k (1..max_L), and for each position r (0..period-1),
    # the number x = (r + k) % period + 1 (in [1, period])
    # For each such x, precompute which indices i in 1..12 divide x
    # Store as a bitmask: bit i set means i divides x
    
    # Precompute div_mask[x-1] for x in 1..period
    # div_mask[x-1] = bitmask where bit (i-1) is set if i divides x
    div_mask = [0] * period
    for i in range(1, n + 1):
        for x in range(i, period + 1, i):
            div_mask[x - 1] |= (1 << (i - 1))
    
    # For a given r and L, the numbers in (r, r+L] mod period are:
    # For k=1..L: x_k = (r + k) % period, mapped to 1..period
    # x_k = ((r + k - 1) % period) + 1
    
    # Precompute for each r, the sequence of div_masks for offsets 1..max_L
    # mask_seq[r][k] = div_mask for the number at offset k from r
    # This is a 27720 x 144 array of ints - about 4MB, fine
    
    # Actually, let's compute on the fly but cache per r
    # For each r, build the list of masks incrementally
    
    def solve_for_r(r):
        """Find f(12, r) = smallest L such that matching exists."""
        # Build candidate masks incrementally
        # masks[k] = div_mask for number at offset k+1 from r
        # i.e., number = ((r + k) % period) + 1, so index = (r + k) % period
        
        # For efficiency, precompute the mask sequence for this r
        # masks[k] for k=0..max_L-1 corresponds to offsets 1..max_L
        
        # We'll use a greedy + backtracking approach with bitmasks
        # State: which right positions are used (bitmask over L positions)
        # But L can be up to 144, so we can't use a single bitmask for positions
        
        # Instead, use DFS with the left side (12 nodes) and track used positions
        # For each left node i, candidates are positions k where masks[k] has bit i set
        
        # Optimization: process left nodes in order of fewest candidates first
        
        # Let's try a different approach: since n=12 is small, use recursive backtracking
        # with memoization on (node_idx, used_mask) but used_mask is over L positions...
        
        # Better: use the standard approach but with bitmask for the 12 left nodes
        # and track which positions are used via a separate structure
        
        # Actually, the fastest approach for this size:
        # For each L from 1 to max_L:
        #   Build adj: for each i in 0..11, list of positions k where mask[k] has bit i
        #   Check perfect matching via DFS
        
        # Precompute the mask sequence once for this r
        mask_seq = [0] * max_L
        for k in range(max_L):
            idx = (r + k) % period  # 0-indexed, corresponds to number idx+1
            mask_seq[k] = div_mask[idx]
        
        # Now for each L, check if matching exists
        # Use DFS with bitmasks for efficiency
        
        # For a given L, adj[i] = [k for k in range(L) if mask_seq[k] & (1 << i)]
        # To speed up, precompute for each i the full list of k where mask has bit i
        # Then for a given L, adj[i] = [k for k in full_list[i] if k < L]
        
        full_lists = [[] for _ in range(n)]
        for k in range(max_L):
            m = mask_seq[k]
            for i in range(n):
                if m & (1 << i):
                    full_lists[i].append(k)
        
        # For each L, check matching
        # Use DFS: match left node i to some position k
        # match_pos[k] = which left node is matched to position k
        
        for L in range(1, max_L + 1):
            # Quick check: if any i has no candidates in [0, L), fail
            valid = True
            for i in range(n):
                # Count candidates for i in range [0, L)
                # full_lists[i] is sorted, so count how many are < L
                cnt = 0
                for k in full_lists[i]:
                    if k < L:
                        cnt += 1
                    else:
                        break
                if cnt == 0:
                    valid = False
                    break
            if not valid:
                continue
            
            # DFS matching
            # matchR[k] = left node matched to position k, or -1
            matchR = [-1] * L
            
            # Order left nodes by scarcity (fewest candidates first)
            order = list(range(n))
            # For this L, compute candidate counts
            cand_counts = []
            for i in range(n):
                cnt = 0
                for k in full_lists[i]:
                    if k < L:
                        cnt += 1
                    else:
                        break
                cand_counts.append(cnt)
            order.sort(key=lambda i: cand_counts[i])
            
            # DFS: assign left nodes in order
            # state: which positions are used
            used = [False] * L
            
            def dfs(idx_in_order):
                if idx_in_order == n:
                    return True
                i = order[idx_in_order]
                for k in full_lists[i]:
                    if k >= L:
                        break
                    if not used[k]:
                        used[k] = True
                        matchR[k] = i
                        if dfs(idx_in_order + 1):
                            return True
                        used[k] = False
                        matchR[k] = -1
                return False
            
            if dfs(0):
                return L
        
        return max_L + 1  # Should not happen
    
    best_L = 0
    best_r = 0
    
    for r in range(period):
        if time.time() > deadline - 2:
            break
        
        L = solve_for_r(r)
        if L <= max_L and L > best_L:
            best_L = L
            best_r = r
            if best_L >= max_L:
                break
    
    # Find m in [0, 1000000] with m % period == best_r
    # m = best_r + k * period
    if best_r <= 1000000:
        m = best_r
    else:
        m = best_r % (1000000 // period + 1)
    
    return int(m)
# EVOLVE-BLOCK-END
