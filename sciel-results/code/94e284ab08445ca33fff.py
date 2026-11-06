"""Structural change: Multi-start randomized vertex removal with simulated annealing repair and fast validity checking."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin for 120s limit
    
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    # Build adjacency bitmask: adj[i] bit j = 1 means i -> j
    adj = [0] * p
    for i in range(p):
        mask = 0
        for j in range(p):
            if i == j:
                continue
            diff = (j - i) % p
            if diff in squares:
                mask |= (1 << j)
        adj[i] = mask
    
    def get_in_masks(adj_list):
        n = len(adj_list)
        in_masks = [0] * n
        for i in range(n):
            m = adj_list[i]
            for j in range(n):
                if i != j and (m >> j) & 1:
                    in_masks[j] |= (1 << i)
        return in_masks
    
    def count_violations_fast(in_masks, n):
        """Count 4-sets with no common dominator using bitmask intersections."""
        violations = 0
        # Precompute pairwise intersections to speed up
        pair_int = [[0] * n for _ in range(n)]
        for a in range(n):
            ma = in_masks[a]
            for b in range(a + 1, n):
                pair_int[a][b] = ma & in_masks[b]
        
        for a in range(n):
            for b in range(a + 1, n):
                mab = pair_int[a][b]
                if mab == 0:
                    continue
                for c in range(b + 1, n):
                    mabc = mab & in_masks[c]
                    if mabc == 0:
                        continue
                    for d in range(c + 1, n):
                        if (mabc & in_masks[d]) == 0:
                            violations += 1
        return violations
    
    def check_valid(adj_list):
        n = len(adj_list)
        if n < 5:
            return True, 0
        in_masks = get_in_masks(adj_list)
        v = count_violations_fast(in_masks, n)
        return v == 0, v
    
    def remove_vertex(adj_list, idx):
        """Remove vertex at index idx, return new adj list."""
        n = len(adj_list)
        new_n = n - 1
        new_adj = [0] * new_n
        for i in range(n):
            if i == idx:
                continue
            new_i = i if i < idx else i - 1
            mask = 0
            for j in range(n):
                if j == idx:
                    continue
                if (adj_list[i] >> j) & 1:
                    new_j = j if j < idx else j - 1
                    mask |= (1 << new_j)
            new_adj[new_i] = mask
        return new_adj
    
    def flip_edge(adj_list, u, v):
        """Flip edge between u and v in place."""
        if (adj_list[u] >> v) & 1:
            adj_list[u] &= ~(1 << v)
            adj_list[v] |= (1 << u)
        else:
            adj_list[u] |= (1 << v)
            adj_list[v] &= ~(1 << u)
    
    def sa_repair(adj_list, max_iter=300, t0=1.0, t1=0.01):
        """Simulated annealing to reduce violations."""
        n = len(adj_list)
        if n < 5:
            return adj_list, 0
        
        in_masks = get_in_masks(adj_list)
        cur_viol = count_violations_fast(in_masks, n)
        if cur_viol == 0:
            return adj_list, 0
        
        best_adj = adj_list[:]
        best_viol = cur_viol
        cur_adj = adj_list[:]
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            if cur_viol == 0:
                return cur_adj, 0
            
            temp = t0 * (t1 / t0) ** (it / max_iter)
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            flip_edge(cur_adj, u, v)
            
            new_in_masks = get_in_masks(cur_adj)
            new_viol = count_violations_fast(new_in_masks, n)
            
            delta = new_viol - cur_viol
            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                cur_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_adj = cur_adj[:]
            else:
                flip_edge(cur_adj, u, v)  # revert
            
            if cur_viol == 0:
                break
        
        return best_adj, best_viol
    
    def try_remove(adj_list, idx):
        """Try removing vertex idx, with SA repair if needed."""
        if len(adj_list) <= 5:
            return adj_list, False
        
        new_adj = remove_vertex(adj_list, idx)
        is_valid, viol = check_valid(new_adj)
        if is_valid:
            return new_adj, True
        
        if viol > 3:
            return new_adj, False
        
        repaired, final_viol = sa_repair(new_adj, max_iter=150, t0=0.5, t1=0.05)
        if final_viol == 0:
            return repaired, True
        return new_adj, False
    
    # Verify base Paley 67 is valid
    is_valid, _ = check_valid(adj)
    if not is_valid:
        # Fallback: return as-is (shouldn't happen)
        n = p
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': n, 'orientation': orientation}
    
    best_n = p
    best_adj = adj[:]
    
    # Multi-start: try several random removal orders
    random.seed(42)
    num_starts = 8
    
    for s in range(num_starts):
        if time.time() - start_time > deadline - 10:
            break
        
        cur_adj = adj[:]
        cur_n = p
        removed = [False] * p
        
        # Random order of vertices to try removing
        order = list(range(p))
        random.shuffle(order)
        
        for v in order:
            if time.time() - start_time > deadline - 5:
                break
            if cur_n <= 5:
                break
            if removed[v]:
                continue
            
            # Find current index of v in cur_adj
            # We track which original vertices are still present
            # cur_adj is reindexed, so we need to map
            # Simpler: just try removing index 0, 1, etc. in random order
            pass
        
        # Simpler approach: sequential random removal
        cur_adj = adj[:]
        cur_n = p
        while cur_n > 5:
            if time.time() - start_time > deadline - 5:
                break
            
            # Pick a random vertex to try removing
            idx = random.randint(0, cur_n - 1)
            new_adj, success = try_remove(cur_adj, idx)
            
            if success:
                cur_adj = new_adj
                cur_n = len(cur_adj)
                if cur_n < best_n:
                    best_n = cur_n
                    best_adj = cur_adj[:]
                    # Try to continue removing
            else:
                # If this random pick failed, try a few more
                found = False
                for _ in range(3):
                    if time.time() - start_time > deadline - 5:
                        break
                    idx2 = random.randint(0, cur_n - 1)
                    new_adj2, success2 = try_remove(cur_adj, idx2)
                    if success2:
                        cur_adj = new_adj2
                        cur_n = len(cur_adj)
                        if cur_n < best_n:
                            best_n = cur_n
                            best_adj = cur_adj[:]
                        found = True
                        break
                if not found:
                    break
    
    # Final verification
    is_valid, _ = check_valid(best_adj)
    if not is_valid:
        best_adj = adj[:]
        best_n = p
    
    n = best_n
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                orientation.append(0)
            else:
                orientation.append(1)
    
    return {'n': n, 'orientation': orientation}
# EVOLVE-BLOCK-END
