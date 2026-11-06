import time
import sys
import random

N = 9
FULL_MASK = (1 << (1 << N)) - 1
TOTAL_VERTICES = 1 << N

# Precompute neighbor masks for all 512 vertices
# neighbor_mask[v] is a 512-bit integer with bits set for all neighbors of v
_NEIGHBOR_MASKS = [0] * TOTAL_VERTICES
for v in range(TOTAL_VERTICES):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASKS[v] = m

# Precompute popcount for 16-bit chunks for faster bit_count if needed, 
# but Python 3.10+ int.bit_count() is very fast. We assume Python 3.10+.

def _greedy_extend(path, used_mask, blocked_mask, direction=1, max_steps=1000):
    """
    Greedily extend the path from one end using Warnsdorff.
    direction=1: add to end (path[-1])
    direction=-1: add to front (path[0])
    Returns the extended path.
    """
    extended = list(path)
    ext_used = used_mask
    ext_blocked = blocked_mask
    
    steps = 0
    while steps < max_steps:
        if direction == 1:
            cur = extended[-1]
        else:
            cur = extended[0]
            
        # Valid next vertices: neighbors of cur, not used, not blocked
        valid_mask = _NEIGHBOR_MASKS[cur] & ~ext_used & ~ext_blocked
        
        if not valid_mask:
            break
            
        # Find best candidate using Warnsdorff: fewest exits
        best_v = -1
        best_score = 100
        vm = valid_mask
        while vm:
            lsb = vm & -vm
            v = lsb.bit_length() - 1
            vm ^= lsb
            
            # Calculate exits from v
            # If we go to v, we mark v as used and cur's neighbors as blocked (except v itself? No, blocked is neighbors of all internal nodes)
            # Actually, the blocked mask in the main DFS logic is: neighbors of all vertices in path[1:-1].
            # When extending, the new 'cur' becomes an internal node (unless it's the new end).
            # So if we move from cur to v:
            # - v becomes the new end.
            # - cur becomes an internal node (if direction=1, cur was end, now internal).
            # - So we add _NEIGHBOR_MASKS[cur] to blocked.
            # - We add v to used.
            
            new_used = ext_used | (1 << v)
            new_blocked = ext_blocked | _NEIGHBOR_MASKS[cur]
            
            exits = (_NEIGHBOR_MASKS[v] & ~new_used & ~new_blocked).bit_count()
            
            if exits < best_score:
                best_score = exits
                best_v = v
                
        if best_v < 0:
            break
            
        if direction == 1:
            extended.append(best_v)
        else:
            extended.insert(0, best_v)
            
        ext_used |= 1 << best_v
        ext_blocked |= _NEIGHBOR_MASKS[cur]
        steps += 1
        
    return extended

def _count_exits(v, used_mask, blocked_mask, cur):
    """
    Calculate the number of valid exits from v, assuming we move from cur to v.
    """
    new_used = used_mask | (1 << v)
    # cur becomes internal, so its neighbors are blocked
    new_blocked = blocked_mask | _NEIGHBOR_MASKS[cur]
    valid_exits = _NEIGHBOR_MASKS[v] & ~new_used & ~new_blocked
    return valid_exits.bit_count()

def construct():
    start_time = time.time()
    # Leave 5 seconds buffer for finalization and return
    deadline = start_time + 115.0
    
    sys.setrecursionlimit(10000)
    
    best_path = [0]
    best_len = 1
    
    # 1. Initial Greedy Path to establish a lower bound
    # Start at 0, go to 1
    init_path = [0, 1]
    init_used = (1 << 0) | (1 << 1)
    init_blocked = 0 # No internal nodes yet
    
    # Greedily extend forward
    greedy_path = _greedy_extend(init_path, init_used, init_blocked, direction=1, max_steps=TOTAL_VERTICES)
    if len(greedy_path) > best_len:
        best_path = greedy_path
        best_len = len(best_path)
        
    # Reconstruct masks for best_path for potential extension later
    def get_masks(path):
        used = 0
        for v in path:
            used |= 1 << v
        blocked = 0
        for i in range(1, len(path) - 1):
            blocked |= _NEIGHBOR_MASKS[path[i]]
        return used, blocked

    # 2. Iterative Deepening DFS
    # Symmetry breaking: Start at 0, first move to 1 (bit 0 flip)
    # This is valid because any snake can be shifted and reflected to start at 0->1.
    start_v = 0
    first_v = 1
    
    # Initial state
    path = [start_v, first_v]
    used_mask = (1 << start_v) | (1 << first_v)
    # No internal nodes yet, so blocked is 0
    blocked_mask = 0
    
    # We will search for paths of length L, increasing L.
    # Max possible length is 512.
    # We don't know the exact max, so we iterate.
    # However, IDDFS restarts from scratch. To avoid restarting from 0 every time, 
    # we can use a "deepening" approach where we keep the best found path and try to extend it, 
    # but standard IDDFS is simpler and more robust for finding the global max.
    # Given the time limit, we will perform a DFS with a depth limit, and increase the limit.
    
    # Let's do a single deep DFS with Warnsdorff, but with backtracking.
    # The current seed does DFS but might get stuck.
    # A better approach: 
    # Run a DFS that tries to reach a target length.
    # If it fails, backtracking handles it.
    # We will use a global best_path and update it whenever we find a longer path.
    
    def dfs(path, used_mask, blocked_mask, depth):
        nonlocal best_path, best_len, deadline
        
        if time.time() > deadline:
            return False # Timeout
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
            
        cur = path[-1]
        
        # Generate candidates
        valid_mask = _NEIGHBOR_MASKS[cur] & ~used_mask & ~blocked_mask
        
        if not valid_mask:
            return True # Dead end, but valid path
        
        # Heuristic: Sort candidates by Warnsdorff score (fewest exits first)
        # To avoid sorting overhead every time, we can just iterate and pick the best, 
        # but for backtracking, we need to explore all.
        # Sorting 9 candidates is cheap.
        candidates = []
        vm = valid_mask
        while vm:
            lsb = vm & -vm
            v = lsb.bit_length() - 1
            vm ^= lsb
            score = _count_exits(v, used_mask, blocked_mask, cur)
            candidates.append((score, v))
            
        # Sort by score
        candidates.sort(key=lambda x: x[0])
        
        # Randomize ties to explore different branches
        # Add small noise to scores
        for i in range(len(candidates)):
            if candidates[i][0] == candidates[i-1][0] if i > 0 else False:
                 pass # Simplified: just add noise to all for diversity in restarts?
                 # Actually, deterministic Warnsdorff is strong. 
                 # We rely on backtracking to handle bad choices.
        
        for score, v in candidates:
            if time.time() > deadline:
                return False
            
            # Move to v
            path.append(v)
            new_used = used_mask | (1 << v)
            new_blocked = blocked_mask | _NEIGHBOR_MASKS[cur]
            
            # Recurse
            if not dfs(path, new_used, new_blocked, depth + 1):
                # Timeout
                path.pop()
                return False
                
            path.pop()
            
        return True
    
    # We will run multiple restarts with different randomizations if the first one doesn't find a very long path
    # But IDDFS/Deep DFS with backtracking should find a good one.
    # Let's run the DFS once. If it times out, we return best_path.
    # To ensure we explore different parts of the space, we can shuffle the candidate order slightly in ties.
    
    # Modification to dfs for diversity:
    # We'll pass a random seed or just use random.shuffle on ties.
    
    def dfs_diverse(path, used_mask, blocked_mask, depth, rng):
        nonlocal best_path, best_len, deadline
        
        if time.time() > deadline:
            return False
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
            
        cur = path[-1]
        
        valid_mask = _NEIGHBOR_MASKS[cur] & ~used_mask & ~blocked_mask
        
        if not valid_mask:
            return True
        
        candidates = []
        vm = valid_mask
        while vm:
            lsb = vm & -vm
            v = lsb.bit_length() - 1
            vm ^= lsb
            score = _count_exits(v, used_mask, blocked_mask, cur)
            candidates.append((score, v))
            
        # Group by score
        candidates.sort(key=lambda x: x[0])
        
        # Shuffle candidates with the same score
        i = 0
        while i < len(candidates):
            j = i
            while j < len(candidates) and candidates[j][0] == candidates[i][0]:
                j += 1
            if j > i + 1:
                rng.shuffle(candidates[i:j])
            i = j
            
        for score, v in candidates:
            if time.time() > deadline:
                return False
            
            path.append(v)
            new_used = used_mask | (1 << v)
            new_blocked = blocked_mask | _NEIGHBOR_MASKS[cur]
            
            if not dfs_diverse(path, new_used, new_blocked, depth + 1, rng):
                path.pop()
                return False
                
            path.pop()
            
        return True

    # Run 1-2 DFS attempts
    # Attempt 1: Deterministic-ish (rng fixed)
    rng1 = random.Random(42)
    dfs_diverse(path, used_mask, blocked_mask, 2, rng1)
    
    # If we have time left, try another with different randomization
    if time.time() < deadline - 10:
        rng2 = random.Random(123)
        # Reset path to start
        path = [start_v, first_v]
        used_mask = (1 << start_v) | (1 << first_v)
        blocked_mask = 0
        dfs_diverse(path, used_mask, blocked_mask, 2, rng2)
        
    # Final Polish: Try to extend the best_path from both ends
    if best_len > 1:
        ext_used, ext_blocked = get_masks(best_path)
        
        # Extend forward
        ext_fwd = _greedy_extend(best_path, ext_used, ext_blocked, direction=1, max_steps=TOTAL_VERTICES)
        if len(ext_fwd) > best_len:
            best_path = ext_fwd
            best_len = len(best_path)
            
        # Re-calc masks if path changed
        if len(best_path) > best_len: # This check is redundant if we update best_len, but safe
             pass
            
        ext_used, ext_blocked = get_masks(best_path)
        
        # Extend backward
        ext_bwd = _greedy_extend(best_path, ext_used, ext_blocked, direction=-1, max_steps=TOTAL_VERTICES)
        if len(ext_bwd) > best_len:
            best_path = ext_bwd
            best_len = len(best_path)
            
        # Try forward again after backward?
        ext_used, ext_blocked = get_masks(best_path)
        ext_fwd2 = _greedy_extend(best_path, ext_used, ext_blocked, direction=1, max_steps=TOTAL_VERTICES)
        if len(ext_fwd2) > best_len:
            best_path = ext_fwd2
            best_len = len(best_path)

    return best_path

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    p = construct()
    t1 = time.time()
    print(f"Length: {len(p)}, Time: {t1-t0:.2f}s")
    # Validate
    for i in range(len(p)-1):
        if bin(p[i] ^ p[i+1]).count('1') != 1:
            print("Invalid consecutive edge at", i)
            break
    for i in range(len(p)):
        for j in range(i+2, len(p)):
            if bin(p[i] ^ p[j]).count('1') == 1:
                print("Invalid non-consecutive edge at", i, j)
                break
