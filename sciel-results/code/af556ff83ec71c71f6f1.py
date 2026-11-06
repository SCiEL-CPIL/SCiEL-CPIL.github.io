import time
import random
import sys

N = 9
MASK = (1 << N) - 1
TOTAL_VERTS = 1 << N  # 512
DEADLINE = 145.0  # Safety margin

# Precompute neighbors
_NEIGHBORS = []
for v in range(TOTAL_VERTS):
    nbrs = []
    for b in range(N):
        nbrs.append(v ^ (1 << b))
    _NEIGHBORS.append(nbrs)

# Precompute bit_count for all possible masks? No, 512 vertices, masks are 512 bits.
# Python int bit_count is fast.

def _is_valid_snake_full(path):
    """Full validation: consecutive differ by 1, non-consecutive differ by >= 2."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _warnsdorff_search(start_v, deadline, rng, max_depth=None):
    """
    DFS with Warnsdorff's rule.
    Returns the longest path found.
    """
    best_path = [start_v]
    path = [start_v]
    used_mask = 1 << start_v
    
    # We need to track the best path found during the search
    # Since we backtrack, we update best_path whenever we find a longer one
    
    def dfs(cur, depth):
        nonlocal best_path, used_mask
        
        if time.time() > deadline:
            return False
        
        # Update best if current path is longer
        if len(path) > len(best_path):
            best_path = list(path)
            
        # If we hit max depth or time, stop
        if max_depth and depth >= max_depth:
            return True
            
        cur = path[-1]
        
        # Generate candidates
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if (used_mask >> v) & 1:
                continue
            
            # Check induced constraint: v must not be adjacent to any used vertex except cur
            valid = True
            for w in _NEIGHBORS[v]:
                if w == cur:
                    continue
                if (used_mask >> w) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Heuristic: Count onward moves for v
            # An onward move u from v is valid if:
            # 1. u is not used
            # 2. u is not adjacent to any used vertex (except v)
            # Note: v is now "used" for the purpose of checking u's neighbors?
            # Actually, if we move to v, the used set becomes used_mask | (1<<v).
            # So for a candidate u of v, we check if u is adjacent to any w in (used_mask | (1<<v)) \ {v}?
            # No, the condition is: u must not be adjacent to any w in used_mask \ {cur}.
            # And u must not be adjacent to v? No, u is adjacent to v by definition.
            # The condition for u to be a valid *next* move from v is:
            # u is not in used_mask.
            # No w in used_mask (w != v, w != cur? No, w can be anything in used_mask except v) is adjacent to u.
            # Wait, if we add v to used_mask, then for u to be valid, u must not be adjacent to any w in (used_mask | {v}) \ {v}?
            # No, the snake condition is: non-consecutive vertices differ by >= 2 bits.
            # If we have path ... cur -> v -> u, then u must not be adjacent to any vertex in path[:-2].
            # The vertices in path[:-2] are in used_mask \ {cur, v}? No, used_mask contains all path vertices.
            # So u must not be adjacent to any w in used_mask such that w != cur and w != v.
            
            onward_count = 0
            new_used_mask = used_mask | (1 << v)
            for b2 in range(N):
                u = v ^ (1 << b2)
                if (new_used_mask >> u) & 1:
                    continue
                # Check if u is adjacent to any w in new_used_mask \ {v}
                # i.e., w in used_mask (since v is new) OR w == v? No, v is excluded.
                # So w in used_mask.
                blocked = False
                for w in _NEIGHBORS[u]:
                    if w == v:
                        continue
                    if (used_mask >> w) & 1:
                        blocked = True
                        break
                if not blocked:
                    onward_count += 1
            
            candidates.append((onward_count, v))
            
        if not candidates:
            return False
            
        # Sort by onward_count (Warnsdorff: fewest onward moves first)
        candidates.sort(key=lambda x: x[0])
        
        # Add some randomness to tie-breakers or top few to avoid determinism issues
        # If all counts are same, randomize order
        if len(candidates) > 1 and candidates[0][0] == candidates[-1][0]:
            rng.shuffle(candidates)
        else:
            # Randomize within same count?
            # Simple approach: shuffle the list, then stable sort by count
            # Or just pick randomly from those with min count?
            # Let's shuffle candidates with min count
            min_count = candidates[0][0]
            min_cands = [c for c in candidates if c[0] == min_count]
            other_cands = [c for c in candidates if c[0] > min_count]
            rng.shuffle(min_cands)
            candidates = min_cands + other_cands
            
        for count, v in candidates:
            path.append(v)
            used_mask |= (1 << v)
            
            if dfs(v, depth + 1):
                return True
                
            path.pop()
            used_mask ^= (1 << v)
            
            if time.time() > deadline:
                return False
                
        return False
        
    dfs(start_v, 0)
    return best_path

def _greedy_extend_deterministic(path, used_mask):
    """Deterministic greedy: lowest bit flip first."""
    while True:
        cur = path[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if (used_mask >> v) & 1:
                continue
            # Check induced constraint
            valid = True
            for w in _NEIGHBORS[v]:
                if w == cur:
                    continue
                if (used_mask >> w) & 1:
                    valid = False
                    break
            if valid:
                path.append(v)
                used_mask |= (1 << v)
                extended = True
                break
        if not extended:
            break
    return path, used_mask

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Multiple Warnsdorff restarts
    # The search is expensive, so we do a limited number of restarts
    # Each restart tries to find a long snake from a random start
    
    # We can also try to extend the best path found so far
    
    for i in range(50): # Max 50 restarts
        if time.time() > DEADLINE - 5:
            break
            
        start_v = rng.randint(0, TOTAL_VERTS - 1)
        
        # Run Warnsdorff search with a time limit per run
        # If we allow it to run until global deadline, it might get stuck in one long search
        # Better to have a per-run budget or just let it run and check global time inside
        # The dfs function checks global time.
        
        path = _warnsdorff_search(start_v, DEADLINE - 2, rng)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            # Try to extend this best path greedily
            used_mask = 0
            for v in best_path:
                used_mask |= (1 << v)
            extended_path, _ = _greedy_extend_deterministic(list(best_path), used_mask)
            if len(extended_path) > best_len:
                best_len = len(extended_path)
                best_path = list(extended_path)
        
        # If we found a very long snake, we can stop early?
        # Max possible is 512. Best known is ~190+.
        # If we get > 200, it's good.
        if best_len > 200:
             # Try a few more to see if we can push higher
             pass
             
    # Final validation
    if best_path and not _is_valid_snake_full(best_path):
        # Fallback
        path = [0]
        used_mask = 1 << 0
        path, _ = _greedy_extend_deterministic(path, used_mask)
        best_path = path
        best_len = len(path)
        
    # Ensure we have at least the seed length
    if len(best_path) < 104:
        path = [0]
        used_mask = 1 << 0
        path, _ = _greedy_extend_deterministic(path, used_mask)
        best_path = path
        best_len = len(path)
        
    return best_path

if __name__ == "__main__":
    result = construct()
    print(f"Snake length: {len(result)}")
