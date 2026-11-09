"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from collections import deque

N = 9
TIME_LIMIT = 115  # seconds, leave buffer before 120s

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using backtracking with
    constraint propagation and heuristic ordering.
    
    The induced path property requires:
    1. Consecutive vertices differ in exactly one bit (path edges)
    2. Non-consecutive vertices differ in at least two bits (no chords)
    
    We use depth-first search with:
    - Bit-level conflict tracking for fast chord detection
    - Heuristic neighbor ordering (prefer bits that keep more options open)
    - Time-bounded search with fallback to best found
    """
    best_path = [0]
    best_len = 1
    
    def bit_count(x):
        return x.bit_count()
    
    def try_extend(path, used, bit_used, start_depth):
        """
        Try to extend the path from the current end.
        Returns the extended path or None if stuck.
        """
        nonlocal best_path, best_len
        
        cur = path[-1]
        
        # Generate candidate next vertices (neighbors not yet used)
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v would create a chord with any non-consecutive vertex
            # A chord exists if v is at Hamming distance 1 from any path[i] where i < len(path)-2
            # (i.e., all vertices except the last two: the last is cur, second-last is path[-2])
            # Actually, non-consecutive means not adjacent in the path.
            # v is adjacent to cur (path[-1]) by construction.
            # v must not be adjacent to path[i] for any i where i is not len(path)-1 (cur)
            # and i is not len(path)-2 (path[-2], which is adjacent to cur, not necessarily to v)
            # Wait: v is a new vertex. It's adjacent to cur. We need v to NOT be adjacent to
            # any other vertex in the path except cur.
            # So check: for all p in path[:-1] (all except cur), bit_count(v ^ p) != 1
            # But path[-2] is adjacent to cur, and v is adjacent to cur. v could be adjacent to path[-2]?
            # If v == path[-2], then v is in used, so we'd skip it.
            # If v != path[-2], can v be adjacent to path[-2]? Yes, if they differ in one bit.
            # So we need to check all vertices in path except cur.
            
            valid = True
            for p in path[:-1]:  # all vertices except the last one (cur)
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            
            if valid:
                candidates.append((b, v))
        
        if not candidates:
            return None
        
        # Heuristic: order candidates by how "open" they are
        # Prefer candidates that have more unused neighbors available
        def score_candidate(b, v):
            # Count how many neighbors of v are still unused and valid
            count = 0
            for bb in range(N):
                vv = v ^ (1 << bb)
                if vv not in used and vv != cur:
                    # Quick check: would vv be chord-free if added later?
                    # Just count for now as a heuristic
                    count += 1
            # Also penalize if v is close to many existing path vertices
            # (higher min distance to non-adjacent vertices is better)
            min_dist = 9
            for p in path[:-1]:
                d = (v ^ p).bit_count()
                if d < min_dist:
                    min_dist = d
            return (count, min_dist)
        
        candidates.sort(key=score_candidate, reverse=True)
        
        # Try each candidate
        for b, v in candidates:
            # Check time
            if time.time() - start_depth > TIME_LIMIT - 5:
                return None
            
            new_used = used | {v}
            new_path = path + [v]
            
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = new_path[:]
            
            # Try to extend further
            result = try_extend(new_path, new_used, bit_used, start_depth)
            if result is not None:
                return result
        
        return None
    
    # Initial greedy to get a decent starting path
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    
    best_path = path[:]
    best_len = len(path)
    
    # Now try to improve with backtracking from various points
    # Strategy: try to extend the greedy path using backtracking
    start_time = time.time()
    
    # Try backtracking extension from the end of the greedy path
    # We'll do a limited DFS trying different choices at each step
    def dfs_extend(current_path, current_used, depth, start_time):
        nonlocal best_path, best_len
        
        if time.time() - start_time > TIME_LIMIT - 3:
            return
        
        cur = current_path[-1]
        
        # Generate valid next vertices
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in current_used:
                continue
            valid = True
            for p in current_path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                candidates.append(v)
        
        if not candidates:
            if len(current_path) > best_len:
                best_len = len(current_path)
                best_path = current_path[:]
            return
        
        # Order candidates: prefer those with more future options
        def cand_score(v):
            cnt = 0
            for bb in range(N):
                vv = v ^ (1 << bb)
                if vv not in current_used and vv != cur:
                    cnt += 1
            return cnt
        
        candidates.sort(key=cand_score, reverse=True)
        
        for v in candidates:
            if time.time() - start_time > TIME_LIMIT - 3:
                return
            
            new_path = current_path + [v]
            new_used = current_used | {v}
            
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = new_path[:]
            
            dfs_extend(new_path, new_used, depth + 1, start_time)
    
    # Try extending from the greedy path
    dfs_extend(path, used, 0, start_time)
    
    # Try different starting strategies: random restarts with different initial bits
    if time.time() - start_time < TIME_LIMIT - 10:
        random.seed(42)
        for _ in range(50):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Start with a random vertex
            start_v = random.randint(0, (1 << N) - 1)
            p = [start_v]
            u = {start_v}
            
            # Greedy extension
            while True:
                cur = p[-1]
                nxt = None
                # Randomize bit order
                bits = list(range(N))
                random.shuffle(bits)
                for b in bits:
                    v = cur ^ (1 << b)
                    if v in u:
                        continue
                    valid = True
                    for pp in p[:-1]:
                        if (v ^ pp).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        nxt = v
                        break
                if nxt is None:
                    break
                p.append(nxt)
                u.add(nxt)
            
            if len(p) > best_len:
                best_len = len(p)
                best_path = p[:]
            
            # Try to extend with backtracking
            if time.time() - start_time < TIME_LIMIT - 5:
                dfs_extend(p, u, 0, start_time)
    
    return best_path
# EVOLVE-BLOCK-END
