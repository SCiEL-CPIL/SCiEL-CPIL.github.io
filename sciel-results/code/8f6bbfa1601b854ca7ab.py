import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for all vertices
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(SIZE)]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_path = []
    best_len = 0
    
    def try_search(start_v, time_limit):
        used = {start_v}
        forbidden = set()
        path = [start_v]
        
        while time.time() < time_limit:
            cur = path[-1]
            neighbors = NEIGHBORS[cur]
            
            candidates = []
            for u in neighbors:
                if u in used or u in forbidden:
                    continue
                candidates.append(u)
            
            if not candidates:
                break
            
            if len(candidates) == 1:
                next_v = candidates[0]
            else:
                # Warnsdorff Heuristic: Prefer the candidate that has the FEWEST
                # options for the next step. This is a "fail-first" strategy that
                # tends to find long paths by avoiding dead-ends early.
                best_score = 100  # Max possible score is 9
                best_v = None
                
                # Pre-calculate the set of neighbors that will be forbidden
                # after we move from 'cur' to 'next_v'.
                # These are neighbors(cur).
                # Note: 'next_v' is in neighbors(cur), but it will be in 'used'.
                # We check if w is in neighbors(cur) to see if it becomes forbidden.
                
                for u in candidates:
                    score = 0
                    for w in NEIGHBORS[u]:
                        if w in used: continue
                        if w in forbidden: continue
                        if w in NEIGHBORS[cur]: continue # Will be forbidden
                        score += 1
                    
                    # Tie-breaking: random to diversify
                    if score < best_score:
                        best_score = score
                        best_v = u
                    elif score == best_score and random.random() < 0.5:
                        best_v = u
                
                next_v = best_v
            
            path.append(next_v)
            used.add(next_v)
            
            # Update forbidden: add neighbors of the vertex we just left (cur)
            # cur is path[-2]
            for n in NEIGHBORS[cur]:
                forbidden.add(n)
            
        return path

    def try_improve(path, time_limit):
        """
        Try to extend the path at the beginning or end by finding a vertex
        that can be prepended or appended.
        """
        if not path:
            return path
        
        used = set(path)
        
        # We can try to extend at the end (path[-1]) or start (path[0]).
        # The constraints are symmetric if we reverse the path.
        # Let's just try to extend the end first, then the start.
        
        # Extend End
        # Current end is path[-1].
        # Forbidden for new end v: v must not be adjacent to any p_i for i < len(path)-1.
        # So forbidden_end = union of neighbors(p_i) for i in range(len(path)-1)
        
        # This is expensive to compute from scratch every time.
        # But we can just check candidates.
        
        def can_extend_end(v):
            # v must be adjacent to path[-1]
            if (v ^ path[-1]).bit_count() != 1:
                return False
            if v in used:
                return False
            # v must not be adjacent to any p_i for i < len(path)-1
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    return False
            return True

        def can_extend_start(v):
            # v must be adjacent to path[0]
            if (v ^ path[0]).bit_count() != 1:
                return False
            if v in used:
                return False
            # v must not be adjacent to any p_i for i > 0
            for i in range(1, len(path)):
                if (v ^ path[i]).bit_count() == 1:
                    return False
            return True

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to extend end
            end_candidates = [u for u in NEIGHBORS[path[-1]] if can_extend_end(u)]
            if end_candidates:
                # Pick one (random or heuristic)
                # Let's pick one that allows further extension? Or just random.
                # Random is fast.
                next_v = random.choice(end_candidates)
                path.append(next_v)
                used.add(next_v)
                improved = True
                continue
            
            # Try to extend start
            start_candidates = [u for u in NEIGHBORS[path[0]] if can_extend_start(u)]
            if start_candidates:
                next_v = random.choice(start_candidates)
                path.insert(0, next_v)
                used.add(next_v)
                improved = True
                continue
                
        return path

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        start_v = random.randint(0, SIZE - 1)
        
        # Run greedy search
        p = try_search(start_v, deadline)
        
        # Try to improve
        if len(p) > 1:
            p = try_improve(p, deadline)
            
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
    return best_path

if __name__ == "__main__":
    # Quick test
    p = construct()
    print(f"Length: {len(p)}")
    # Verify
    if len(p) > 1:
        for i in range(len(p)-1):
            if (p[i] ^ p[i+1]).bit_count() != 1:
                print(f"Error: consecutive {p[i]} and {p[i+1]} not adjacent")
                break
        for i in range(len(p)):
            for j in range(i+2, len(p)):
                if (p[i] ^ p[j]).bit_count() == 1:
                    print(f"Error: non-consecutive {p[i]} and {p[j]} adjacent")
                    break
