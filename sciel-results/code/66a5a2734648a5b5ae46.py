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
    
    # Heuristic: prefer vertices that have fewer options (Most Constrained Variable)
    # But for snake-in-the-box, we just extend one end.
    # Actually, standard approach:
    # 1. Start with a random vertex.
    # 2. Maintain a set of "blocked" vertices for the current end.
    #    A vertex v is blocked for end E if v is adjacent to any vertex in the path 
    #    except E itself.
    #    Wait, the constraint is: new vertex v must not be adjacent to ANY vertex in path 
    #    except the current end.
    
    # Let's use a faster validity check.
    # Path P = [p0, p1, ..., pk].
    # Candidate v.
    # v must not be in P.
    # v must not be adjacent to p0, p1, ..., p_{k-1}.
    # v must be adjacent to pk.
    
    # Optimization:
    # Keep a set `used`.
    # Keep a set `forbidden`. `forbidden` contains all vertices adjacent to P[0...k-1].
    # When we add pk to the path, we add all neighbors of p_{k-1} to `forbidden`.
    # We remove pk from `forbidden`? No, pk is now the end.
    # Let's trace:
    # P = [0]. used={0}. forbidden={}.
    # Next v must be adj to 0, not in used, not in forbidden.
    # Say v=1.
    # New P = [0, 1].
    # used = {0, 1}.
    # forbidden for next step:
    #   Next vertex w must not be adj to 0.
    #   So forbidden = neighbors(0).
    #   Note: 1 is in neighbors(0). But 1 is in used.
    #   Also, we must allow w to be adj to 1.
    
    # Let's refine the `forbidden` set logic.
    # `forbidden` = { u | u is adjacent to some p_i where i < k }
    # When we extend P by v (so new end is v, old end was p_{k-1}):
    # New `forbidden` = (old `forbidden` union neighbors(p_{k-1}))
    # But we must allow the new end v to be adjacent to the next candidate.
    # The next candidate w must be:
    # 1. Adjacent to v.
    # 2. Not in `used`.
    # 3. Not in `forbidden`.
    
    # Wait, if w is in neighbors(p_{k-1}), it is in `forbidden`.
    # But w is adjacent to v.
    # Is it possible that w is adjacent to p_{k-1} and v?
    # Yes, if v and p_{k-1} share a neighbor.
    # In a hypercube, if v is adjacent to p_{k-1}, they differ in 1 bit.
    # Any other neighbor w of v differs from v in 1 bit.
    # If w is also adjacent to p_{k-1}, then w differs from p_{k-1} in 1 bit.
    # This forms a 4-cycle: p_{k-1} - v - w - p_{k-1}.
    # In Q_n, 4-cycles exist.
    # If w is adjacent to p_{k-1}, it violates the "induced path" condition 
    # because w would be adjacent to a non-consecutive vertex p_{k-1}.
    # So `forbidden` correctly blocks w.
    
    # So the algorithm is:
    # 1. used = {start}
    # 2. forbidden = set()
    # 3. path = [start]
    # 4. While possible:
    #    candidates = [u for u in neighbors(path[-1]) if u not in used and u not in forbidden]
    #    if not candidates: break
    #    choose u from candidates (heuristic)
    #    path.append(u)
    #    used.add(u)
    #    forbidden.update(neighbors(path[-2]))  # The vertex we just left
    #    # Note: we do NOT remove u from forbidden, because u is in used.
    #    # Actually, we don't need to remove anything from forbidden.
    #    # Forbidden only grows.
    
    # Is `forbidden` sufficient?
    # Condition: w not adjacent to any p_i for i < k.
    # `forbidden` contains all neighbors of p_0...p_{k-1}.
    # So yes, if w in forbidden, it is adjacent to some p_i (i<k).
    # If w not in forbidden, it is not adjacent to any p_i (i<k).
    # This is correct.
    
    # Heuristic for choosing u:
    # Standard: choose u that minimizes the number of further options?
    # Or choose u that maximizes something?
    # Let's try a simple heuristic:
    # Count the number of valid next steps for each candidate u.
    # Valid next steps for u: neighbors(u) \ (used \ {u}) \ (forbidden \ neighbors(u)??)
    # Actually, if we pick u, the new forbidden will be forbidden U neighbors(path[-1]).
    # Let's call current end E.
    # Candidate u.
    # New end u.
    # New forbidden = forbidden U neighbors(E).
    # Next candidates for u: neighbors(u) \ used_new \ new_forbidden.
    # used_new = used U {u}.
    # So next_candidates(u) = neighbors(u) \ (used U {u}) \ (forbidden U neighbors(E)).
    
    # We want to pick u that has the most next_candidates? Or fewest?
    # In constraint satisfaction, "fail-first" (fewest options) often prunes faster.
    # But for maximization, we might want to keep options open.
    # Let's try "most options" first, as it's a greedy maximization.
    # Or random if tied.
    
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
                # Heuristic: pick the candidate that leaves the most options for the next step
                best_score = -1
                best_v = None
                # Pre-calculate neighbors of cur to update forbidden later
                # For scoring, we need to simulate the move
                # new_forbidden = forbidden | set(NEIGHBORS[cur])
                # But we don't want to create a new set every time if possible.
                # Since N=9, sets are small.
                
                # Optimization: just pick random?
                # Let's try a light heuristic:
                # Score = number of neighbors of u that are not in used and not in (forbidden U neighbors(cur))
                
                # To speed up, we can just pick the first one if we want speed.
                # But let's try the heuristic.
                
                # Note: neighbors(cur) are the ones being added to forbidden.
                # So u is valid now.
                # For u's next move, it can go to any neighbor w of u such that:
                # w not in used (and w != u, which is true since u not in used yet)
                # w not in forbidden
                # w not in neighbors(cur)  <-- because neighbors(cur) will be added to forbidden
                
                # Let's calculate this score.
                # This is O(degree(u) * 1) = O(9) per candidate.
                # Candidates <= 9.
                # Total O(81) per step. Very fast.
                
                for u in candidates:
                    score = 0
                    for w in NEIGHBORS[u]:
                        if w in used: continue
                        if w in forbidden: continue
                        if w in NEIGHBORS[cur]: continue # Will be forbidden
                        score += 1
                    
                    if score > best_score:
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

    # Run multiple searches with different starting vertices and random seeds
    # To ensure variety, we can also randomize the tie-breaking in the heuristic.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        start_v = random.randint(0, SIZE - 1)
        
        # Allow some time for each search
        # If we do 1000 searches, we have ~0.1s each.
        # But the search might terminate early.
        # Let's just run until deadline.
        
        p = try_search(start_v, deadline)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
        # If we found a very long path, maybe we can try to extend it with backtracking?
        # For now, just restart.
        
        # Occasionally, try starting from 0 or other structured starts?
        # Random is usually fine.

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
