"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TIME_LIMIT = 115.0  # Safety margin before 120s/150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # 1. Greedy Seed
    def get_greedy_seed():
        path, used = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
            
    best_path = get_greedy_seed()
    best_len = len(best_path)
    
    # 2. Bitset Data Structures for Fast Validity
    # adj_mask[v] is a bitmask of all vertices at Hamming distance 1 from v
    adj_mask = [0] * (1 << N)
    for v in range(1 << N):
        m = 0
        for b in range(N):
            m |= 1 << (v ^ (1 << b))
        adj_mask[v] = m
        
    # 3. Simulated Annealing
    T = 50.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Restart if stuck or to explore different basins
        # We restart SA occasionally to avoid getting trapped in a single local maximum
        # Here we run a short burst of SA, then restart with a slightly perturbed seed
        # Actually, let's just run SA on the current best and occasionally restart from scratch
        # to find better basins.
        
        # For simplicity and robustness, we run SA on the current best path.
        # If we don't improve for a long time, we might restart, but let's stick to one long run
        # or a few restarts. Given the 115s budget, a few restarts is good.
        
        # Let's implement a single SA run that modifies the path in place.
        # We need to be careful with the "induced path" constraint.
        
        # State: path (list of ints)
        # Moves:
        # A. Extend: If possible, extend the path (greedy step).
        # B. Shift: Pick a segment of the path, remove it, and try to re-insert it elsewhere or shift it.
        #    A simpler move: Pick two non-adjacent vertices in the path, try to "swap" their positions?
        #    No, swapping breaks the path connectivity.
        #    
        #    Better Move: "Segment Re-routing"
        #    Pick i, j (i < j). The segment path[i:j] is a sub-path.
        #    Can we remove path[i:j] and insert it at position k?
        #    This is complex to validate quickly.
        
        # Let's use a simpler local search move: "Flip" a bit in a vertex?
        # If we change a vertex v to v', it must differ by 1 bit from prev and next, and not be adjacent to others.
        # This is a "local mutation".
        
        # Let's try a different approach: 
        # 1. Start with greedy seed.
        # 2. While time remains:
        #    a. Try to extend. If success, update best.
        #    b. If stuck, perform a random "kick": 
        #       - Pick a random internal vertex, try to change it to a neighbor that maintains validity.
        #       - Or, pick a random pair of vertices (u, v) in the path that are not adjacent in the path 
        #         but are adjacent in the cube? No, that would create a shortcut (invalid induced path).
        
        # Actually, the most effective move for snakes is "Segment Shift".
        # Let's implement a simplified "Kick" that tries to replace a vertex with a neighbor.
        
        cur_path = list(best_path)
        cur_len = len(cur_path)
        
        # Precompute neighbors for current path? 
        # To check validity of a new vertex v at position i:
        # 1. v must be adjacent to cur_path[i-1] and cur_path[i+1] (if they exist).
        # 2. v must not be adjacent to any other vertex in cur_path.
        
        # We will use a set of used vertices for fast lookup.
        # But checking "not adjacent to any other" requires checking all others.
        # We can maintain a "blocked" set: vertices that are adjacent to any vertex in the path.
        # If we add v, it becomes blocked. If we remove v, we need to update blocked.
        
        # Let's maintain:
        # used: set of vertices in path
        # blocked: set of vertices adjacent to any vertex in path (excluding the path itself? no, including)
        # Actually, a vertex x is "forbidden" if x in used OR x is adjacent to any vertex in used.
        # Wait, the condition is: non-consecutive vertices differ in at least 2 bits.
        # This means for any v in path, no other u in path (except prev/next) can be v^bit.
        # So, for a candidate v, it is valid at position i if:
        # - v is not in used (unless it's the same vertex, which we're replacing)
        # - v is not adjacent to any u in used, except possibly cur_path[i-1] and cur_path[i+1].
        
        # Let's implement the "Replace" move.
        # Pick i. Try all 9 neighbors of cur_path[i].
        # For each neighbor v:
        #   1. Check if v is adjacent to cur_path[i-1] (if i>0).
        #   2. Check if v is adjacent to cur_path[i+1] (if i<len-1).
        #   3. Check if v is adjacent to any other cur_path[j] (j != i-1, i, i+1).
        
        # This check is O(L). With L~200, it's fast enough for thousands of iterations.
        
        # Let's run SA with "Replace" moves.
        # Acceptance: always accept if valid (since length doesn't change, but it might allow future extensions).
        # Actually, "Replace" moves don't change length. We need "Extend" moves to improve length.
        # So the strategy is: 
        # 1. Try to extend.
        # 2. If stuck, do random Replace moves to "shake up" the path, hoping to create a new extension point.
        
        # We'll alternate: 
        # - If we can extend, do it.
        # - Else, do a batch of random Replace moves.
        
        # To make it faster, we'll keep `used` as a set.
        
        used = set(cur_path)
        
        # Helper to check if v can be at position i
        def can_insert(v, i, path, used_set):
            # v must not be in used_set (unless it's the vertex being replaced, but we handle that separately)
            # For extension, v is new.
            # For replacement, v is new, old is removed.
            
            # Check adjacency to neighbors in path
            if i > 0 and (v ^ path[i-1]).bit_count() != 1:
                return False
            if i < len(path)-1 and (v ^ path[i+1]).bit_count() != 1:
                return False
            
            # Check adjacency to others
            for j, p in enumerate(path):
                if j == i or j == i-1 or j == i+1:
                    continue
                if (v ^ p).bit_count() == 1:
                    return False
            return True

        # SA Loop
        no_improve_count = 0
        max_no_improve = 10000
        
        while time.time() < deadline:
            # 1. Try to extend
            extended = False
            if cur_len < (1 << N):
                # Try to extend at the end
                # Also try to extend at the start?
                # Let's stick to end extension for simplicity, or random end.
                # We can reverse the path to extend from the other end.
                
                # Try end
                last = cur_path[-1]
                for b in range(N):
                    v = last ^ (1 << b)
                    if v in used:
                        continue
                    if can_insert(v, len(cur_path), cur_path, used):
                        cur_path.append(v)
                        used.add(v)
                        cur_len += 1
                        extended = True
                        no_improve_count = 0
                        if cur_len > best_len:
                            best_len = cur_len
                            best_path = list(cur_path)
                        break
                if extended:
                    continue
                
                # Try start (reverse path logic)
                # Actually, we can just check the first element
                first = cur_path[0]
                for b in range(N):
                    v = first ^ (1 << b)
                    if v in used:
                        continue
                    if can_insert(v, -1, cur_path, used): # -1 means before 0
                        # Special check for start
                        if len(cur_path) > 0 and (v ^ cur_path[0]).bit_count() != 1:
                            continue
                        for j, p in enumerate(cur_path):
                            if j == 0:
                                continue
                            if (v ^ p).bit_count() == 1:
                                break
                        else:
                            cur_path.insert(0, v)
                            used.add(v)
                            cur_len += 1
                            extended = True
                            no_improve_count = 0
                            if cur_len > best_len:
                                best_len = cur_len
                                best_path = list(cur_path)
                            break
                if extended:
                    continue
            
            # 2. If stuck, do a Replace move
            # Pick a random position i (1 to len-2 to avoid ends which are harder to fix? or any)
            # Let's pick any internal position.
            if cur_len > 2:
                i = random.randint(1, cur_len - 2)
                old_v = cur_path[i]
                # Try a random neighbor
                b = random.randint(0, N-1)
                new_v = old_v ^ (1 << b)
                
                if new_v in used:
                    no_improve_count += 1
                    continue
                
                if can_insert(new_v, i, cur_path, used):
                    # Perform move
                    used.remove(old_v)
                    used.add(new_v)
                    cur_path[i] = new_v
                    no_improve_count = 0
                    # Check if this allows extension? (handled in next loop)
                else:
                    no_improve_count += 1
            
            # Cooling / Restart logic
            if no_improve_count > max_no_improve:
                # Restart from best
                cur_path = list(best_path)
                used = set(cur_path)
                cur_len = len(cur_path)
                no_improve_count = 0
                T *= 0.9
                if T < T_min:
                    break

    return best_path
# EVOLVE-BLOCK-END
