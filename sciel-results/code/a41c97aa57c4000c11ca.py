import random
import time
import math

def construct():
    N = 180
    vertices = list(range(1, N + 1))
    
    # Precompute forbidden triples
    # 1/a = 1/b + 1/c  =>  c = ab / (b-a) for b > a
    # We store for each vertex v, the list of pairs (u, w) such that {u, v, w} is a forbidden triple.
    # Actually, it's better to store for each vertex v, the list of (u, w) such that if u and w are present, v cannot be added.
    # Or simpler: for each pair (u, w), store the set of v's that conflict with them.
    # Let's stick to: triples = {v: [(u, w), ...]} where u, w are the other two elements.
    
    triples = {i: [] for i in vertices}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c <= N and c != a and c != b:
                    # Triple (a, b, c)
                    triples[a].append((b, c))
                    triples[b].append((a, c))
                    triples[c].append((a, b))
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 50.0
    
    # Initial solution: Greedy
    random.shuffle(vertices)
    current_set = set()
    for v in vertices:
        can_add = True
        for (u, w) in triples[v]:
            if u in current_set and w in current_set:
                can_add = False
                break
        if can_add:
            current_set.add(v)
    
    best_set = current_set.copy()
    best_size = len(current_set)
    
    # Annealing parameters
    # Temperature schedule: T = T0 * alpha^steps or T = T0 * exp(-k*t)
    # Let's use a simple linear decay or exponential.
    # We will iterate for a fixed number of steps or until time limit.
    
    T = 10.0
    alpha = 0.9999
    min_T = 0.01
    
    # To make it robust, we run multiple annealing runs (restarts) with different seeds/initials
    # but within the time limit.
    
    while time.time() - start_time < time_limit:
        # Reset temperature for each "run" or decay continuously?
        # Let's do a single long annealing run with restarts if stuck.
        # Or simpler: Just one long run with T decaying.
        
        # Actually, random restarts with SA are better.
        # Let's do: 
        # 1. Greedy init
        # 2. SA phase for ~5 seconds
        # 3. If time remains, restart
        
        # Wait, the current code does random restarts of greedy.
        # My new approach:
        # 1. Greedy init
        # 2. SA phase
        
        # Let's implement a single SA phase that runs for the remaining time?
        # No, SA benefits from restarts.
        # Let's do:
        # while time < limit:
        #    greedy_init()
        #    sa_phase(duration=5s)
        #    update_best()
        
        # Let's refine the SA step.
        # State: current_set
        # Move: 
        #   1. Pick v not in set, try to add. If valid, accept.
        #   2. Pick v in set, try to remove. If it allows adding 2+ others, accept.
        #   3. Swap: remove v, add w.
        
        # To make it fast, we need efficient conflict checking.
        # conflict_count[v] = number of pairs (u, w) in triples[v] such that u, w in current_set.
        # If we add x:
        #   For each (u, w) in triples[x], if u, w in set, conflict.
        #   Also, adding x increases conflict_count for neighbors?
        #   Actually, we just need to check if adding x creates a triple.
        #   A triple is formed if x is added and there exist u, w in set such that (u, w) is in triples[x].
        #   So, `conflicts_for_x = sum(1 for (u, w) in triples[x] if u in current_set and w in current_set)`
        #   If conflicts_for_x == 0, we can add.
        
        # Let's maintain `in_set` boolean array for O(1) lookup.
        
        in_set = [False] * (N + 1)
        for v in current_set:
            in_set[v] = True
            
        # We will run SA for a fixed number of steps or time slice.
        # Let's say 2 seconds per SA run.
        sa_start = time.time()
        while time.time() - sa_start < 2.0:
            T *= alpha
            if T < min_T:
                break
            
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a vertex
                candidates = [v for v in vertices if not in_set[v]]
                if not candidates:
                    continue
                v = random.choice(candidates)
                
                # Check if v can be added
                # v can be added if no (u, w) in triples[v] has both u, w in set
                valid = True
                for (u, w) in triples[v]:
                    if in_set[u] and in_set[w]:
                        valid = False
                        break
                
                if valid:
                    current_set.add(v)
                    in_set[v] = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set.copy()
                else:
                    # Reject or accept with probability exp(-delta/T)?
                    # Delta is "badness". If we add v, we create conflicts.
                    # But we can't have conflicts in the final set.
                    # So we can only accept if valid.
                    # Unless we allow temporary invalid states? 
                    # No, the constraint is hard. We must maintain a valid set.
                    # So "Add" move is only accepted if valid.
                    pass
                    
            elif move_type < 0.7:
                # Try to remove a vertex
                if not current_set:
                    continue
                v = random.choice(list(current_set))
                
                # Removing v is always valid (set remains valid).
                # But it decreases size. Accept with probability exp(-1/T)?
                # Or always accept if it helps?
                # Let's just remove it. It might allow adding others.
                # Actually, removing a vertex that is not part of any "blocking" pair is useless.
                # But we don't know that easily.
                # Let's accept removal with probability p_remove.
                # If we remove, size decreases by 1.
                # We accept if random() < math.exp(-1.0 / T)
                if random.random() < math.exp(-1.0 / T):
                    current_set.remove(v)
                    in_set[v] = False
                    
            else:
                # Swap: Remove v, Add w
                if not current_set:
                    continue
                v = random.choice(list(current_set))
                candidates = [w for w in vertices if not in_set[w]]
                if not candidates:
                    continue
                w = random.choice(candidates)
                
                # Check if w can be added after removing v
                # Temporarily remove v
                in_set[v] = False
                current_set.remove(v)
                
                valid = True
                for (u, u2) in triples[w]:
                    if in_set[u] and in_set[u2]:
                        valid = False
                        break
                
                if valid:
                    # Accept swap (size same, or we might have removed a "bad" vertex)
                    current_set.add(w)
                    in_set[w] = True
                    # Note: size is same. We don't update best.
                else:
                    # Revert
                    current_set.add(v)
                    in_set[v] = True
                    # Reject swap
                
        # End SA phase
        
        # If we finished SA early or T too low, we might restart.
        # The loop condition `while time.time() - start_time < time_limit` handles the outer loop.
        # We should update best_set from current_set if it's better? 
        # We already update best_size/best_set inside.
        
    return sorted(best_set)
