import random
import time

N = 9
V = 1 << N
MAX_TIME = 110

# Precompute neighbors
NEIGHBORS = [[] for _ in range(V)]
for v in range(V):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    best_path = []
    best_len = 0
    start_time = time.time()
    
    def greedy_run():
        start = random.randint(0, V - 1)
        path = [start]
        used = {start}
        blocked = set()
        
        # Initial blocked: neighbors of start? No, start is the only node.
        # Internal nodes: none.
        # So blocked is empty.
        
        while True:
            cur = path[-1]
            # Find valid moves
            valid_moves = []
            for nb in NEIGHBORS[cur]:
                if nb not in used and nb not in blocked:
                    valid_moves.append(nb)
            
            if not valid_moves:
                break
            
            # Calculate freedom for each valid move
            # We want to minimize the number of future options (Warnsdorff)
            # but with randomness.
            
            # To speed up, we can pre-compute the "potential" of each neighbor
            # without iterating all neighbors of the candidate?
            # No, we need to know how many neighbors of the candidate are free.
            
            scored_moves = []
            for v in valid_moves:
                count = 0
                # Iterate neighbors of v
                for w in NEIGHBORS[v]:
                    # w must not be used
                    # w must not be in blocked
                    # w must not be cur (cur is in used, so covered)
                    if w not in used and w not in blocked:
                        count += 1
                scored_moves.append((count, v))
            
            # Sort by count
            scored_moves.sort(key=lambda x: x[0])
            
            # Selection
            if len(scored_moves) > 1:
                # With 70% probability, pick from the best 3
                # With 30% probability, pick the absolute best
                # This balances exploitation and exploration
                if random.random() < 0.7 and len(scored_moves) >= 3:
                    v = random.choice(scored_moves[:3])[1]
                elif random.random() < 0.7 and len(scored_moves) == 2:
                    v = random.choice(scored_moves)[1]
                else:
                    v = scored_moves[0][1]
            else:
                v = scored_moves[0][1]
                
            path.append(v)
            used.add(v)
            
            # Update blocked
            # New blocked = Old blocked U (Neighbors(cur) \ {v})
            # We can update in place? 
            # No, we need the old blocked for the freedom calc of the NEXT step?
            # Yes.
            # But we are updating it for the NEXT iteration's validity check.
            # The freedom calc for the NEXT step will use the NEW blocked.
            
            # Optimization: 
            # Instead of creating a new set, we can just add to the existing set.
            # But we must not remove anything.
            # Is `blocked` monotonic?
            # Yes, as established before.
            # So we can just add.
            
            for nb in NEIGHBORS[cur]:
                if nb != v:
                    blocked.add(nb)
            
        return path

    # Main loop
    while time.time() - start_time < MAX_TIME:
        p = greedy_run()
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        
    return best_path
