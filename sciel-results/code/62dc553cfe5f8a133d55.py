import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # 5s margin
    
    best_set = None
    
    # Try a few different strategies/restarts
    # Strategy: Greedy with max blocking power
    # We can randomize the tie-breaking or the initial set of candidates
    
    for restart in range(100): # Max 100 restarts, break if time runs out
        if time.time() > deadline:
            break
            
        S = []
        K = set()
        
        # To add randomness, we can shuffle the range or use random priorities
        # But "max blocking" is deterministic given S.
        # We can randomize the order in which we check candidates if there's a tie in blocking power?
        # Or just run the deterministic heuristic. It might get stuck in a local optimum.
        # Let's add a small random perturbation: with prob 0.1, pick a random available instead of max blocking.
        
        while True:
            # Find all available candidates
            available = []
            for y in range(1, N + 1):
                if y in S:
                    continue
                # Check if y is available
                if 2 * y in K:
                    continue
                ok = True
                for s in S:
                    if y + s in K:
                        ok = False
                        break
                if ok:
                    available.append(y)
            
            if not available:
                break
            
            # Calculate blocking power for each available y
            # Blocking power: number of z in available \ {y} that become unavailable if y is added
            best_y = None
            max_block = -1
            candidates_for_block = available[:]
            
            # To speed up, we can just pick the first one if we don't want to compute block power for all?
            # No, computing block power is cheap.
            
            # Randomize the order of available to break ties or introduce variation
            random.shuffle(candidates_for_block)
            
            for y in candidates_for_block:
                # Simulate adding y
                new_K = K | {2 * y}
                for s in S:
                    new_K.add(y + s)
                
                new_S = S + [y]
                
                block_count = 0
                for z in available:
                    if z == y:
                        continue
                    # Check if z is still available with new_K, new_S
                    if 2 * z in new_K:
                        continue
                    z_ok = True
                    for s in new_S:
                        if z + s in new_K:
                            z_ok = False
                            break
                    if z_ok:
                        block_count += 1
                        # Wait, if z is STILL available, it was NOT blocked.
                        # We want to count how many ARE blocked.
                        # So if z_ok is True, it was not blocked.
                        # We want to maximize the number of blocked ones.
                        # So we want to MINIMIZE the number of still-available ones.
                        pass
                
                # Actually, let's just count how many are blocked.
                # blocked = len(available) - 1 - (number of still available)
                # Let's re-evaluate.
                
                still_available = 0
                for z in available:
                    if z == y:
                        continue
                    if 2 * z in new_K:
                        continue
                    z_ok = True
                    for s in new_S:
                        if z + s in new_K:
                            z_ok = False
                            break
                    if z_ok:
                        still_available += 1
                
                blocked = (len(available) - 1) - still_available
                
                if blocked > max_block:
                    max_block = blocked
                    best_y = y
                    
                # If we want randomness, we could add noise to 'blocked'
        
            if best_y is None:
                # Should not happen if available is not empty
                break
                
            # With some probability, pick a random one to escape local optima?
            if random.random() < 0.1:
                best_y = random.choice(available)
                
            S.append(best_y)
            K.add(2 * best_y)
            for s in S[:-1]: # S[:-1] is the old S
                K.add(best_y + s)
                
        if best_set is None or len(S) < len(best_set):
            best_set = S[:]
            
    if best_set is None:
        # Fallback to simple greedy if something went wrong
        S = []
        K = set()
        for x in range(1, N + 1):
            added = [x + a for a in S] + [2 * x]
            if len(set(added)) == len(added) and not K.intersection(added):
                S.append(x)
                K.update(added)
        best_set = S
        
    return best_set
