import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append((x, y, z))
            
            trans_sorted = [tuple(sorted(t)) for t in transversals]
            trans_indices = [triple_to_idx[t] for t in trans_sorted]
            
            k_idx = len(k222_list)
            k222_list.append(set(trans_indices))
            
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)
    
    num_k222s = len(k222_list)
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
        
    # 1. Initial Greedy Construction
    # Randomize order, add if valid
    def greedy_init():
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            if time_left() < 5:
                break
            # Check validity
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
                    
        return active, counts, current_size

    best_size = 0
    best_active = None
    best_counts = None
    
    # Run multiple SA restarts
    # SA parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.9999 # Cooling rate
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time_left() < 10:
            break
            
        active, counts, current_size = greedy_init()
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
            
        T = T_start
        # Fixed number of iterations or time-based? Time-based is safer.
        # Let's do iterations until time runs out for this restart
        # Estimate iterations: 1 move ~ 1us? 120k K222s, ~15 per triple.
        # 15 checks per move. 15 * 100ns = 1.5us. ~600k moves/sec.
        # 110s -> 60M moves. Plenty.
        
        while time_left() > 5:
            # Pick a random triple
            t_idx = random.randint(0, num_triples - 1)
            
            old_size = current_size
            
            if active[t_idx]:
                # Try to remove
                # Decrement counts
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] -= 1
                current_size -= 1
                
                # Acceptance
                if current_size >= old_size: # Never true for removal
                    pass
                else:
                    # Delta = -1
                    if random.random() < pow(2.718281828, (current_size - old_size) / T):
                        # Accept
                        active[t_idx] = False
                    else:
                        # Reject: Revert
                        active[t_idx] = True
                        for k_idx in triple_to_k222s[t_idx]:
                            counts[k_idx] += 1
                        current_size = old_size
            else:
                # Try to add
                # Check validity
                valid = True
                for k_idx in triple_to_k222s[t_idx]:
                    if counts[k_idx] == 7:
                        valid = False
                        break
                
                if valid:
                    # Increment counts
                    for k_idx in triple_to_k222s[t_idx]:
                        counts[k_idx] += 1
                    current_size += 1
                    
                    # Acceptance
                    if current_size > old_size:
                        # Always accept improvement
                        active[t_idx] = True
                    else:
                        # This branch is never taken because valid add increases size
                        pass
                else:
                    # Invalid move, do nothing
                    pass
            
            # Update best
            if current_size > best_size:
                best_size = current_size
                best_active = active[:]
                best_counts = counts[:]
                
            # Cool down
            T *= alpha
            
            # If T is very low, we are effectively doing hill climbing
            # We can break early if T < T_end and no improvements for a while?
            # Or just let it run.
            
    # Construct result from best_active
    if best_active is None:
        # Fallback
        active, counts, current_size = greedy_init()
        best_active = active
        
    result = []
    for i in range(num_triples):
        if best_active[i]:
            result.append(list(all_triples[i]))
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
