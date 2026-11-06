"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    N = 128
    
    # 1. Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Identify all bad triples in [1, N]
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    bad_triples_by_element = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                    bad_triples.append((a, b, c))
                    bad_triples_by_element[a].append((b, c))
                    bad_triples_by_element[b].append((a, c))
                    bad_triples_by_element[c].append((a, b))

    def is_valid_insertion(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for y, z in bad_triples_by_element[x]:
            if y in S and z in S:
                return False
        return True

    def greedy_remove_and_reinsert(seed):
        random.seed(seed)
        S = set(range(1, N + 1))
        
        # active_count[x] = number of bad triples containing x where all 3 elements are in S
        active_count = [0] * (N + 1)
        for x in range(1, N + 1):
            active_count[x] = len(bad_triples_by_element[x])
        
        removed = []
        
        # Phase 1: Greedy Removal
        while True:
            max_count = -1
            victim = -1
            for x in S:
                if active_count[x] > max_count:
                    max_count = active_count[x]
                    victim = x
            
            if victim == -1 or max_count == 0:
                break
            
            S.remove(victim)
            removed.append(victim)
            
            for y, z in bad_triples_by_element[victim]:
                if y in S and z in S:
                    active_count[y] -= 1
                    active_count[z] -= 1
        
        # Phase 2: Random Re-insertion
        random.shuffle(removed)
        
        for x in removed:
            if is_valid_insertion(x, S):
                S.add(x)
                removed.remove(x) # Keep track of who is still out? 
                # Actually, we just want to return S. 
                # But for shake-ups, we need to know who is out.
                # Let's just return S and the list of those NOT in S.
        
        return S, list(set(range(1, N + 1)) - S)

    # Initial best
    best_S, best_removed = greedy_remove_and_reinsert(0)
    best_size = len(best_S)
    
    # Local Search with Shake-ups
    # We will try to improve by shaking out some elements and re-inserting.
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    while time.time() - start_time < time_limit:
        current_S, current_removed = best_S, best_removed
        
        # Shake: Remove k random elements from current_S
        k = random.randint(2, 5)
        if len(current_S) < k:
            break
        to_shake = random.sample(list(current_S), k)
        
        new_S = set(current_S)
        new_removed = list(current_removed)
        
        for x in to_shake:
            new_S.remove(x)
            new_removed.append(x)
        
        # Shuffle the removed pool
        random.shuffle(new_removed)
        
        # Re-insertion phase
        for x in new_removed:
            if is_valid_insertion(x, new_S):
                new_S.add(x)
                # We don't strictly need to remove from new_removed if we just check membership later,
                # but it helps if we want to know who is still out.
                # However, is_valid_insertion is O(degree).
                # We can just build the final set.
        
        # Note: The loop above adds to new_S, but new_removed still contains them.
        # We need to calculate the actual set of removed elements for the next iteration.
        actual_removed = list(set(range(1, N + 1)) - new_S)
        
        if len(new_S) > best_size:
            best_S = new_S
            best_removed = actual_removed
            best_size = len(new_S)
            
            # Optional: If we hit the theoretical max or a very high number, we could stop,
            # but we don't know the max.
            
    return list(best_S)
# EVOLVE-BLOCK-END
