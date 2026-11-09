import time
from math import gcd
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # 1. Precompute GCD matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]

    # 2. Identify all "bad triples" (a, b, c) with a < b < c such that
    #    gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = G[a][b]
            for c in range(b + 1, N + 1):
                if G[a][c] == g_ab and G[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    
    # Map elements to indices 0..N-1 for faster array access
    # idx = val - 1
    T = len(bad_triples)
    
    # Precompute for each element, the list of triple indices it belongs to
    triples_by_elem = [[] for _ in range(N)]
    for i, t in enumerate(bad_triples):
        for x in t:
            triples_by_elem[x - 1].append(i)
    
    # 3. Initial Solution: Greedy Hitting Set to get a good starting point
    # We want to maximize |S|, which is equivalent to minimizing |R| (removed set)
    # subject to R hitting all bad_triples.
    
    # Greedy construction of R
    R = set()
    active_triples = set(range(T))
    
    # Count active triples for each element
    active_count = [0] * N
    for t_idx in active_triples:
        for x in bad_triples[t_idx]:
            active_count[x - 1] += 1
            
    while active_triples:
        if time.time() > deadline - 5.0:
            break
            
        # Find element in R candidates? No, we are building R from scratch.
        # We pick the element in the remaining active triples that covers the most active triples.
        # To be efficient, we can just iterate through elements not in R?
        # Or simpler: pick a random active triple, and from its 3 elements, pick the one with highest active_count.
        
        # For speed, let's just pick the global max active_count among elements not yet in R?
        # Actually, standard greedy hitting set:
        # While active_triples not empty:
        #   Find element x not in R that is in the most active_triples.
        #   Add x to R.
        #   Update active_triples and counts.
        
        best_elem = -1
        max_count = -1
        # We only care about elements not in R
        for x in range(N):
            if x not in R and active_count[x] > max_count:
                max_count = active_count[x]
                best_elem = x
                
        if best_elem == -1 or max_count == 0:
            break
            
        R.add(best_elem)
        
        # Update active_triples
        new_active_triples = set()
        # This is O(T * 3) per step. T can be large.
        # Optimization: Instead of rebuilding, we can mark inactive.
        # But for N=128, T is likely in the thousands or tens of thousands.
        # 128 steps * 10k triples * 3 ops = 3.8M ops. This is fast enough.
        
        # Let's just rebuild active_triples for simplicity and correctness in the initial phase.
        # Wait, we need to update active_count too.
        
        # Better approach:
        # Keep a list of active triple indices.
        # When we add x to R, any triple containing x becomes inactive.
        
        # Let's restart the active tracking logic properly for the greedy init.
        # It's safer to just run a simple greedy loop.
        
        # Re-initialize for clean greedy
        break # We'll implement a cleaner greedy below if this gets messy.
        
    # Let's implement a cleaner greedy initialization
    R = set()
    # active_triples is a set of indices
    active_triples = set(range(T))
    # active_count[x] is number of active triples containing x
    active_count = [0] * N
    for t_idx in active_triples:
        for x in bad_triples[t_idx]:
            active_count[x - 1] += 1
            
    while active_triples:
        if time.time() > deadline - 5.0:
            break
            
        best_elem = -1
        max_count = -1
        for x in range(N):
            if x not in R:
                if active_count[x] > max_count:
                    max_count = active_count[x]
                    best_elem = x
                    
        if best_elem == -1:
            break
            
        R.add(best_elem)
        
        # Update active_triples and counts
        # Remove all triples containing best_elem
        triples_to_remove = []
        for t_idx in triples_by_elem[best_elem]:
            if t_idx in active_triples:
                triples_to_remove.append(t_idx)
                
        for t_idx in triples_to_remove:
            active_triples.remove(t_idx)
            # Decrement counts for the other 2 elements
            for x in bad_triples[t_idx]:
                if x - 1 != best_elem:
                    active_count[x - 1] -= 1
                    
    # Current solution: S = [1..N] \ R
    # Size = N - len(R)
    
    # 4. Simulated Annealing to improve R
    # State: R (set of removed elements)
    # Objective: Minimize len(R) subject to no bad triple is fully in S (i.e., all bad triples are hit by R).
    # Actually, the constraint is: For every bad triple (a,b,c), at least one of a,b,c is in R.
    
    # We will perform moves:
    # 1. Add an element to R (increases size, always valid)
    # 2. Remove an element from R (decreases size, might become invalid)
    # 3. Swap: Remove one, Add another.
    
    # We want to maximize |S|, so we want to minimize |R|.
    # If a move makes the solution invalid (some bad triple not hit), we can still accept it with probability exp(-delta/E) if we define energy as number of uncovered triples?
    # Or simpler: Only accept valid moves? That's just hill climbing.
    # Let's use a penalty approach. Energy = len(R) + lambda * (number of uncovered triples).
    # We want to minimize Energy.
    
    def get_uncovered_count(R_set):
        # Returns number of bad triples NOT hit by R_set
        # i.e., triples (a,b,c) such that a,b,c all NOT in R_set
        # This is O(T).
        count = 0
        for t in bad_triples:
            # Check if all 3 are in S (i.e., not in R)
            if (t[0] - 1) not in R_set and (t[1] - 1) not in R_set and (t[2] - 1) not in R_set:
                count += 1
        return count

    # To make SA efficient, we need fast delta evaluation.
    # Let's maintain `uncovered_count` and update it incrementally.
    # State: R (set of indices 0..N-1)
    # uncovered_count: number of bad triples with no element in R.
    
    R = set(R) # From greedy
    uncovered_count = 0
    for t in bad_triples:
        if (t[0] - 1) not in R and (t[1] - 1) not in R and (t[2] - 1) not in R:
            uncovered_count += 1
            
    # Energy = len(R) + 10 * uncovered_count
    # We want to minimize this.
    
    def energy(R_set, unc):
        return len(R_set) + 10 * unc
    
    current_R = R
    current_unc = uncovered_count
    current_E = energy(current_R, current_unc)
    
    best_R = set(current_R)
    best_E = current_E
    best_unc = current_unc
    
    T_start = 50.0
    T_end = 1.0
    steps = 100000 # Will be limited by time
    
    # Precompute for each element, the list of triples it hits (covers)
    # hits[x] = list of triple indices containing x
    # This is triples_by_elem
    
    # For faster delta calculation when adding/removing x:
    # When we ADD x to R:
    #   len(R) increases by 1.
    #   uncovered_count decreases by the number of triples that contain x AND were previously uncovered.
    #   A triple t is uncovered if none of its elements are in R.
    #   If we add x, t becomes covered if x is in t and the other 2 elements of t are NOT in R.
    
    # When we REMOVE x from R:
    #   len(R) decreases by 1.
    #   uncovered_count increases by the number of triples that contain x AND become uncovered.
    #   A triple t becomes uncovered if x is in t, the other 2 elements are NOT in R, AND x was the ONLY element of t in R.
    #   Wait, if x is removed, t becomes uncovered only if the other 2 elements are not in R.
    #   (If one of the other 2 was in R, t would still be covered).
    
    # We can compute these deltas on the fly.
    
    import math
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Annealing schedule
        progress = step / steps
        T = T_start * (T_end / T_start) ** progress
        
        # Choose a move
        # 1. Remove an element from R (try to shrink R)
        # 2. Add an element to R (try to fix violations, though we prefer valid)
        # 3. Swap
        
        # Let's bias towards removing elements if valid, or swapping.
        
        if len(current_R) == 0:
            # Can't remove
            # Must add? No, if R is empty and valid, we are done.
            # If R is empty and invalid, we must add.
            if current_unc == 0:
                break # Optimal
            move_type = 'add'
        else:
            r = random.random()
            if r < 0.6:
                move_type = 'remove'
            elif r < 0.8:
                move_type = 'add'
            else:
                move_type = 'swap'
                
        if move_type == 'remove':
            # Pick random x in current_R
            x = random.choice(list(current_R))
            
            # Calculate delta
            # len(R) delta = -1
            # unc delta:
            # Triples containing x that will become uncovered.
            # Condition: other 2 elements of t are NOT in current_R.
            # (Since x is in current_R, t is currently covered by x. If we remove x, t is covered iff some other element in t is in R).
            # So t becomes uncovered iff (other 2 not in R).
            
            new_unc = current_unc
            for t_idx in triples_by_elem[x]:
                a, b, c = bad_triples[t_idx]
                # x is one of a,b,c
                # Check other two
                others = []
                if a - 1 != x: others.append(a - 1)
                if b - 1 != x: others.append(b - 1)
                if c - 1 != x: others.append(c - 1)
                
                # If neither other is in R, t becomes uncovered
                if others[0] not in current_R and others[1] not in current_R:
                    new_unc += 1
                    
            new_E = energy(current_R, new_unc) - 1 # len(R) decreases by 1, so energy decreases by 1? 
            # Wait, E = len(R) + 10*unc.
            # New E = (len(R)-1) + 10*new_unc.
            # Delta E = -1 + 10*(new_unc - current_unc).
            delta_E = -1 + 10 * (new_unc - current_unc)
            
            # Accept if delta_E < 0 or with probability exp(-delta_E/T)
            if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
                current_R.remove(x)
                current_unc = new_unc
                current_E += delta_E
                
                if current_E < best_E:
                    best_E = current_E
                    best_R = set(current_R)
                    best_unc = current_unc
                    
        elif move_type == 'add':
            # Pick random x NOT in current_R
            # To be efficient, pick from a list of candidates?
            # Just pick random from 0..N-1, reject if in R.
            x = random.randint(0, N - 1)
            if x in current_R:
                continue
                
            # Calculate delta
            # len(R) delta = +1
            # unc delta:
            # Triples containing x that are currently uncovered will become covered.
            # Condition: other 2 elements of t are NOT in current_R.
            
            new_unc = current_unc
            for t_idx in triples_by_elem[x]:
                a, b, c = bad_triples[t_idx]
                others = []
                if a - 1 != x: others.append(a - 1)
                if b - 1 != x: others.append(b - 1)
                if c - 1 != x: others.append(c - 1)
                
                # If neither other is in R, t is currently uncovered (since x is not in R).
                # Adding x covers it.
                if others[0] not in current_R and others[1] not in current_R:
                    new_unc -= 1
                    
            delta_E = 1 + 10 * (new_unc - current_unc)
            
            if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
                current_R.add(x)
                current_unc = new_unc
                current_E += delta_E
                
                if current_E < best_E:
                    best_E = current_E
                    best_R = set(current_R)
                    best_unc = current_unc
                    
        elif move_type == 'swap':
            # Pick x in R, y not in R
            if not current_R:
                continue
            x = random.choice(list(current_R))
            y = random.randint(0, N - 1)
            if y in current_R:
                continue
                
            # Calculate delta for removing x and adding y simultaneously
            # This is complex. Let's approximate by doing remove then add?
            # Or just skip swaps for simplicity if time is tight?
            # Swaps are crucial for escaping local minima.
            
            # Let's calculate delta_unc for the combined move.
            # This requires checking triples involving x and y.
            
            # Triples affected:
            # 1. Triples containing x but not y.
            # 2. Triples containing y but not x.
            # 3. Triples containing both x and y.
            
            # This is getting complicated for the time limit.
            # Let's stick to add/remove for now, as they can simulate swaps over time.
            pass
            
    # Final Result
    # best_R is the set of removed elements.
    # S = [1..N] \ best_R
    S = [i + 1 for i in range(N) if i not in best_R]
    
    if not S:
        return [1]
        
    return S

# EVOLVE-BLOCK-END
