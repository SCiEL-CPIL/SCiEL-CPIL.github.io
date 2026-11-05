import random
import time

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    N = 192
    MAX_K = 48
    
    def greedy_with_residues(order):
        S = []
        in_S = [False] * (N + 1)
        # bad_residues[a] is a set of residues r such that exists y,z in S\{a} with (y+z) % a == -r % a
        # i.e., x % a == r implies x + y + z is divisible by a.
        bad_residues = [set() for _ in range(N + 1)]
        
        for x in order:
            if in_S[x]:
                continue
            
            # Check 2: Does any a in S divide x + y + z?
            # This is equivalent to x % a being in bad_residues[a]
            valid = True
            for a in S:
                if x % a in bad_residues[a]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 1: Does x divide y + z + w for any y,z,w in S?
            # Only need to check if len(S) >= 3
            if len(S) >= 3:
                S_list = S
                n = len(S_list)
                # Iterate combinations of 3
                for i in range(n):
                    y = S_list[i]
                    for j in range(i + 1, n):
                        z = S_list[j]
                        for k in range(j + 1, n):
                            w = S_list[k]
                            if (y + z + w) % x == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
            
            if valid:
                # Add x to S
                S.append(x)
                in_S[x] = True
                
                # Update bad_residues
                # For existing a in S (excluding x, but x is new so S currently has x appended? 
                # No, we should update based on S_old.
                # Let's say S_old is the list before appending x.
                # We appended x to S. So S[-1] is x.
                # We need to update bad_residues for all a in S_old.
                
                S_old = S[:-1]
                
                # 1. Update for existing a
                for a in S_old:
                    # New pairs (x, y) where y in S_old and y != a
                    for y in S_old:
                        if y == a:
                            continue
                        # We need to add residue r such that (x + y) % a == -r % a
                        # r = (-(x + y)) % a
                        r = (-(x + y)) % a
                        bad_residues[a].add(r)
                
                # 2. Initialize bad_residues[x]
                # Pairs (y, z) in S_old
                if len(S_old) >= 2:
                    for i in range(len(S_old)):
                        y = S_old[i]
                        for j in range(i + 1, len(S_old)):
                            z = S_old[j]
                            r = (-(y + z)) % x
                            bad_residues[x].add(r)
                
                if len(S) >= MAX_K:
                    break
        
        return S

    best_size = 0
    best_set = []
    
    # Run multiple greedy attempts with different orders
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        
        # Strategy: Mix random and sorted orders
        if attempts % 3 == 0:
            order = list(range(N, 0, -1))
            random.shuffle(order)
        elif attempts % 3 == 1:
            order = list(range(N, 0, -1))
        else:
            # Try to favor numbers that are "harder" to place? 
            # Or just random
            order = list(range(1, N + 1))
            random.shuffle(order)
        
        S_curr = greedy_with_residues(order)
        
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_set = S_curr
            if best_size == MAX_K:
                break
        
        # Optional: Local search on best_set
        # Try swapping elements to see if we can add more?
        # Given time, just do more greedy restarts.
        
    return best_set
