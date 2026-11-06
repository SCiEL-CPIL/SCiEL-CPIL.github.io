import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin
    
    N = 256
    
    # Precomputation
    # 3-APs: (a, b, c) such that a+c=2b
    three_aps = []
    threes_involving = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx = len(three_aps)
            three_aps.append((a, b, c))
            threes_involving[a].append(idx)
            threes_involving[b].append(idx)
            threes_involving[c].append(idx)
            
    # 4-APs: (a, b, c, d) such that a+d=b+c=2b... a, a+d, a+2d, a+3d
    four_aps = []
    fours_involving = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = len(four_aps)
            four_aps.append((a, b, c, e))
            fours_involving[a].append(idx)
            fours_involving[b].append(idx)
            fours_involving[c].append(idx)
            fours_involving[e].append(idx)

    # Helper functions using bitsets
    # S_bits: bit i is 1 if i in S
    # current_3ap: count of 3-APs in S
    
    def gain_add(S_bits, x):
        gain = 0
        for idx in threes_involving[x]:
            a, b, c = three_aps[idx]
            # Check if the other two are in S_bits
            # We need to identify the other two
            if a == x:
                if ((S_bits >> b) & 1) and ((S_bits >> c) & 1):
                    gain += 1
            elif b == x:
                if ((S_bits >> a) & 1) and ((S_bits >> c) & 1):
                    gain += 1
            else: # c == x
                if ((S_bits >> a) & 1) and ((S_bits >> b) & 1):
                    gain += 1
        return gain

    def can_add(S_bits, x):
        for idx in fours_involving[x]:
            a, b, c, d = four_aps[idx]
            if a == x:
                if ((S_bits >> b) & 1) and ((S_bits >> c) & 1) and ((S_bits >> d) & 1):
                    return False
            elif b == x:
                if ((S_bits >> a) & 1) and ((S_bits >> c) & 1) and ((S_bits >> d) & 1):
                    return False
            elif c == x:
                if ((S_bits >> a) & 1) and ((S_bits >> b) & 1) and ((S_bits >> d) & 1):
                    return False
            else: # d == x
                if ((S_bits >> a) & 1) and ((S_bits >> b) & 1) and ((S_bits >> c) & 1):
                    return False
        return True

    def greedy_construct():
        S_bits = 0
        current_3ap = 0
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline:
                break
            if can_add(S_bits, x):
                S_bits |= (1 << x)
                current_3ap += gain_add(S_bits, x)
        return S_bits, current_3ap

    def local_search(S_bits, current_3ap, max_iter=5000):
        best_S = S_bits
        best_3ap = current_3ap
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try Add
            # Pick a random x not in S
            # To do this fast, we can pick a random number and check, or maintain a list of not-in-S
            # Maintaining a list is O(1) removal if we use swap-remove
            # But let's just pick random x from 1..N and check if not in S
            
            x = random.randint(1, N)
            if not (S_bits >> x) & 1:
                if can_add(S_bits, x):
                    new_3ap = current_3ap + gain_add(S_bits, x)
                    if new_3ap >= best_3ap:
                        S_bits |= (1 << x)
                        current_3ap = new_3ap
                        if new_3ap > best_3ap:
                            best_S = S_bits
                            best_3ap = new_3ap
            
            # Try Remove
            # Pick a random x in S
            # We can iterate through bits or pick random and check
            # To pick random from S efficiently, we can just pick random x and check
            # If x in S, remove it
            x = random.randint(1, N)
            if (S_bits >> x) & 1:
                # Calculate loss
                loss = gain_add(S_bits, x) # This is the number of 3-APs destroyed
                new_3ap = current_3ap - loss
                # Accept if it allows future improvements? 
                # Hill climbing: accept if new_3ap > best_3ap? No, removing usually decreases 3ap.
                # We should only remove if it helps us add more later, or if we are stuck.
                # Let's do: Accept if random() < exp(-delta/T) or simply accept if it leads to a better state after a subsequent add.
                # Simpler: Just try Swap.
                pass
            
            # Try Swap
            # Pick x_out in S, x_in not in S
            x_out = random.randint(1, N)
            if (S_bits >> x_out) & 1:
                x_in = random.randint(1, N)
                if not (S_bits >> x_in) & 1:
                    # Check if x_in can be added to S \ {x_out}
                    S_temp = S_bits & ~(1 << x_out)
                    if can_add(S_temp, x_in):
                        # Calculate change in 3ap
                        # Remove x_out: lose gain_add(S_bits, x_out)
                        # Add x_in: gain gain_add(S_temp, x_in)
                        # Note: gain_add checks against current S.
                        # When we remove x_out, the "others" for x_out's 3-APs are still in S_temp.
                        # So loss = gain_add(S_bits, x_out) is correct.
                        # When we add x_in to S_temp, the "others" for x_in's 3-APs are in S_temp.
                        # So gain = gain_add(S_temp, x_in) is correct.
                        
                        loss = gain_add(S_bits, x_out)
                        gain = gain_add(S_temp, x_in)
                        new_3ap = current_3ap - loss + gain
                        
                        if new_3ap > best_3ap:
                            S_bits = S_temp | (1 << x_in)
                            current_3ap = new_3ap
                            best_S = S_bits
                            best_3ap = new_3ap
                        elif new_3ap == best_3ap and random.random() < 0.1:
                            S_bits = S_temp | (1 << x_in)
                            current_3ap = new_3ap
                            best_S = S_bits
                            best_3ap = new_3ap

        return best_S, best_3ap

    # Run multiple trials
    best_S_global = 0
    best_3ap_global = 0
    
    for trial in range(20): # Adjust based on time
        if time.time() > deadline:
            break
        
        # Greedy start
        S_bits, current_3ap = greedy_construct()
        
        # Local search
        S_bits, current_3ap = local_search(S_bits, current_3ap, max_iter=2000)
        
        if current_3ap > best_3ap_global:
            best_3ap_global = current_3ap
            best_S_global = S_bits

    # Convert bitset to list
    result_set = []
    for i in range(1, N + 1):
        if (best_S_global >> i) & 1:
            result_set.append(i)
            
    if not result_set:
        result_set = [1]
        
    return sorted(result_set)
