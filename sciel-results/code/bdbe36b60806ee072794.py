import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    n = 4096
    
    best_set = []
    best_size = 0
    
    def greedy_fill(skeleton, order_func):
        nonlocal best_set, best_size
        
        S = sorted(skeleton)
        if not S:
            S = []
        
        # Validate skeleton
        used_sums = set()
        pair_sums = set()
        
        # Add skeleton sums
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                pair_sums.add(S[i] + S[j])
        
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s_val = S[i] + S[j] + S[l]
                    if s_val in used_sums:
                        return # Invalid skeleton
                    used_sums.add(s_val)
                    
        S_set = set(S)
        candidates = [x for x in range(1, n + 1) if x not in S_set]
        candidates = order_func(candidates)
        
        current_S = list(S)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Check conflicts
            # 1. 3x
            if 3 * x in used_sums:
                continue
            
            # 2. 2x + s
            conflict = False
            for s in current_S:
                if (2 * x + s) in used_sums:
                    conflict = True
                    break
            if conflict:
                continue
            
            # 3. x + p
            for p in pair_sums:
                if (x + p) in used_sums:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Add x
            current_S.append(x)
            
            # Update used_sums
            used_sums.add(3 * x)
            for s in current_S[:-1]: # Exclude x itself from the 's' loop? 
                                     # Wait, 2x+s where s is from OLD S.
                                     # The loop above used current_S before append?
                                     # No, I appended x.
                                     # Let's re-structure.
                pass
            
            # Correct Update Logic:
            # Old S is current_S[:-1]
            old_S = current_S[:-1]
            
            # New sums:
            # 3x
            used_sums.add(3 * x)
            
            # 2x + s for s in old_S
            for s in old_S:
                used_sums.add(2 * x + s)
                
            # x + p for p in old_pair_sums
            # Note: pair_sums currently contains pairs from old_S.
            # We need to iterate over the OLD pair_sums.
            # But we are going to update pair_sums.
            # Let's store old_pair_sums
            # Actually, we can just iterate over the current pair_sums before updating it.
            
            for p in pair_sums:
                used_sums.add(x + p)
                
            # Update pair_sums
            # New pairs: x + s for s in old_S, and x + x
            for s in old_S:
                pair_sums.add(x + s)
            pair_sums.add(2 * x)
            
            # Update best
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_set = list(current_S)

    # Order functions
    def order_asc(cands):
        return sorted(cands)
    
    def order_desc(cands):
        return sorted(cands, reverse=True)
    
    def order_random(cands):
        c = cands[:]
        random.shuffle(c)
        return c

    # Skeletons
    def get_skeletons():
        skels = []
        # Powers of 3
        p = 1
        while p <= n:
            skels.append(p)
            p *= 3
        skels.append(list(skels)) # No, this is one skeleton
        
        # Let's list distinct skeletons
        skeletons = []
        
        # Powers of 3
        p3 = [1]
        while p3[-1] * 3 <= n:
            p3.append(p3[-1] * 3)
        skeletons.append(p3)
        
        # Powers of 4
        p4 = [1]
        while p4[-1] * 4 <= n:
            p4.append(p4[-1] * 4)
        skeletons.append(p4)
        
        # Powers of 5
        p5 = [1]
        while p5[-1] * 5 <= n:
            p5.append(p5[-1] * 5)
        skeletons.append(p5)
        
        # Random Geometric
        for _ in range(5):
            base = random.uniform(2.5, 5.0)
            skel = [1]
            while skel[-1] * base <= n:
                next_val = int(skel[-1] * base + random.uniform(-0.5, 0.5))
                if next_val > skel[-1] and next_val <= n:
                    skel.append(next_val)
                else:
                    break
            if len(skel) > 1:
                skeletons.append(skel)
                
        # Empty
        skeletons.append([])
        
        return skeletons

    skeletons = get_skeletons()
    
    # Run strategies
    # 1. Ascending
    for skel in skeletons:
        if time.time() > deadline: break
        greedy_fill(skel, order_asc)
        
    # 2. Descending
    for skel in skeletons:
        if time.time() > deadline: break
        greedy_fill(skel, order_desc)
        
    # 3. Random
    # Run many random restarts
    while time.time() < deadline:
        skel = random.choice(skeletons)
        greedy_fill(skel, order_random)

    return best_set
