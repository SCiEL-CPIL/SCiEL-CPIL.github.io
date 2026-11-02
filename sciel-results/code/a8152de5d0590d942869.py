import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for 120s limit
    
    def calc_score(U):
        if not U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        ratio = len(diffs) / len(sums)
        if ratio <= 1.0:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    best_score = 0.0
    best_U = [0]
    
    # Run multiple restarts
    # Estimate how many restarts fit in 115s
    # Each restart ~ 200 steps * calc_score time.
    # calc_score for N=10 is ~100 ops. Very fast.
    # So we can do many restarts.
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Initialize U
        if restart_count == 1:
            U = [0, 1, 3, 7, 12]
        else:
            # Random initialization
            size = random.randint(3, 10)
            U = [0] + random.sample(range(1, 50), size-1)
            U.sort()
        
        current_U = list(U)
        current_score = calc_score(current_U)
        
        if current_score > best_score:
            best_score = current_score
            best_U = list(current_U)
        
        # Local search
        # Adaptive step count based on remaining time
        # Just do a fixed number, loop handles the rest
        max_steps = 100
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Generate neighbor
            op = random.choice(['add', 'remove', 'replace'])
            new_U = list(current_U)
            
            if op == 'add':
                if len(new_U) > 20: # Cap size
                    continue
                new_val = random.randint(1, 50)
                if new_val not in new_U:
                    new_U.append(new_val)
                    new_U.sort()
                else:
                    continue
                    
            elif op == 'remove':
                if len(new_U) <= 2:
                    continue
                removable = [x for x in new_U if x != 0]
                if not removable:
                    continue
                idx = random.choice(removable)
                new_U.remove(idx)
                
            elif op == 'replace':
                if len(new_U) <= 2:
                    continue
                removable = [x for x in new_U if x != 0]
                if not removable:
                    continue
                old_val = random.choice(removable)
                new_val = old_val + random.randint(-5, 5)
                if new_val >= 0 and new_val not in new_U:
                    new_U.remove(old_val)
                    new_U.append(new_val)
                    new_U.sort()
                else:
                    continue
                    
            new_score = calc_score(new_U)
            
            if new_score > current_score:
                current_U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(current_U)
        
    return best_U
