import random
import time

def construct():
    start_time = time.time()
    deadline = 55.0
    
    # Precompute divisors up to max possible sum
    # Max sum = 256 * 30 = 7680
    max_sum = 7680
    divisors = [set() for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1, i):
            divisors[j].add(i)
            
    best_set = []
    
    # Try different candidate pools
    # Pool 1: All 1..256
    # Pool 2: 50..256
    # Pool 3: 100..256
    
    pools = [
        list(range(1, 257)),
        list(range(50, 257)),
        list(range(100, 257))
    ]
    
    for pool in pools:
        for _ in range(200):
            if time.time() - start_time > deadline:
                break
                
            candidates = list(pool)
            random.shuffle(candidates)
            
            current_set = []
            current_sums = set()
            
            for x in candidates:
                if x in current_sums:
                    continue
                
                # Check x against current_sums using divisors
                conflict = False
                for s in current_sums:
                    if s in divisors[x] or x in divisors[s]:
                        conflict = True
                        break
                if conflict:
                    continue
                
                new_sums_list = [x]
                valid = True
                for s in current_sums:
                    y = x + s
                    if y in current_sums:
                        valid = False
                        break
                    new_sums_list.append(y)
                
                if not valid:
                    continue
                
                # Check new_sums_list against current_sums
                for y in new_sums_list:
                    for s in current_sums:
                        if s in divisors[y] or y in divisors[s]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check new_sums_list against itself
                n = len(new_sums_list)
                for i in range(n):
                    y1 = new_sums_list[i]
                    for j in range(i+1, n):
                        y2 = new_sums_list[j]
                        if y1 in divisors[y2] or y2 in divisors[y1]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current_set.append(x)
                    current_sums.update(new_sums_list)
                    
                    if len(current_set) > len(best_set):
                        best_set = list(current_set)
                        
    return list(best_set)
