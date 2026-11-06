# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n_max = 4096
    max_sum = 3 * n_max  # 12288
    sum_size = max_sum + 1  # 12289
    
    def can_add(elements, x, used_sums):
        """Check if x can be added to elements without creating sum collisions."""
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                s = x + elements[i] + elements[j]
                if used_sums[s]:
                    return False, []
        for i in range(len(elements)):
            s = x + x + elements[i]
            if used_sums[s]:
                return False, []
        s = x + x + x
        if used_sums[s]:
            return False, []
        # Collect new sums
        new_sums = []
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                new_sums.append(x + elements[i] + elements[j])
            new_sums.append(x + x + elements[i])
        new_sums.append(x + x + x)
        return True, new_sums
    
    def greedy_build(initial_elements, used_sums):
        """Greedily extend the set."""
        elements = initial_elements[:]
        used = used_sums[:]  # copy
        
        for x in range(elements[-1] + 1, n_max + 1):
            if time.time() > deadline:
                break
            ok, new_sums = can_add(elements, x, used)
            if ok:
                for s in new_sums:
                    used[s] = 1
                elements.append(x)
        return elements
    
    best = []
    
    # Try several different initial seeds
    initial_seeds = [
        [1, 2, 5],
        [1, 3, 7],
        [1, 4, 10],
        [2, 5, 13],
        [1, 2, 6],
        [1, 3, 8],
        [2, 3, 7],
        [1, 5, 12],
    ]
    
    for seed in initial_seeds:
        if time.time() > deadline:
            break
        used = bytearray(sum_size)
        # Initialize sums for the seed
        for i in range(len(seed)):
            for j in range(i, len(seed)):
                for k in range(j, len(seed)):
                    s = seed[i] + seed[j] + seed[k]
                    if s <= max_sum:
                        used[s] = 1
        result = greedy_build(seed, used)
        if len(result) > len(best):
            best = result
    
    # Random restarts
    for _ in range(100):
        if time.time() > deadline:
            break
        # Random initial set of 3
        while True:
            a = random.randint(1, 30)
            b = random.randint(a + 1, 60)
            c = random.randint(b + 1, 100)
            seed = [a, b, c]
            used = bytearray(sum_size)
            ok = True
            for i in range(3):
                for j in range(i, 3):
                    for k in range(j, 3):
                        s = seed[i] + seed[j] + seed[k]
                        if used[s]:
                            ok = False
                            break
                        used[s] = 1
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                break
        
        result = greedy_build(seed, used)
        if len(result) > len(best):
            best = result
    
    return best
# EVOLVE-BLOCK-END
