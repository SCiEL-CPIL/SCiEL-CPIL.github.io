import random
import time

def construct():
    n = 128
    target_start = 128
    target_end = 256
    
    def build_counts(A_set):
        counts = [0] * (2 * n + 1)
        A_list = list(A_set)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                counts[a + b] += 1
        return counts

    def get_metrics(counts):
        min_c = float('inf')
        max_c = 0
        for s in range(target_start, target_end + 1):
            c = counts[s]
            if c < min_c:
                min_c = c
            if c > max_c:
                max_c = c
        return min_c, max_c

    best_A = None
    best_peak = float('inf')
    
    start_time = time.time()
    timeout = 55.0
    
    for attempt in range(10):
        if time.time() - start_time > timeout:
            break
        
        size = random.choice([16, 17, 18, 19, 20])
        A = set(random.sample(range(n + 1), size))
        
        counts = build_counts(A)
        min_c, max_c = get_metrics(counts)
        
        no_improve = 0
        while time.time() - start_time < timeout and no_improve < 500:
            a = random.choice(list(A))
            candidates = [x for x in range(n + 1) if x not in A]
            if not candidates:
                break
            b = random.choice(candidates)
            
            # Incremental update
            # Remove a
            # Pairs involving a: (a, a) and (a, x) for x in A, x != a
            # Note: In our count representation, (x, a) is counted when x < a or x == a?
            # We use i <= j in build_counts. So (a, b) is counted once.
            # If we remove a, we subtract:
            # 1. count[2a] -= 1
            # 2. for x in A: if x != a: count[a+x] -= 1
            
            # Add b
            # 1. count[2b] += 1
            # 2. for x in A: if x != b: count[b+x] += 1
            # Note: A currently contains b? No, we haven't added it yet.
            # So we iterate over current A (which has a removed? No, we are simulating).
            
            # Let's do it carefully:
            # Current A has a. We want to swap a with b.
            # 1. Remove a from A.
            #    Subtract contributions of a with all other elements in A (including a).
            #    For x in A:
            #       if x == a: counts[2a] -= 1
            #       else: counts[a+x] -= 1
            #    (This works because we iterate over the set BEFORE removal? No, if we remove first, we lose a.
            #    So iterate over list(A) first, then remove.)
            
            A_list = list(A)
            for x in A_list:
                if x == a:
                    counts[2 * a] -= 1
                else:
                    counts[a + x] -= 1
            A.remove(a)
            
            # 2. Add b to A.
            #    Add contributions of b with all elements in A (which does not contain b yet).
            #    For x in A:
            #       counts[b+x] += 1
            #    counts[2b] += 1
            
            for x in A:
                counts[b + x] += 1
            counts[2 * b] += 1
            A.add(b)
            
            # 3. Evaluate
            new_min, new_max = get_metrics(counts)
            
            if new_min > 0 and new_max < max_c:
                max_c = new_max
                min_c = new_min
                no_improve = 0
                if max_c < best_peak:
                    best_peak = max_c
                    best_A = set(A)
            else:
                # Revert
                # Remove b
                A_list = list(A)
                for x in A_list:
                    if x == b:
                        counts[2 * b] -= 1
                    else:
                        counts[b + x] -= 1
                A.remove(b)
                
                # Add a
                for x in A:
                    counts[a + x] += 1
                counts[2 * a] += 1
                A.add(a)
                
                no_improve += 1
                
    if best_A is None:
        best_A = set(range(64, 129))
        
    return list(best_A)
