import time

def construct():
        t_start = time.time()
        deadline = t_start + 55.0
        
        def get_sums(S):
            n = len(S)
            # Max sum is 256 * n
            max_sum = 256 * n
            # dp[k] is a set of sums
            sums = [set() for _ in range(n + 1)]
            sums[0].add(0)
            for x in S:
                for k in range(n, 0, -1):
                    if sums[k-1]:
                        # Optimization: use list comprehension or set comprehension
                        new_sums = {s + x for s in sums[k-1]}
                        sums[k].update(new_sums)
            return sums

        def is_valid(S):
            sums = get_sums(S)
            n = len(S)
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    if sums[i] & sums[j]:
                        return False
            return True

        def greedy_build():
            S = []
            sums = [set() for _ in range(61)] # Max size 60
            sums[0].add(0)
            current_max_k = 0
            
            while True:
                if time.time() > deadline:
                    break
                if len(S) >= 60:
                    break
                
                # Compute forbidden set for the next element
                # If we add x, new sums for k are old_sums[k] U (old_sums[k-1] + x)
                # We need (old_sums[k-1] + x) to be disjoint from all sums[j] for j != k
                # AND (old_sums[k-1] + x) to be disjoint from (old_sums[j-1] + x) for j != k?
                # No, the new sums for j are old_sums[j] U (old_sums[j-1] + x).
                # We already checked old_sums are disjoint.
                # We need to check:
                # 1. (old_sums[k-1] + x) vs old_sums[j] for j != k
                # 2. (old_sums[k-1] + x) vs (old_sums[j-1] + x) for j != k
                
                # Let's just try all x in 1..256
                best_x = None
                # To be faster, we can check candidates in increasing order
                for x in range(1, 257):
                    if x in S:
                        continue
                    
                    valid = True
                    # Check condition 1
                    for k in range(1, len(S) + 1):
                        if not sums[k-1]:
                            continue
                        new_part = {s + x for s in sums[k-1]}
                        # Check against all j != k
                        for j in range(len(S) + 1):
                            if j == k:
                                continue
                            if new_part & sums[j]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Check condition 2: New parts vs New parts
                    # new_part_k = sums[k-1] + x
                    # new_part_j = sums[j-1] + x
                    # Intersection implies sums[k-1] + x = sums[j-1] + x => sums[k-1] = sums[j-1]
                    # Since sums are disjoint for k-1 != j-1, this is empty.
                    # Wait, if k-1 == j-1, then k=j, which is skipped.
                    # So condition 2 is automatically satisfied if the original set was valid!
                    
                    if valid:
                        best_x = x
                        break
                
                if best_x is None:
                    break
                
                S.append(best_x)
                # Update sums
                for k in range(len(S), 0, -1):
                    if sums[k-1]:
                        sums[k].update({s + best_x for s in sums[k-1]})
                
            return S

        # Run multiple greedy attempts with different orders?
        # The deterministic greedy above always picks the smallest valid x.
        # This might get stuck.
        # Let's try random shuffles of the candidate list.
        
        best_set = []
        
        import random
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            S = []
            sums = [set() for _ in range(61)]
            sums[0].add(0)
            
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                if len(S) >= 60:
                    break
                
                valid = True
                for k in range(1, len(S) + 1):
                    if not sums[k-1]:
                        continue
                    new_part = {s + x for s in sums[k-1]}
                    for j in range(len(S) + 1):
                        if j == k:
                            continue
                        if new_part & sums[j]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    S.append(x)
                    for k in range(len(S), 0, -1):
                        if sums[k-1]:
                            sums[k].update({s + x for s in sums[k-1]})
            
            if len(S) > len(best_set):
                best_set = S[:]
                
        return best_set
