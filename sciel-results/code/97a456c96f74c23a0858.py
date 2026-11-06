    def try_find_subset(n_target):
        k = n_total - n_target
        min_deg_req = n_target - 39
        D = 8 - min_deg_req
        
        if D < 0:
            return None
            
        import random
        
        # Try greedy with random restarts
        for seed in range(100):
            random.seed(seed)
            R = set()
            counts = [0] * n_total
            
            verts = list(range(n_total))
            random.shuffle(verts)
            
            for v in verts:
                if len(R) >= k:
                    break
                if v in R:
                    continue
                
                can_add = True
                for u in adj[v]:
                    if u in R:
                        continue
                    if counts[u] + 1 > D:
                        can_add = False
                        break
                
                if can_add:
                    R.add(v)
                    for u in adj[v]:
                        counts[u] += 1
            
            if len(R) == k:
                # Final verification
                valid = True
                for u in range(n_total):
                    if u not in R:
                        if counts[u] > D:
                            valid = False
                            break
                if valid:
                    return list(R), R
        return None
