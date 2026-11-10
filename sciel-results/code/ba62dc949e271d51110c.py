        free = np.ones(2401, dtype=bool)
        num_free = 2401
        S = []
        
        while num_free > 0:
            # Pick random free point
            while True:
                p_idx = random.randint(0, 2400)
                if free[p_idx]:
                    break
            
            # Find best direction
            min_count = 1000
            best_d = -1
            for d in range(50):
                count = 0
                for j in range(48):
                    q_idx = lines[p_idx, d, j]
                    if free[q_idx]:
                        count += 1
                if count < min_count:
                    min_count = count
                    best_d = d
                    if count == 0:
                        break
            
            S.append(p_idx)
            
            # Mark points on line as not free
            for j in range(48):
                q_idx = lines[p_idx, best_d, j]
                if free[q_idx]:
                    free[q_idx] = False
                    num_free -= 1
