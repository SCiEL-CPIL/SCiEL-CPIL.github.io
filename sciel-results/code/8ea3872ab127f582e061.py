def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_set = [0, 1, 4, 7]
    best_score = score_set(best_set)
    
    while time.time() < deadline:
        # Random restart
        # Start with small random set
        n = random.randint(3, 6)
        # Ensure 0 is in set
        U = [0]
        for _ in range(n - 1):
            U.append(random.randint(1, 20))
        U = sorted(list(set(U)))
        
        current_set = U
        current_score = score_set(current_set)
        
        T = 0.05
        T_min = 0.001
        
        # Run SA for a fixed number of steps or time
        steps = 5000
        for i in range(steps):
            if time.time() > deadline:
                break
            
            neighbor = current_set[:]
            op = random.choice(['add', 'remove', 'move'])
            
            if op == 'add':
                max_u = max(neighbor)
                new_val = random.randint(1, max_u + 5)
                if new_val not in neighbor:
                    neighbor.append(new_val)
            elif op == 'remove':
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    neighbor.pop(idx)
            else:
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    new_val = old_val + random.randint(-5, 5)
                    if new_val >= 0 and new_val not in neighbor:
                        neighbor[idx] = new_val
                    else:
                        continue
                        
            neighbor.sort()
            new_score = score_set(neighbor)
            
            if new_score > current_score:
                current_set = neighbor
                current_score = new_score
            else:
                delta = new_score - current_score
                if delta < 0:
                    p = math.exp(delta / T)
                    if random.random() < p:
                        current_set = neighbor
                        current_score = new_score
                else:
                    current_set = neighbor
                    current_score = new_score
            
            T = max(T_min, T * 0.999)
            
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]

    return best_set
