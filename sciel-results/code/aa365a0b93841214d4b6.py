    for _ in range(10000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_triples)
        current_set = set()
        adj = [[set() for _ in range(n)] for __ in range(n)]
        
        for i, t in enumerate(all_triples):
            # Pruning
            if len(current_set) + (len(all_triples) - i) <= best_count:
                break
                
            u, v, w = t
            # ... rest of logic
