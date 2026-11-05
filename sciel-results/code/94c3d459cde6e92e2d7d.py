def greedy_cover_construct():
    S = []
    S_bits = 0
    covered = 0
    candidates = list(range(N2))
    
    while True:
        random.shuffle(candidates)
        best_p = -1
        best_gain = -1
        best_new_covered = 0
        
        for p in candidates:
            if (S_bits >> p) & 1:
                continue
            # ... rest of logic
