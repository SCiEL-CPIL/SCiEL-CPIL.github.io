<<<<<<< SEARCH
    for target_rank in range(16, rank_28):
        if time.time() > deadline:
            break
        
        found = False
        num_trials = 100 if target_rank <= 20 else 50
=======
    for target_rank in range(16, rank_28):
        if time.time() > deadline:
            break
        
        found = False
        # More trials for lower ranks, with time-aware budgeting
        remaining_time = deadline - time.time()
        if target_rank <= 18:
            num_trials = min(150, int(remaining_time / 1.5))
        elif target_rank <= 22:
            num_trials = min(80, int(remaining_time / 2.0))
        else:
            num_trials = min(40, int(remaining_time / 3.0))
        num_trials = max(num_trials, 5)
>>>>>>> REPLACE

