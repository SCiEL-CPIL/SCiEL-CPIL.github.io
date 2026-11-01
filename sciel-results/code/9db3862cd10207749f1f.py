    # Try lower ranks
    for target_rank in [41, 40, 39, 38, 37]:
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        # Fewer attempts to save time
        for attempt in range(10):
            if time.time() - START_TIME > DEADLINE - 5:
                break
            # ...
