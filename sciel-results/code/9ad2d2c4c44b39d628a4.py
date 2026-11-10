<<<<<<< SEARCH
        # SA parameters: start hot, cool down
        # For planted SAT, a good starting temp is around 1-5
        t_start = 3.0 + restart * 0.2
        t_end = 0.01
        min_steps = 2000
        max_steps = 50000  # safety cap per restart
        
        a, u = sa_search(a, t_start, t_end, min_steps, max_steps)
        update_best(a)
        if u == 0:
            return a
        
        restart += 1
        if restart > 300:
            break
=======
        # SA parameters: start hot, cool down
        t_start = 3.0
        t_end = 0.01
        # Reduce max_steps to allow more restarts if needed, or keep high but ensure time check works
        # 10000 steps should be fast enough in Python if optimized? 
        # Let's try 5000 steps per restart to be safe on time.
        min_steps = 1000
        max_steps = 5000
        
        a, u = sa_search(a, t_start, t_end, min_steps, max_steps)
        update_best(a)
        if u == 0:
            return a
        
        restart += 1
        if restart > 1000: # Allow many restarts if they are fast
            break
>>>>>>> REPLACE

