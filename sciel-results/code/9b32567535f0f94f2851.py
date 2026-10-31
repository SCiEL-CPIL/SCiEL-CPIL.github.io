<<<<<<< SEARCH
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 8:
            break
        
        t_start = time.time()
        x_opt, s_opt = _optimize_from_start(x0, deadline, best_score)
        _update_best(x_opt.reshape(N, 2))
        
        elapsed = time.time() - t_start
        if i % 5 == 0:
            print(f"Init {i}: score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
=======
    # Optimize from each start using coordinate descent primarily
    # This is faster and more reliable than generic minimizers for this metric
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        points_start = x0.reshape(N, 2)
        
        # 1. Quick coordinate descent
        if time.time() < deadline - 5:
            cd_pts, cd_s = _coordinate_descent_polish(points_start, deadline, max_iter=100)
            _update_best(cd_pts)
        
        # 2. If score is good, try SA for refinement
        if time.time() < deadline - 3 and best_score > 0.01:
            sa_pts, sa_s = _simulated_annealing_polish(cd_pts, deadline, t_start=0.05, t_end=0.001, max_iter=50)
            _update_best(sa_pts)
            
        if i % 5 == 0:
            print(f"Init {i}: score={best_score:.6f}", flush=True)
>>>>>>> REPLACE

