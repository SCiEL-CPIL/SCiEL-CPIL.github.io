def _local_search_greedy(h, time_limit=5.0, use_sa=False, T0=0.01):
    """
    Greedy local search with random kicks.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() - start_time < time_limit:
        # Phase 1: Greedy Single-Bit Flip
        best_flip_idx = -1
        best_flip_val = current_val
        
        # Vectorized evaluation of all single-bit flips?
        # We can't easily vectorize the max sidelobe calc for all N flips at once without N FFTs.
        # But N=400, so 400 FFTs is fast.
        
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            old_h_i = h[i]
            h[i] = -old_h_i
            
            ac_new = _compute_autocorr(h)
            val_new = _max_sidelobe(ac_new)
            
            if val_new < best_flip_val:
                best_flip_val = val_new
                best_flip_idx = i
                best_flip_ac = ac_new
            else:
                h[i] = old_h_i
        
        if best_flip_idx != -1:
            # Apply the best flip
            h[best_flip_idx] = -h[best_flip_idx]
            ac = best_flip_ac
            current_val = best_flip_val
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
            no_improve_count = 0
        else:
            # No single bit flip improves
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a random kick (SA style)
                if use_sa:
                    # Pick a random bit to flip regardless of cost
                    i = np.random.randint(N)
                    h[i] = -h[i]
                    ac = _compute_autocorr(h)
                    current_val = _max_sidelobe(ac)
                    no_improve_count = 0
                    # Cool down? We don't have T here, just accept any kick to escape
                else:
                    break
            
    return best_h, best_val
