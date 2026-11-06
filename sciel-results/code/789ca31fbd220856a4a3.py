            # Inside the loop
            if best_len == prev_best_len:
                no_improve_count += 1
            else:
                no_improve_count = 0
                prev_best_len = best_len
            
            if no_improve_count > 50:
                # Stuck, restart
                beam = []
                no_improve_count = 0
