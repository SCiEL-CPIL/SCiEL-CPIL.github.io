        # 1. Try to cover u directly
        # Only try primes that do NOT currently cover u
        for i in range(n_primes):
            p = primes_arr[i]
            # Check if current residue covers u
            # u % p == best_res[i]
            if u % p == best_res[i]:
                continue
                
            r_new = u % p
            # r_new is different from best_res[i] by construction
            
            old_r = best_res[i]
            best_res[i] = r_new
            
            new_L = compute_L(best_res)
            
            if new_L > L:
                L = new_L
                improved = True
                break
            else:
                best_res[i] = old_r
