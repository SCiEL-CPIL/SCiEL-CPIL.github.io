<<<<<<< SEARCH
        best_code = None
        best_len = float('inf')

        # Run as many restarts as possible
        i = 0
        while time.time() < deadline - 20:
            code = greedy_cover(i)
            refined = refine(code)
            
            # Short local search
            ls_time = min(deadline - 10, time.time() + 3.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            
            i += 1

        # Kick phase
        if best_code is not None:
            kick_sizes = [3, 4, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6, 5, 4, 6, 7, 3, 5, 4, 6, 5, 4, 6, 7, 3, 5]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 12:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                
                if time.time() > deadline - 10:
                    break
                
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 8, time.time() + 3.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final polish
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            
            final_code = refine(best_code)
            while time.time() < deadline - 2 and len(final_code) > 150:
                ls_code2 = local_search(final_code, 99, deadline - 1)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)
                else:
                    break
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result
=======
        best_code = None
        best_len = float('inf')

        # Phase 1: Multi-restart greedy + local search
        i = 0
        while time.time() < deadline - 30:
            code = greedy_cover(i)
            refined = refine(code)
            
            # Local search with adaptive time
            elapsed = time.time() - t_start
            remaining = deadline - 30 - elapsed
            ls_time = min(deadline - 15, time.time() + max(2.0, remaining * 0.15))
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            
            i += 1

        # Phase 2: Kick + re-greedy + local search
        if best_code is not None:
            kick_configs = []
            for k in [3, 4, 5, 6, 7, 8, 5, 6, 7, 4, 5, 6, 7, 8, 4, 5, 6, 7, 5, 6, 7, 4, 5, 6]:
                kick_configs.append(k)
            
            for kick_iter, k in enumerate(kick_configs):
                if time.time() > deadline - 15:
                    break
                rng_k = random.Random(5000 + kick_iter)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(7000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                
                if time.time() > deadline - 12:
                    break
                
                ls_time = min(deadline - 10, time.time() + 3.0)
                ls_code = local_search(re_refined, 8000 + kick_iter, ls_time)
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Phase 3: Final extended local search
        if best_code is not None and time.time() < deadline - 8:
            ls_code = local_search(best_code, 42, deadline - 5)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            
            final_code = refine(best_code)
            while time.time() < deadline - 3 and len(final_code) > 150:
                ls_code2 = local_search(final_code, 99, deadline - 2)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)
                else:
                    break
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result
>>>>>>> REPLACE
