<<<<<<< SEARCH
            # Strategy 1: Try adding a new element
            for new_elem in range(1, 120):
                if new_elem in U_set:
                    continue
                candidate = sorted(list(U_set) + [new_elem])
                s = compute_score_fast(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate
                    current_U = candidate
                    U_set = set(current_U)
                    improved = True
                    break
                if time.time() > deadline - 15:
                    break
=======
            # Strategy 1: Try adding a new element
            # Expanded range
            for new_elem in range(1, 500):
                if new_elem in U_set:
                    continue
                candidate = sorted(list(U_set) + [new_elem])
                s = compute_score_fast(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate
                    current_U = candidate
                    U_set = set(current_U)
                    improved = True
                    break
                if time.time() > deadline - 15:
                    break
>>>>>>> REPLACE

