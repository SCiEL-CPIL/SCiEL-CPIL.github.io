<<<<<<< SEARCH
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
=======
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
                for delta in range(-25, 26):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
>>>>>>> REPLACE

