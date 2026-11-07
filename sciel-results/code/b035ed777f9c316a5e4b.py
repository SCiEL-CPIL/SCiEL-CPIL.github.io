            # Try Add
            if rng.random() < 0.3: # 30% chance to try add
                cand = rng.randint(1, 256)
                if cand in current_set:
                    continue
                trial_set = current_set + [cand]
                trial_sums = _compute_sums(trial_set)
                if len(trial_sums) == (1 << len(trial_set)) - 1:
                    if _is_valid_sums(trial_sums):
                        current_set = trial_set
                        current_sums = trial_sums
                        if len(current_set) > len(best_set):
                            best_set = list(current_set)
                            best_sums = set(current_sums)
            else:
                # Swap logic
                # ...
