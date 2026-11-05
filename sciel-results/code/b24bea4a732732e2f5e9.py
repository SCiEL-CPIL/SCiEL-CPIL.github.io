        # Local Search
        max_swaps = 100
        swaps_made = 0
        while swaps_made < max_swaps:
            # Try to add
            addable = [x for x in numbers if x not in current_set and can_add(current_set, x)]
            if addable:
                x = random.choice(addable)
                current_set.add(x)
                swaps_made = 0 # Reset counter on success
                continue
            
            # No addable, try a random valid swap
            S_list = list(current_set)
            random.shuffle(S_list)
            swap_done = False
            for a in S_list:
                if swap_done: break
                candidates_b = [x for x in numbers if x not in current_set]
                random.shuffle(candidates_b)
                for b in candidates_b:
                    S_minus_a = current_set - {a}
                    if can_add(S_minus_a, b):
                        current_set = S_minus_a | {b}
                        swap_done = True
                        swaps_made += 1
                        break
            
            if not swap_done:
                break
