import math
# ... inside solve_n
        while time.time() < start_time + time_limit and T > min_T:
            idx = random.randint(0, num_edges - 1)
            old_color = colors[idx]
            new_color = random.randint(0, 3)
            # Ensure new_color != old_color
            if new_color == old_color:
                # simple way to pick a different one
                # or just skip? No, we want to change.
                # pick from [0,1,2,3] \ {old_color}
                # random.randint(0, 2) + (1 if new_color >= old_color else 0) ... 
                # simpler:
                new_color = (old_color + random.randint(1, 3)) % 4
            
            colors[idx] = new_color
            new_score = evaluate(colors)
            
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_colors = colors.copy()
                    if best_score == 0:
                        return best_colors.tolist()
            else:
                # accept with prob exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_colors = colors.copy()
                        if best_score == 0:
                            return best_colors.tolist()
                else:
                    colors[idx] = old_color
                    
            T *= cooling
