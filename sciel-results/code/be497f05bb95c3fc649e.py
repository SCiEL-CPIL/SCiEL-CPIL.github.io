<<<<<<< SEARCH
    # Local search: try random color reassignments to reduce color count
    if time.time() < deadline - 3.0:
        random.seed(42)
        best_count = count_colors(current_colors)
        best_state = list(current_colors)
        
        # Simulated annealing
        temp = 5.0
        for step in range(20000):
            if time.time() > deadline - 3.0:
                break
            
            pos = random.randint(0, n - 1)
            old_color = current_colors[pos]
            # Pick a new color from existing used colors or a slightly different one
            used = set(current_colors)
            if random.random() < 0.8:
                new_color = random.choice(list(used))
            else:
                new_color = random.randint(0, max(3, best_count) - 1)
            
            if new_color == old_color:
                continue
            
            current_colors[pos] = new_color
            if verify(current_colors):
                new_count = count_colors(current_colors)
                if new_count < best_count:
                    best_count = new_count
                    best_state = list(current_colors)
                    temp = max(0.5, temp * 0.95)
                else:
                    # Accept with probability based on temperature
                    if random.random() < temp / max(1, best_count):
                        pass
                    else:
                        current_colors[pos] = old_color
            else:
                current_colors[pos] = old_color
            
            temp *= 0.9995

        current_colors = best_state
=======
    # Local search: try random color reassignments to reduce color count
    if time.time() < deadline - 3.0:
        random.seed(42)
        best_count = count_colors(current_colors)
        best_state = current_colors.copy()
        
        # Simulated annealing
        temp = 5.0
        steps = 0
        while time.time() < deadline - 3.0:
            steps += 1
            if steps % 1000 == 0:
                # Check if we can break early if no improvement for a long time? 
                # For now just rely on time.
                pass
            
            pos = random.randint(0, n - 1)
            old_color = current_colors[pos]
            # Pick a new color from existing used colors or a slightly different one
            used = np.unique(current_colors)
            if random.random() < 0.8:
                new_color = used[random.randint(0, len(used)-1)]
            else:
                new_color = random.randint(0, max(3, best_count) - 1)
            
            if new_color == old_color:
                continue
            
            current_colors[pos] = new_color
            if verify(current_colors):
                new_count = count_colors(current_colors)
                if new_count < best_count:
                    best_count = new_count
                    best_state = current_colors.copy()
                    temp = max(0.5, temp * 0.95)
                else:
                    # Accept with probability based on temperature
                    if random.random() < temp / max(1, best_count):
                        pass
                    else:
                        current_colors[pos] = old_color
            else:
                current_colors[pos] = old_color
            
            temp *= 0.9995

        current_colors = best_state
>>>>>>> REPLACE

