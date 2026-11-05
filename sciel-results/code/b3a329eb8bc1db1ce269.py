            # Local Search
            freq = [0] * MAX_COLOR
            for c in colors:
                freq[c] += 1
            current_distinct = sum(1 for f in freq if f > 0)
            
            if current_distinct > best_count:
                best_count = current_distinct
                best_colors = colors[:]

            for step in range(5000):
                if time.time() > deadline:
                    break
                
                ei = random.randint(0, NUM_EDGES - 1)
                old_color = colors[ei]
                
                # Choose new color
                if random.random() < 0.7:
                    unused = [c for c in range(MAX_COLOR) if freq[c] == 0]
                    if unused:
                        new_color = random.choice(unused)
                    else:
                        new_color = random.randint(0, MAX_COLOR - 1)
                else:
                    new_color = random.randint(0, MAX_COLOR - 1)
                    
                if new_color == old_color:
                    continue
                
                # Check validity
                if not try_assign_color(colors, ei, new_color, cycles, edge_to_cycles):
                    continue
                
                # Move is valid.
                # Calculate delta
                old_freq = freq[old_color]
                new_freq = freq[new_color]
                
                # Update freq
                freq[old_color] -= 1
                freq[new_color] += 1
                
                # Calculate delta in distinct count
                delta = 0
                if old_freq == 1 and new_freq == 0:
                    delta = 0
                elif old_freq > 1 and new_freq == 0:
                    delta = 1
                elif old_freq == 1 and new_freq > 0:
                    delta = -1
                else:
                    delta = 0
                
                # Acceptance
                accepted = False
                if delta > 0:
                    accepted = True
                elif delta == 0:
                    # Accept neutral moves to explore
                    if random.random() < 0.9:
                        accepted = True
                else:
                    # Delta < 0
                    # Accept with very low probability? Or never?
                    # Let's never accept to keep the best so far.
                    # But we might need to go down to go up?
                    # Given the time limit, hill climbing is safer.
                    accepted = False
                
                if accepted:
                    current_distinct += delta
                    if current_distinct > best_count:
                        best_count = current_distinct
                        best_colors = colors[:]
                else:
                    # Revert
                    colors[ei] = old_color
                    freq[old_color] += 1
                    freq[new_color] -= 1
