            max_count = -1
            candidates = []
            for i in range(num_blocks):
                if counts[i] > max_count:
                    max_count = counts[i]
                    candidates = [i]
                elif counts[i] == max_count:
                    candidates.append(i)
            
            if not candidates:
                break
                
            chosen_idx = random.choice(candidates)
