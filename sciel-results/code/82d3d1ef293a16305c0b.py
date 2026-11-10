            best_gain = -1
            best_count = 0
            best_idx = -1
            
            for i in active:
                gain = (uncovered & masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_count = 1
                    best_idx = i
                elif gain == best_gain:
                    best_count += 1
                    # Reservoir sampling: replace with probability 1/count
                    if rng.random() < (1.0 / best_count):
                        best_idx = i
            
            if best_gain <= 0:
                return None
            
            idx = best_idx
            blocks.append(list(blocks_list[idx]))
            active.remove(idx)
            uncovered &= ~masks[idx]
