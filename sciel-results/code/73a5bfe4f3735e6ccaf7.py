        for _ in range(num_restarts):
            uncovered_mask = (1 << num_t_subs) - 1
            solution = []
            
            while uncovered_mask:
                best_gain = 0
                best_block = None
                best_mask = None
                candidates = []
                
                for block, mask in block_data:
                    gain = (uncovered_mask & mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                        best_mask = mask
                        candidates = [(block, mask)]
                    elif gain == best_gain and gain > 0:
                        candidates.append((block, mask))
                
                if best_gain == 0:
                    break
                    
                # Pick randomly from candidates
                if candidates:
                    best_block, best_mask = random.choice(candidates)
                
                solution.append(list(best_block))
                uncovered_mask &= ~best_mask
