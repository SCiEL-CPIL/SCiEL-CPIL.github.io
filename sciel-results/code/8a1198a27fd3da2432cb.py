    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        selected = []
        gain = [0] * n_k
        touched = []

        while uncovered:
            # Reset gain for touched indices
            for idx in touched:
                gain[idx] = 0
            touched = []
            
            # Calculate gains
            for t_idx in uncovered:
                for k_idx in t_sub_to_k_idxs[t_idx]:
                    if gain[k_idx] == 0:
                        touched.append(k_idx)
                    gain[k_idx] += 1
            
            # Find best block
            best_gain = -1
            best_candidates = []
            for i in touched:
                if gain[i] > best_gain:
                    best_gain = gain[i]
                    best_candidates = [i]
                elif gain[i] == best_gain:
                    best_candidates.append(i)
            
            if best_gain <= 0:
                break
                
            best_idx = rng.choice(best_candidates)
            selected.append(best_idx)
            
            # Update uncovered
            for t_idx in k_sub_to_t_idxs[best_idx]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    
            if time.time() > deadline - 10:
                break

        return selected
