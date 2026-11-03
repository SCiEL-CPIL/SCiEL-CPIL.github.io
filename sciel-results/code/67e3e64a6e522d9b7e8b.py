    def run_greedy(seed):
        random.seed(seed) # Not strictly needed if no random ops inside
        uncovered = full_mask
        code_indices = []
        
        while uncovered:
            best_c = -1
            max_cover = 0
            for c in range(total_words):
                cover = (uncovered & balls[c]).bit_count()
                if cover > max_cover:
                    max_cover = cover
                    best_c = c
            
            if best_c == -1:
                break
                
            code_indices.append(best_c)
            uncovered &= ~balls[best_c]
            
        code_indices = prune_code(code_indices, balls, full_mask)
        return code_indices
