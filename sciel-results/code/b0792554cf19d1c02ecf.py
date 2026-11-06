def construct():
        import time
        start = time.time()
        
        # Seed: 32 words
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        
        # Greedy search for more words
        # Candidates: all words of weight 8 and 9
        candidates = []
        for w in range(2**17):
            if w.bit_count() in (8, 9):
                candidates.append(w)
                
        # Sort by weight or random
        import random
        random.shuffle(candidates)
        
        current_code = set(words)
        
        for c in candidates:
            if time.time() - start > 50:
                break
            # Check distance against all current words
            valid = True
            for w in current_code:
                if (c ^ w).bit_count() < 6:
                    valid = False
                    break
            if valid:
                current_code.add(c)
                
        return list(current_code)
