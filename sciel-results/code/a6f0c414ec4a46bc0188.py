    # 2. Greedy Construction
    uncovered = set(range(total_words))
    code = []
    code_set = set()
    
    while uncovered:
        if time.time() > deadline:
            break
        best_c = -1
        max_covered = -1
        for c in range(total_words):
            if c in code_set:
                continue
            # Count how many words in balls[c] are in uncovered
            covered = 0
            for w in balls[c]:
                if w in uncovered:
                    covered += 1
                    if covered > max_covered:
                        # Early exit if we can't beat max? 
                        # Max possible is 13. If max_covered is 13, we can't beat it.
                        pass 
            if covered > max_covered:
                max_covered = covered
                best_c = c
        if best_c == -1:
            break
        code.append(best_c)
        code_set.add(best_c)
        uncovered -= balls[best_c]
