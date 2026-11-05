            best_score = -1
            candidates = []
            
            for d in range(1, n+1):
                cand_suffix = suffix_tuple + (d,)
                
                if cand_suffix in all_perms:
                    if cand_suffix not in seen:
                        score = 1
                    else:
                        score = 0
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    candidates = [d]
                elif score == best_score:
                    candidates.append(d)
                    
            if not candidates:
                best_digit = random.randint(1, n)
            else:
                best_digit = random.choice(candidates)
