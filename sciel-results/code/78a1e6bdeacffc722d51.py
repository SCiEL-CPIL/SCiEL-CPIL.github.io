import time
import random

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    def evaluate(S):
        S_set = set(S)
        valid_diffs = set()
        for d in range(1, 256):
            found = False
            max_a = 512 - 2 * d
            for a in range(1, max_a + 1):
                if a in S_set and (a + d) in S_set and (a + 2 * d) in S_set:
                    found = True
                    break
            if found:
                valid_diffs.add(d)
        return len(valid_diffs), valid_diffs

    # Initial candidates
    candidates = []
    candidates.append(list(range(1, 33)))
    for _ in range(50):
        candidates.append(random.sample(range(1, 513), 32))
        
    best_S = None
    best_score = -1
    best_diffs = set()
    for c in candidates:
        s, diffs = evaluate(c)
        if s > best_score:
            best_score = s
            best_S = c[:]
            best_diffs = diffs
            if best_score == 255:
                return best_S

    current_S = best_S[:]
    current_score = best_score
    current_diffs = best_diffs

    all_diffs = set(range(1, 256))

    while time.time() - start_time < timeout:
        # Try targeted moves
        missing_diffs = list(all_diffs - current_diffs)
        if not missing_diffs:
            break # Found all possible differences
            
        random.shuffle(missing_diffs)
        
        move_made = False
        # Try up to 100 missing differences in random order
        for d in missing_diffs[:100]:
            max_a = 512 - 2 * d
            partials = []
            for a in range(1, max_a + 1):
                x1 = a
                x2 = a + d
                x3 = a + 2 * d
                count = 0
                missing_elem = None
                if x1 in current_S:
                    count += 1
                else:
                    missing_elem = x1
                if x2 in current_S:
                    count += 1
                else:
                    if missing_elem is not None:
                        break
                    missing_elem = x2
                if x3 in current_S:
                    count += 1
                else:
                    if missing_elem is not None:
                        break
                    missing_elem = x3
                
                if count == 2:
                    if 1 <= missing_elem <= 512:
                        if missing_elem not in current_S:
                            partials.append((a, missing_elem))
            
            if not partials:
                continue
                
            a, x = random.choice(partials)
            
            elems = [a, a+d, a+2*d]
            existing_in_triple = [e for e in elems if e in current_S]
            
            candidates_for_removal = []
            for val in current_S:
                if val not in existing_in_triple:
                    candidates_for_removal.append(val)
            
            if not candidates_for_removal:
                continue
                
            random.shuffle(candidates_for_removal)
            for y in candidates_for_removal[:5]:
                new_S = current_S[:]
                new_S.remove(y)
                new_S.append(x)
                
                new_score, new_diffs = evaluate(new_S)
                
                if new_score > current_score:
                    current_S = new_S
                    current_score = new_score
                    current_diffs = new_diffs
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = current_S[:]
                        best_diffs = current_diffs
                        if best_score == 255:
                            return best_S
                    
                    move_made = True
                    break
            if move_made:
                break

        if not move_made:
            if random.random() < 0.5:
                i, j = random.sample(range(32), 2)
                old_i, old_j = current_S[i], current_S[j]
                current_S[i], current_S[j] = current_S[j], current_S[i]
                
                new_score, new_diffs = evaluate(current_S)
                
                if new_score >= current_score:
                    current_score = new_score
                    current_diffs = new_diffs
                    if current_score > best_score:
                        best_score = current_score
                        best_S = current_S[:]
                        best_diffs = current_diffs
                        if best_score == 255:
                            return best_S
                else:
                    current_S[i], current_S[j] = old_i, old_j
            else:
                i = random.randint(0, 31)
                old_val = current_S[i]
                new_val = random.randint(1, 512)
                if new_val in current_S:
                    continue
                current_S[i] = new_val
                
                new_score, new_diffs = evaluate(current_S)
                
                if new_score >= current_score:
                    current_score = new_score
                    current_diffs = new_diffs
                    if current_score > best_score:
                        best_score = current_score
                        best_S = current_S[:]
                        best_diffs = current_diffs
                        if best_score == 255:
                            return best_S
                else:
                    current_S[i] = old_val

    return best_S
