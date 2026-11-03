import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s, but task says 120s. Use 110s to be safe.
    deadline = start_time + 105
    
    # 1. Precomputation
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * q + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n):
            word[i] = val % q
            val //= q
        return word

    total_words = q ** n
    all_words = list(range(total_words))
    
    # Precompute balls as sorted lists for faster iteration
    balls = {}
    for w in all_words:
        b = [w]
        w_list = int_to_list(w, n)
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    b.append(list_to_int(new_word))
        balls[w] = b
    
    # Also keep as sets for intersection operations
    ball_sets = {w: set(balls[w]) for w in all_words}

    # 2. Initial Construction: Direct Sum of Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_int(word))
            
    code = set(initial_codewords)
    
    # Incremental coverage tracking
    # coverage[w] = number of codewords whose ball contains w
    # unique_coverer[w] = the codeword if coverage[w]==1, else None
    coverage = [0] * total_words
    unique_coverer = [None] * total_words
    
    def add_codeword(c):
        code.add(c)
        for w in balls[c]:
            coverage[w] += 1
            if coverage[w] == 1:
                unique_coverer[w] = c
            else:
                unique_coverer[w] = None
    
    def remove_codeword(c):
        code.discard(c)
        for w in balls[c]:
            coverage[w] -= 1
            if coverage[w] == 0:
                unique_coverer[w] = None
            elif coverage[w] == 1:
                # Need to find which codeword covers w
                # This is expensive, so we defer to recompute if needed
                unique_coverer[w] = None  # Mark as needing recompute
            else:
                unique_coverer[w] = None
    
    def recompute_unique_coverers():
        """Recompute unique_coverer from scratch using coverage counts."""
        for w in all_words:
            if coverage[w] == 0:
                unique_coverer[w] = None
            elif coverage[w] == 1:
                # Find the single codeword that covers w
                found = None
                for c in code:
                    if w in ball_sets[c]:
                        found = c
                        break
                unique_coverer[w] = found
            else:
                unique_coverer[w] = None
    
    def compute_coverage_from_scratch():
        """Reset coverage and unique_coverer from scratch."""
        for w in all_words:
            coverage[w] = 0
            unique_coverer[w] = None
        for c in code:
            for w in balls[c]:
                coverage[w] += 1
                if coverage[w] == 1:
                    unique_coverer[w] = c
                else:
                    unique_coverer[w] = None
    
    # Initialize coverage
    compute_coverage_from_scratch()

    def is_valid():
        for w in all_words:
            if coverage[w] == 0:
                return False
        return True

    def greedy_remove():
        nonlocal code
        if time.time() > deadline:
            return False
        
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                return False
            
            # Find codewords that are not unique coverers of any word
            essential = set()
            for w in all_words:
                if unique_coverer[w] is not None:
                    essential.add(unique_coverer[w])
            
            removable = code - essential
            if removable:
                for c in removable:
                    remove_codeword(c)
                    changed = True
                if changed:
                    # Recompute unique coverers after removals
                    recompute_unique_coverers()
        
        return changed

    # Run greedy removal until stable
    while greedy_remove():
        pass

    def try_swap_or_remove():
        nonlocal code
        if time.time() > deadline:
            return False
            
        # Build unique groups
        unique_groups = {}
        for w in all_words:
            uc = unique_coverer[w]
            if uc is not None:
                if uc not in unique_groups:
                    unique_groups[uc] = []
                unique_groups[uc].append(w)
        
        # Sort by size of unique group (smaller first - easier to cover)
        candidates = sorted(unique_groups.items(), key=lambda x: len(x[1]))
        
        for c, U_c in candidates:
            if not U_c:
                continue
            
            # Find valid replacements: c' such that ball(c') covers all of U_c
            valid_cs = set(ball_sets[U_c[0]])
            for w in U_c[1:]:
                valid_cs &= ball_sets[w]
                if not valid_cs:
                    break
            
            if not valid_cs:
                continue
            
            candidate_new = valid_cs - code
            
            if not candidate_new:
                continue
            
            # Evaluate swap quality: count how many additional unique words c_new covers
            # that are currently uniquely covered by other codewords
            best_c_new = None
            best_score = -1
            cand_list = list(candidate_new)
            if len(cand_list) > 20:
                cand_list = random.sample(cand_list, 20)
            
            for c_new in cand_list:
                # Count how many words in U_c are also uniquely covered by other codewords
                # (i.e., removing c would leave them uncovered, but c_new covers them)
                # Also count how many other unique words c_new covers (potential for enabling removals)
                score = 0
                for w in ball_sets[c_new]:
                    if coverage[w] > 1:
                        score += 1  # Covers redundant words - good
                    elif coverage[w] == 1 and unique_coverer[w] != c:
                        score += 2  # Would make another codeword's unique word non-unique
                if score > best_score:
                    best_score = score
                    best_c_new = c_new
            
            if best_c_new is None:
                continue
            
            # Perform swap
            old_size = len(code)
            remove_codeword(c)
            add_codeword(best_c_new)
            
            # Try to remove redundancies
            greedy_remove()
            
            new_size = len(code)
            if new_size < old_size:
                return True
            elif new_size == old_size:
                # Accept swap if it enables further removals (check if any codeword is now removable)
                # For now, accept with some probability to explore
                if random.random() < 0.3:
                    return True
                else:
                    # Revert
                    remove_codeword(best_c_new)
                    add_codeword(c)
                    recompute_unique_coverers()
                    continue
            
            # Revert if size increased
            remove_codeword(best_c_new)
            add_codeword(c)
            recompute_unique_coverers()
        
        return False

    def try_remove_and_repair():
        nonlocal code
        if time.time() > deadline:
            return False
            
        if not code:
            return False
        
        # Try each codeword, prioritizing those with smaller unique groups
        unique_counts = {}
        for w in all_words:
            uc = unique_coverer[w]
            if uc is not None:
                unique_counts[uc] = unique_counts.get(uc, 0) + 1
        
        # Sort codewords by number of unique words they cover (fewest first)
        sorted_code = sorted(code, key=lambda c: unique_counts.get(c, 0))
        
        for c in sorted_code:
            if time.time() > deadline:
                break
            
            code_copy = set(code)
            old_coverage = coverage[:]
            old_unique = unique_coverer[:]
            
            remove_codeword(c)
            
            # Find uncovered words
            uncovered = [w for w in all_words if coverage[w] == 0]
            
            if not uncovered:
                # c was redundant - keep it removed
                recompute_unique_coverers()
                greedy_remove()
                return True
            
            # Find c' that covers all uncovered
            valid_cs = set(ball_sets[uncovered[0]])
            for w in uncovered[1:]:
                valid_cs &= ball_sets[w]
                if not valid_cs:
                    break
            
            if not valid_cs:
                # Revert
                code.clear()
                code.update(code_copy)
                coverage[:] = old_coverage
                unique_coverer[:] = old_unique
                continue
            
            candidate_new = valid_cs - code
            
            if candidate_new:
                # Pick the one that covers the most additional redundant words
                best_c_new = None
                best_score = -1
                cand_list = list(candidate_new)
                if len(cand_list) > 20:
                    cand_list = random.sample(cand_list, 20)
                
                for c_new in cand_list:
                    score = 0
                    for w in ball_sets[c_new]:
                        if coverage[w] >= 1:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_c_new = c_new
                
                if best_c_new is not None:
                    add_codeword(best_c_new)
                    recompute_unique_coverers()
                    greedy_remove()
                    return True
            
            # Revert
            code.clear()
            code.update(code_copy)
            coverage[:] = old_coverage
            unique_coverer[:] = old_unique
        
        return False

    def perturb():
        nonlocal code
        if time.time() > deadline:
            return False
        
        # Remove 1-3 random codewords
        n_remove = random.randint(1, 3)
        to_remove = random.sample(list(code), min(n_remove, len(code)))
        for c in to_remove:
            remove_codeword(c)
        
        # Greedy repair: add best codewords to cover uncovered words
        for _ in range(10):
            if time.time() > deadline:
                break
            
            uncovered = [w for w in all_words if coverage[w] == 0]
            if not uncovered:
                break
            
            # Find best codeword to add (maximizes coverage of uncovered words)
            best_c = None
            best_count = 0
            # Sample candidates
            sample_size = min(200, total_words)
            candidates = random.sample(all_words, sample_size)
            for w in candidates:
                if w in code:
                    continue
                count = sum(1 for u in uncovered if u in ball_sets[w])
                if count > best_count:
                    best_count = count
                    best_c = w
            
            if best_c is not None and best_count > 0:
                add_codeword(best_c)
            else:
                break
        
        # Polish
        recompute_unique_coverers()
        greedy_remove()
        return True

    # Main Loop
    best_size = len(code)
    best_code = set(code)
    
    # Initial polish
    greedy_remove()
    best_size = len(code)
    best_code = set(code)
    
    stall_count = 0
    
    while time.time() < deadline:
        made_progress = False
        
        # 1. Try to remove and repair (best for size reduction)
        if try_remove_and_repair():
            if len(code) < best_size:
                best_size = len(code)
                best_code = set(code)
                stall_count = 0
            made_progress = True
            continue
        
        # 2. Try swap to enable future removals
        if try_swap_or_remove():
            if len(code) < best_size:
                best_size = len(code)
                best_code = set(code)
                stall_count = 0
            made_progress = True
            continue
        
        # 3. If stuck, perturb
        if stall_count >= 3 or random.random() < 0.15:
            if perturb():
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = set(code)
                    stall_count = 0
                made_progress = True
                continue
        
        if not made_progress:
            stall_count += 1
            if stall_count > 10:
                # Restore best and try different strategy
                code.clear()
                code.update(best_code)
                compute_coverage_from_scratch()
                greedy_remove()
                stall_count = 0
                # Try perturbation
                if perturb():
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = set(code)
                continue
            continue

    # Final verification and cleanup
    greedy_remove()
    
    # Use best found
    if len(code) > len(best_code):
        code = set(best_code)
    
    result = [int_to_list(c, n) for c in sorted(code)]
    return result
