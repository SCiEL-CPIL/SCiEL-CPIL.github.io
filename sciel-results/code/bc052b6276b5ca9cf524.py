import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. Use 145s for logic.
    deadline = start_time + 145.0
    
    total = q ** n
    steps = [1] * n
    for i in range(1, n):
        steps[i] = steps[i - 1] * q
        
    def int_to_digits(x):
        return [x // steps[i] % q for i in range(n)]
        
    def digits_to_int(d):
        return sum(d[i] * steps[i] for i in range(n) if d[i] != 0)

    # Precompute balls
    balls = []
    for c in range(total):
        digits = int_to_digits(c)
        covered = [c]
        for i in range(n):
            orig = digits[i]
            for d in range(q):
                if d == orig:
                    continue
                digits[i] = d
                w = digits_to_int(digits)
                covered.append(w)
            digits[i] = orig
        balls.append(covered)
        
    # Precompute ball sets for faster subset checks?
    # ball_set[c] = set(balls[c])
    ball_sets = [set(b) for b in balls]

    # 1. Greedy Initialization
    def greedy_init():
        uncovered = np.ones(total, dtype=bool)
        code = []
        # We need to pick the codeword that covers the most uncovered words.
        # To speed this up, we can just scan all c in total.
        # total=2187, ball size=15. 2187*15 ~ 33k ops per step.
        # We might need ~180 steps. 180 * 33k ~ 6M ops. Fast enough.
        
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            best_c = -1
            best_gain = 0
            for c in range(total):
                # Quick check: does c cover any uncovered?
                gain = 0
                for w in balls[c]:
                    if uncovered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            for w in balls[best_c]:
                uncovered[w] = False
        return code

    init_code = greedy_init()
    
    # State management for Local Search
    # cover_count[w] = number of codewords in C that cover w
    # code_set = set of codewords
    # We maintain validity: min(cover_count) >= 1
    
    def minimize_cover(code_list):
        """
        Given a valid cover, try to remove codewords one by one.
        Returns the minimized list.
        """
        code_set = set(code_list)
        cover_count = np.zeros(total, dtype=np.int16)
        for c in code_set:
            for w in balls[c]:
                cover_count[w] += 1
        
        # Check if valid (should be)
        if np.min(cover_count) < 1:
             # If not valid, we can't minimize. Return as is or repair?
             # Greedy init ensures validity. Swaps ensure validity.
             # So this shouldn't happen unless bug.
             pass

        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Iterate through a copy of the list to allow removal
            # We want to remove codewords that are redundant.
            # A codeword c is redundant if for all w in balls[c], cover_count[w] > 1.
            # Equivalently, there is no w in balls[c] such that cover_count[w] == 1.
            
            # To make it faster, we can just check each c in code_set.
            # If we remove c, we must update cover_count.
            
            for c in list(code_set):
                # Check if c is critical for any word
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove c
                    code_set.remove(c)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    changed = True
                    # Since we modified the set, we might want to restart the loop
                    # or continue. Continuing is fine, but we might miss opportunities
                    # if removal of c makes another codeword redundant that we already passed.
                    # However, removing a codeword only DECREASES cover counts.
                    # It cannot make a non-redundant codeword redundant.
                    # It CAN make a redundant codeword... still redundant?
                    # If c was redundant, all its words had count >= 2.
                    # Removing c reduces counts by 1.
                    # Other codewords' redundancy status depends on their own words.
                    # If another codeword c2 had a word w with count 2, and c covered w,
                    # then after removing c, count[w] becomes 1.
                    # So c2 might become NON-redundant.
                    # So removing codewords can "protect" other codewords from being removed.
                    # Therefore, the order matters.
                    # A simple greedy removal until no more are removable yields a "minimal" cover
                    # (no single codeword can be removed).
                    # To find a *smaller* minimal cover, we need swaps.
        
        return list(code_set)

    # ILS Loop
    best_code = minimize_cover(init_code)
    best_size = len(best_code)
    
    # Precompute "critical sets" for swap logic?
    # For a given state C, a swap (c_old -> c_new) is valid iff:
    # CriticalWords(c_old) subset of Ball(c_new)
    # CriticalWords(c_old) = { w in Ball(c_old) | cover_count[w] == 1 }
    
    def perform_ils(current_code):
        code_set = set(current_code)
        cover_count = np.zeros(total, dtype=np.int16)
        for c in code_set:
            for w in balls[c]:
                cover_count[w] += 1
        
        # Ensure we start from a minimal cover
        # The input is already minimized.
        
        current_size = len(code_set)
        
        # ILS Parameters
        # We try to find a better minimal cover.
        # Move: Swap one codeword.
        # Acceptance: Always accept valid swaps? Or use temperature?
        # Since we want to minimize size, and swaps keep size constant,
        # the goal of a swap is to reach a state where a removal is possible.
        
        # Strategy:
        # 1. Try to remove. (Already done, so none possible).
        # 2. Pick a random c_old.
        # 3. Find all c_new such that swap is valid.
        # 4. If any, pick one, apply swap.
        # 5. Try to remove again.
        # 6. If size reduced, update best.
        # 7. If stuck, repeat.
        
        while time.time() < deadline - 2:
            # 1. Minimize (should be no-op if we always minimize after swap)
            # But maybe the swap made a codeword redundant?
            # A swap removes c_old and adds c_new.
            # Counts decrease for Ball(c_old) \ Ball(c_new).
            # Counts increase for Ball(c_new) \ Ball(c_old).
            # So a codeword c_other might become redundant if its critical words
            # were covered by c_old and are now covered by c_new? No.
            # Redundancy means ALL words in Ball(c_other) have count > 1.
            # If counts increase, it's easier to be redundant.
            # So yes, a swap can make other codewords redundant.
            
            # So we must run minimize after every swap.
            
            # 2. Select c_old
            if not code_set:
                break
            c_old = random.choice(list(code_set))
            
            # Calculate Critical Words for c_old
            crit_words = []
            for w in balls[c_old]:
                if cover_count[w] == 1:
                    crit_words.append(w)
            
            # If no critical words, c_old is redundant.
            # But we are in a minimal cover, so this shouldn't happen.
            # Unless we didn't minimize properly.
            if not crit_words:
                 # Should not happen. Remove it.
                 code_set.remove(c_old)
                 for w in balls[c_old]:
                     cover_count[w] -= 1
                 continue

            # 3. Find valid c_new
            # c_new must cover all crit_words.
            # And c_new must not be in code_set.
            # We can iterate through all c in total?
            # 2187 checks. Each check: is crit_words subset of ball_set[c]?
            # crit_words size <= 15.
            # 2187 * 15 ~ 33k ops.
            # This is fast.
            
            valid_swaps = []
            for c_new in range(total):
                if c_new in code_set:
                    continue
                # Check if ball_set[c_new] covers crit_words
                # Using set intersection might be faster if crit_words is small
                # or just loop.
                # Let's use set for c_new balls.
                if ball_sets[c_new].issuperset(crit_words):
                    valid_swaps.append(c_new)
            
            if not valid_swaps:
                # No valid swap for this c_old.
                # Try another c_old? Or restart?
                # If we can't swap out c_old, maybe we can swap out another?
                # We picked c_old randomly. If it fails, we just try again in next iteration.
                # To avoid infinite loop on same c_old, we just loop.
                # But if NO c_old has a valid swap, we are stuck in a local minimum
                # of the "minimal cover" graph.
                # We need a more aggressive perturbation?
                # For now, just continue. The random choice of c_old will change.
                continue
            
            # 4. Pick a swap
            # Heuristic: Prefer swaps that increase redundancy elsewhere?
            # Or just random.
            c_new = random.choice(valid_swaps)
            
            # 5. Apply Swap
            # Remove c_old
            for w in balls[c_old]:
                cover_count[w] -= 1
            code_set.remove(c_old)
            
            # Add c_new
            for w in balls[c_new]:
                cover_count[w] += 1
            code_set.add(c_new)
            
            # 6. Minimize again
            # We can inline the minimization for speed
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                for c in list(code_set):
                    is_redundant = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        code_set.remove(c)
                        for w in balls[c]:
                            cover_count[w] -= 1
                        changed = True
            
            current_size = len(code_set)
            if current_size < best_size:
                best_size = current_size
                best_code = list(code_set)
                
        return best_code, best_size

    # Run ILS
    final_code, final_size = perform_ils(best_code)
    
    # If we have time, try a few more restarts?
    # The ILS loop runs until deadline.
    # But maybe the first run gets stuck.
    # We can run multiple ILS instances if time permits.
    
    # Let's structure it as:
    # 1. Greedy Init
    # 2. Minimize
    # 3. ILS (runs until time is low)
    
    # The current perform_ils runs until deadline.
    # We should probably run it in a loop with restarts if we detect stagnation?
    # Or just let it run.
    
    # Actually, let's add a simple restart mechanism inside the main flow if we have time.
    # But perform_ils consumes the time.
    
    # Let's refactor: perform_ils takes a time_limit.
    
    # Re-doing the logic to fit time better.
    
    def run_search(initial_code, time_limit):
        code_set = set(initial_code)
        cover_count = np.zeros(total, dtype=np.int16)
        for c in code_set:
            for w in balls[c]:
                cover_count[w] += 1
        
        # Initial Minimize
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for c in list(code_set):
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    code_set.remove(c)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    changed = True
        
        current_size = len(code_set)
        best_local = list(code_set)
        best_local_size = current_size
        
        while time.time() < time_limit:
            # Swap Move
            if not code_set:
                break
            c_old = random.choice(list(code_set))
            crit_words = [w for w in balls[c_old] if cover_count[w] == 1]
            
            if not crit_words:
                # Should not happen if minimized, but handle it
                code_set.remove(c_old)
                for w in balls[c_old]:
                    cover_count[w] -= 1
                continue

            valid_swaps = []
            for c_new in range(total):
                if c_new in code_set:
                    continue
                if ball_sets[c_new].issuperset(crit_words):
                    valid_swaps.append(c_new)
            
            if not valid_swaps:
                continue
                
            c_new = random.choice(valid_swaps)
            
            # Apply
            for w in balls[c_old]:
                cover_count[w] -= 1
            code_set.remove(c_old)
            
            for w in balls[c_new]:
                cover_count[w] += 1
            code_set.add(c_new)
            
            # Minimize
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                for c in list(code_set):
                    is_redundant = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        code_set.remove(c)
                        for w in balls[c]:
                            cover_count[w] -= 1
                        changed = True
            
            current_size = len(code_set)
            if current_size < best_local_size:
                best_local_size = current_size
                best_local = list(code_set)
                
        return best_local, best_local_size

    # Multi-start ILS
    best_overall = best_code
    best_overall_size = best_size
    
    # We have ~140s left.
    # Run ILS for 10s, then restart?
    # Or run one long ILS?
    # Long ILS might get stuck in a basin.
    # Let's do 5 restarts of 25s each.
    
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    if time_per_restart < 5:
        num_restarts = max(1, int((deadline - time.time()) / 5))
        time_per_restart = (deadline - time.time()) / num_restarts

    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        # Perturb the current best to create a new start
        # Randomly swap 10% of codewords?
        # Or just use the current best as start?
        # Using the current best is "Iterated".
        # To explore more, we can perturb the best found so far.
        
        start_code = list(best_overall)
        
        # Perturbation: Randomly replace 5% of codewords with valid swaps?
        # That's complex.
        # Let's just restart from the current best. ILS will move it.
        # If it gets stuck, we might need a stronger kick.
        
        # Simple kick: Remove 5 random codewords, then repair greedily?
        # That breaks validity.
        
        # Let's just run ILS from the current best.
        t_end = time.time() + time_per_restart
        local_best, local_size = run_search(start_code, t_end)
        
        if local_size < best_overall_size:
            best_overall_size = local_size
            best_overall = local_best

    # Final output
    final_code = [int_to_digits(c) for c in best_overall]
    
    # Ensure distinctness
    seen = set()
    unique_code = []
    for c in best_overall:
        if c not in seen:
            seen.add(c)
            unique_code.append(c)
            
    if len(unique_code) != len(best_overall):
        final_code = [int_to_digits(c) for c in unique_code]
        
    return final_code
