import time
import itertools
import random
import math

def _score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U = set(U)
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    U_plus = {a + b for a in U for b in U}
    U_minus = {a - b for a in U for b in U}
    val = len(U_minus) / len(U_plus)
    if val <= 0:
        return 0.0
    return 1.0 + math.log(val) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    # 1. Parametric Families and Small Exhaustive Search
    # Family 1: Small sets, exhaustive search
    # We limit the exhaustive search to keep it fast, as it's mainly for seeding
    for n in range(2, 6):
        if time.time() > deadline - 10:
            break
        # Search up to 15 for small sets to be safe
        for combo in itertools.combinations(range(1, 16), n - 1):
            if time.time() > deadline - 10:
                break
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    # Family 2: Powers
    for base in range(2, 12):
        for k in range(2, 10):
            if time.time() > deadline - 10:
                break
            U = [base ** i for i in range(k)]
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    # 2. Improved Local Search with Heuristic Pre-filtering
    def local_search_diff_guided(U, max_iters=2000, max_val=200):
        """
        Local search that uses a heuristic to filter candidate moves.
        It estimates the change in |U-U| and |U+U| before doing full calculation.
        """
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        current_score = _score(U)
        best_U = list(U)
        
        # Precompute sets for heuristic
        cur_sum_set = set()
        cur_diff_set = set()
        for a in best_U:
            for b in best_U:
                cur_sum_set.add(a + b)
                cur_diff_set.add(a - b)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            n = len(best_U)
            U_set = set(best_U)
            
            # --- Move 1: Try moving an existing element ---
            if n > 1:
                # Randomly select an element to move to avoid bias
                indices = [i for i in range(n) if best_U[i] != 0]
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline:
                        break
                    old_val = best_U[i]
                    
                    # Search range for new value
                    # Bias towards keeping max(U) somewhat controlled but allowing growth
                    search_start = max(1, old_val - 10)
                    search_end = min(max_val + 10, old_val + 20)
                    
                    best_new_val = None
                    best_new_score = current_score
                    best_heuristic = -1000
                    
                    # Generate candidates
                    candidates = list(range(search_start, search_end + 1))
                    random.shuffle(candidates)
                    
                    for new_val in candidates:
                        if new_val == old_val or new_val in U_set:
                            continue
                        
                        # Heuristic: Estimate new diffs and sums
                        # We only add/remove differences involving old_val and new_val
                        # This is an approximation but good for filtering
                        
                        # Count new diffs/sums created by new_val
                        new_diffs = 0
                        new_sums = 0
                        
                        for a in U_set:
                            if a == old_val:
                                continue
                            d1 = new_val - a
                            d2 = a - new_val
                            if d1 not in cur_diff_set: new_diffs += 1
                            if d2 not in cur_diff_set: new_diffs += 1
                            s = new_val + a
                            if s not in cur_sum_set: new_sums += 1
                            
                        # Count diffs/sums lost by removing old_val
                        # (Simplified: we just check if they exist in the set minus old_val interactions)
                        # For a rough heuristic, we can just look at net gain
                        # A move is good if it adds more unique diffs than sums
                        heuristic_score = new_diffs - new_sums * 0.8
                        
                        if heuristic_score > best_heuristic:
                            best_heuristic = heuristic_score
                            best_new_val = new_val
                            # Don't break, keep looking for better heuristic score? 
                            # Or just take the first promising one to save time?
                            # Let's take the top few candidates by heuristic
                            # Actually, for speed, let's just evaluate the best heuristic candidate
                            # and maybe a few others if time permits.
                            
                    if best_new_val is not None:
                        # Perform full score calculation for the best candidate
                        new_U = list(best_U)
                        new_U[i] = best_new_val
                        new_U = sorted(set(new_U))
                        if 0 not in new_U or len(new_U) < 2:
                            continue
                        new_score = _score(new_U)
                        if new_score > best_new_score + 1e-10:
                            # Update sets for next iteration's heuristic
                            best_U = new_U
                            current_score = new_score
                            U_set = set(best_U)
                            cur_sum_set = set()
                            cur_diff_set = set()
                            for a in best_U:
                                for b in best_U:
                                    cur_sum_set.add(a + b)
                                    cur_diff_set.add(a - b)
                            improved = True
                            break # Restart loop with new state
        
            # --- Move 2: Try adding a new element ---
            if not improved:
                # Sample candidates for addition
                # Focus on values that might expand the range or fill gaps
                max_u = max(best_U)
                candidates = list(range(1, min(max_u + 20, max_val + 20)))
                random.shuffle(candidates)
                
                best_candidate = None
                best_candidate_score = current_score
                best_heuristic = -1000
                
                for candidate in candidates[:50]: # Limit candidates to check
                    if candidate in U_set:
                        continue
                    
                    new_diffs = 0
                    new_sums = 0
                    for a in U_set:
                        d1 = candidate - a
                        d2 = a - candidate
                        if d1 not in cur_diff_set: new_diffs += 1
                        if d2 not in cur_diff_set: new_diffs += 1
                        s = candidate + a
                        if s not in cur_sum_set: new_sums += 1
                        
                    heuristic_score = new_diffs - new_sums * 0.7
                    if heuristic_score > best_heuristic:
                        best_heuristic = heuristic_score
                        best_candidate = candidate
                        
                if best_candidate is not None:
                    new_U = sorted(best_U + [best_candidate])
                    new_score = _score(new_U)
                    if new_score > best_candidate_score + 1e-10:
                        best_U = new_U
                        current_score = new_score
                        U_set = set(best_U)
                        cur_sum_set = set()
                        cur_diff_set = set()
                        for a in best_U:
                            for b in best_U:
                                cur_sum_set.add(a + b)
                                cur_diff_set.add(a - b)
                        improved = True

            # --- Move 3: Try removing an element ---
            if not improved and n > 2:
                indices = [i for i in range(n) if best_U[i] != 0]
                random.shuffle(indices)
                for i in indices[:10]: # Check a few removals
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2: continue
                    new_score = _score(new_U)
                    if new_score > current_score + 1e-10:
                        best_U = new_U
                        current_score = new_score
                        U_set = set(best_U)
                        cur_sum_set = set()
                        cur_diff_set = set()
                        for a in best_U:
                            for b in best_U:
                                cur_sum_set.add(a + b)
                                cur_diff_set.add(a - b)
                        improved = True
                        break

            if not improved:
                # Try expanding max_val slightly to allow growth
                max_val = min(max_val * 1.2, 1000)
                # If stuck for a while, might want to break
                break
                
        return best_U, current_score

    # 3. Main Search Loop
    # Hill climb from the best found so far
    U_hc, score_hc = local_search_diff_guided(best_set, max_iters=5000, max_val=100)
    if score_hc > best_score:
        best_set = U_hc
        best_score = score_hc

    # Multiple random restarts
    random.seed(42)
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Generate random seed set
        size = random.randint(3, 15)
        max_el = random.randint(10, 100)
        # Ensure 0 is included
        U_rand = [0] + random.sample(range(1, max_el + 1), min(size - 1, max_el))
        
        # Run local search
        U_hc, score_hc = local_search_diff_guided(U_rand, max_iters=3000, max_val=150)
        
        if score_hc > best_score:
            best_set = U_hc
            best_score = score_hc
            
    # Final Polish: Tighter local search on the best set found
    if time.time() < deadline - 2:
        U_polish, score_polish = local_search_diff_guided(best_set, max_iters=2000, max_val=max(best_set) + 50)
        if score_polish > best_score:
            best_set = U_polish
            best_score = score_polish

    return sorted(best_set)
