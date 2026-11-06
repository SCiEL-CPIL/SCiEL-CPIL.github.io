"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # 55s safety margin for 60s limit
    best_result = None
    best_size = float('inf')
    
    # Bytearray size: max sum is 2*256 = 512. Index 0..512.
    SUM_SIZE = 2 * N + 1 

    def compute_sums_ba(s):
        ba = bytearray(SUM_SIZE)
        for i in range(len(s)):
            for j in range(i, len(s)):
                ba[s[i] + s[j]] = 1
        return ba

    def is_sidon_ba(s, ba):
        # Verify s is Sidon using bytearray (should be true if constructed correctly, 
        # but useful for verification if we manipulate s directly)
        # Actually, if we maintain ba correctly, we just need to ensure no duplicates were added.
        # This function is for final verification.
        seen = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in seen:
                    return False
                seen.add(sm)
        return True

    def is_maximal_ba(s, ba):
        s_set = set(s)
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Check if x can be added
            # x is addable if 2x and x+a are all 0 in ba
            if ba[2 * x]:
                continue
            addable = True
            for a in s:
                if ba[x + a]:
                    addable = False
                    break
            if addable:
                return False
        return True

    def try_remove_element_ba(s, ba, x):
        if x not in s:
            return None, None
        
        # Create new s
        new_s = [a for a in s if a != x]
        
        # Recalculate sums bytearray
        new_ba = bytearray(SUM_SIZE)
        for i in range(len(new_s)):
            for j in range(i, len(new_s)):
                new_ba[new_s[i] + new_s[j]] = 1
        
        # Check if still maximal
        if is_maximal_ba(new_s, new_ba):
            return new_s, new_ba
        return None, None

    def local_search_minimize_ba(s, ba, max_iters=50):
        s = list(s)
        ba = bytearray(ba)
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            for x in list(s):
                new_s, new_ba = try_remove_element_ba(s, ba, x)
                if new_s is not None:
                    s = new_s
                    ba = new_ba
                    improved = True
                    break
        return s, ba

    def make_maximal_ba(s, ba):
        s = list(s)
        ba = bytearray(ba)
        s_set = set(s)

        while True:
            # Find all unblocked elements
            unblocked = []
            for x in range(1, N + 1):
                if x in s_set:
                    continue
                if ba[2 * x]:
                    continue
                # Check all x+a
                blocked = False
                for a in s:
                    if ba[x + a]:
                        blocked = True
                        break
                if not blocked:
                    unblocked.append(x)

            if not unblocked:
                break

            # Strategy: Pick element that blocks the most other unblocked elements
            # This is O(U^2 * K). With N=256, U~100, K~10, this is 10000*10 = 100k ops.
            # In Python, this might be ~10ms per step. 15 steps -> 150ms per trial.
            # 100 trials -> 15s. This is feasible.
            
            best_x = None
            best_block_count = -1
            
            for x in unblocked:
                # Calculate block count for x
                block_count = 0
                # Precompute sums involving x for quick check? 
                # x adds 2x and x+a to ba.
                # y is blocked if 2y or y+b is in (old_ba OR new_sums_from_x)
                
                for y in unblocked:
                    if y == x:
                        continue
                    # Check if y is blocked by x
                    # y is blocked if:
                    # 1. 2y was already blocked (ba[2y]==1) -- but y is unblocked, so ba[2y]==0
                    #    So we only care if 2y becomes blocked by x.
                    #    2y blocked by x if 2y == 2x (y=x, handled) or 2y == x+a.
                    # 2. y+b was already blocked (ba[y+b]==1) -- y is unblocked, so ba[y+b]==0 for all b in s.
                    #    So we only care if y+b becomes blocked by x.
                    #    y+b blocked by x if y+b == 2x or y+b == x+a.
                    
                    blocked = False
                    # Check 2y
                    if 2 * y == 2 * x:
                        blocked = True
                    else:
                        # Check if 2y == x+a for some a in s
                        # 2y - x == a
                        a_candidate = 2 * y - x
                        if 0 < a_candidate <= N and a_candidate in s_set:
                            blocked = True
                    
                    if not blocked:
                        # Check y+b
                        # y+b == 2x => b = 2x - y
                        b_candidate = 2 * x - y
                        if 0 < b_candidate <= N and b_candidate in s_set:
                            blocked = True
                        else:
                            # y+b == x+a => b-a = x-y => b = a + x - y
                            # Iterate a? Or check if (y + (a + x - y)) is in sums?
                            # We need to check if there exists a in s such that y+a is in {2x, x+a' | a' in s}
                            # This is equivalent to:
                            # (y+a) == 2x  => a = 2x - y (checked above)
                            # (y+a) == x+a' => a - a' = x - y
                            # So we need to check if there exists a, a' in s such that a - a' = x - y
                            # This is O(K^2) if done naively. 
                            # However, we can precompute differences?
                            # For now, let's just check the condition directly.
                            # Since K is small, maybe just loop?
                            # But we are inside a double loop over unblocked.
                            # Let's optimize: 
                            # We know y is unblocked, so no y+b in old_ba.
                            # We just need to see if x creates a conflict.
                            # Conflict if (y+a) in {2x} U {x+a' | a' in s}
                            # 1. y+a == 2x => a = 2x-y. Checked.
                            # 2. y+a == x+a' => a' = a + y - x.
                            # So for each a in s, check if a + y - x is in s_set.
                            diff = y - x
                            for a in s:
                                a_prime = a + diff
                                if 0 < a_prime <= N and a_prime in s_set:
                                    blocked = True
                                    break
                            
                    if blocked:
                        block_count += 1

                if block_count > best_block_count:
                    best_block_count = block_count
                    best_x = x

            if best_x is None:
                break

            # Add best_x
            # Update ba
            ba[2 * best_x] = 1
            for a in s:
                ba[best_x + a] = 1
            s.append(best_x)
            s_set.add(best_x)

        return s, ba

    # Main search
    random.seed(42)
    
    # Strategy: Random start, make maximal, local search
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Start with small random Sidon set
        start_size = random.randint(4, 8)
        s = []
        ba = bytearray(SUM_SIZE)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(s) >= start_size:
                break
            if ba[2 * x]:
                continue
            # Check all x+a
            ok = True
            for a in s:
                if ba[x + a]:
                    ok = False
                    break
            if ok:
                s.append(x)
                ba[2 * x] = 1
                for a in s[:-1]:
                    ba[x + a] = 1
        
        if len(s) < 3:
            continue
            
        # Make maximal
        s_max, ba_max = make_maximal_ba(s, ba)
        
        # Verify
        if not is_maximal_ba(s_max, ba_max):
            continue
            
        # Local search
        s_min, ba_min = local_search_minimize_ba(s_max, ba_max)
        
        # Final check
        if not is_sidon_ba(s_min, ba_min) or not is_maximal_ba(s_min, ba_min):
            continue
            
        size = len(s_min)
        if size < best_size:
            best_size = size
            best_result = s_min
            if size <= 12: # Heuristic cutoff
                break

    # Fallback
    if best_result is None:
        s = []
        ba = bytearray(SUM_SIZE)
        for x in range(1, N + 1):
            if ba[2 * x]:
                continue
            ok = True
            for a in s:
                if ba[x + a]:
                    ok = False
                    break
            if ok:
                s.append(x)
                ba[2 * x] = 1
                for a in s[:-1]:
                    ba[x + a] = 1
        s_max, ba_max = make_maximal_ba(s, ba)
        s_min, ba_min = local_search_minimize_ba(s_max, ba_max)
        if is_sidon_ba(s_min, ba_min) and is_maximal_ba(s_min, ba_min):
            return s_min
        return s_max

    return best_result
# EVOLVE-BLOCK-END
