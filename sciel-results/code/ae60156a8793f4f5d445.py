"""
Design: Greedy construction + local search to maximize distinct 3-term AP differences.

Strategy:
1. Start with a random initial set of 32 integers in 1..512.
2. Use local search (swap-based) to maximize the number of distinct d > 0
   such that there exists a, a+d, a+2d all in the set.
3. Also try a greedy construction that prioritizes adding elements that
   create new differences.
4. Keep the best result found within the time budget.
"""

import time
import random
import itertools

DEADLINE = 140  # safety margin before 150s kill


def _score_set(s):
    """Return the number of distinct d > 0 for which a, a+d, a+2d are all in s."""
    s_set = set(s)
    diffs = set()
    for a in s_set:
        for d in range(1, 513 - a):
            if (a + d) in s_set and (a + 2 * d) in s_set:
                diffs.add(d)
    return len(diffs)


def _score_set_fast(s):
    """Faster scoring using sorted list and set lookups."""
    s_set = set(s)
    diffs = set()
    for a in s_set:
        # a + 2*d <= 512 => d <= (512 - a) // 2
        max_d = (512 - a) // 2
        for d in range(1, max_d + 1):
            if (a + d) in s_set and (a + 2 * d) in s_set:
                diffs.add(d)
    return len(diffs)


def _local_search(s, time_budget):
    """
    Local search: repeatedly try swapping one element with another random value
    in 1..512 to improve the score.
    """
    s = list(s)
    s_set = set(s)
    best_score = _score_set_fast(s)
    start_time = time.time()
    
    # Precompute which values are in range
    all_vals = list(range(1, 513))
    
    while time.time() - start_time < time_budget:
        improved = False
        random.shuffle(s)  # randomize order for diversity
        for i in range(32):
            old_val = s[i]
            # Try replacing s[i] with a random value not in the set
            for attempt in range(50):
                new_val = random.choice(all_vals)
                if new_val == old_val or new_val in s_set:
                    continue
                # Tentatively swap
                s[i] = new_val
                s_set.remove(old_val)
                s_set.add(new_val)
                new_score = _score_set_fast(s)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    s[i] = old_val
                    s_set.remove(new_val)
                    s_set.add(old_val)
            if time.time() - start_time >= time_budget:
                break
        if not improved:
            # Try a few more rounds with different randomization
            for _ in range(3):
                improved = False
                for i in range(32):
                    old_val = s[i]
                    for attempt in range(100):
                        new_val = random.choice(all_vals)
                        if new_val == old_val or new_val in s_set:
                            continue
                        s[i] = new_val
                        s_set.remove(old_val)
                        s_set.add(new_val)
                        new_score = _score_set_fast(s)
                        if new_score > best_score:
                            best_score = new_score
                            improved = True
                            break
                        else:
                            s[i] = old_val
                            s_set.remove(new_val)
                            s_set.add(old_val)
                    if time.time() - start_time >= time_budget:
                        break
                if improved:
                    break
            if not improved:
                break
    return s, best_score


def _greedy_construct():
    """
    Greedy: start with a few elements, then add elements that maximize
    the number of new differences created.
    """
    s = set()
    # Start with a small random seed
    seed_vals = random.sample(range(1, 513), 8)
    s = set(seed_vals)
    
    for _ in range(24):
        best_val = None
        best_score = _score_set_fast(list(s))
        # Try candidates
        candidates = [v for v in range(1, 513) if v not in s]
        random.shuffle(candidates)
        # Evaluate a subset of candidates for speed
        for v in candidates[:200]:
            s.add(v)
            sc = _score_set_fast(list(s))
            s.remove(v)
            if sc > best_score:
                best_score = sc
                best_val = v
        if best_val is not None:
            s.add(best_val)
        else:
            # Just add a random one
            s.add(random.choice(candidates))
    return list(s)


def construct():
    start_time = time.time()
    random.seed(42)
    
    best_set = list(range(1, 33))
    best_score = _score_set_fast(best_set)
    
    # Phase 1: Try several random initializations with local search
    while time.time() - start_time < DEADLINE - 20:
        # Random initial set
        init = random.sample(range(1, 513), 32)
        init_score = _score_set_fast(init)
        
        # Local search with remaining budget
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining < 2:
            break
        
        # Use a fraction of remaining time for this local search
        ls_time = min(remaining * 0.7, 30)
        s, sc = _local_search(init, ls_time)
        
        if sc > best_score:
            best_score = sc
            best_set = s
        
        # Also try greedy construction
        if time.time() - start_time < DEADLINE - 10:
            g = _greedy_construct()
            gsc = _score_set_fast(g)
            if gsc > best_score:
                best_score = gsc
                best_set = g
            
            # Local search on greedy result
            remaining = DEADLINE - (time.time() - start_time) - 5
            if remaining > 5:
                ls_time = min(remaining * 0.5, 20)
                s2, sc2 = _local_search(g, ls_time)
                if sc2 > best_score:
                    best_score = sc2
                    best_set = s2
    
    # Final: try a few more local search rounds on the best set if time permits
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 3:
        s_final, sc_final = _local_search(best_set, remaining - 2)
        if sc_final > best_score:
            best_set = s_final
            best_score = sc_final
    
    return list(best_set)


# Test
if __name__ == "__main__":
    result = construct()
    assert len(result) == 32
    assert len(set(result)) == 32
    assert all(1 <= x <= 512 for x in result)
    print(f"Score: {_score_set_fast(result)}")
    print(f"Set: {sorted(result)}")
