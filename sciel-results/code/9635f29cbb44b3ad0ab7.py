import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    def score_set(lst):
        """Count distinct d > 0 such that a, a+d, a+2d all in set."""
        s = set(lst)
        valid_diffs = set()
        for a in s:
            for d in range(1, (MAX_VAL - a) // 2 + 1):
                if (a + d) in s and (a + 2 * d) in s:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def gaps_to_set(gaps, offset=1):
        """Convert 31 gaps to a 32-element set starting at offset."""
        s = [offset]
        cur = offset
        for g in gaps:
            cur += g
            s.append(cur)
        return s
    
    def set_to_gaps(lst):
        """Convert sorted list to 31 gaps."""
        lst = sorted(lst)
        return [lst[i+1] - lst[i] for i in range(len(lst)-1)]
    
    def score_gaps(gaps, offset=1):
        """Score a gap sequence by converting to set and counting AP diffs."""
        s = gaps_to_set(gaps, offset)
        if s[-1] > MAX_VAL:
            return 0
        return score_set(s)
    
    def score_gaps_fast(gaps, offset=1):
        """
        Faster scoring: compute all partial sums, then all pairwise differences,
        and count how many distinct positive differences d appear such that
        there exist i < j < k with S_k - S_j == S_j - S_i == d.
        Actually, the score is the number of distinct d > 0 for which there exists
        a, a+d, a+2d in the set. This is equivalent to: d is in the set of
        differences if there exist indices i < j < k such that S_j - S_i = d 
        and S_k - S_j = d. So d = S_j - S_i = S_k - S_j, meaning S_i, S_j, S_k 
        form an AP with common difference d.
        
        Alternative: for each pair (i, k) with i < k, check if (S_i + S_k) / 2 
        is in the set. If so, d = (S_k - S_i) / 2 is a valid difference (if integer).
        """
        s = gaps_to_set(gaps, offset)
        if s[-1] > MAX_VAL:
            return 0
        s_set = set(s)
        valid = set()
        # For each pair (i, k), i < k, check if midpoint is in set
        n = len(s)
        for i in range(n):
            for k in range(i + 2, n):
                diff = s[k] - s[i]
                if diff % 2 == 0:
                    mid = (s[i] + s[k]) // 2
                    if mid in s_set:
                        valid.add(diff // 2)
        return len(valid)
    
    # Strategy: 
    # 1. Generate diverse initial gap sequences
    # 2. Local search: swap adjacent gaps, or swap any two gaps, or move a gap
    # 3. Keep best
    
    def generate_initial_gaps():
        """Generate diverse initial gap sequences."""
        gaps_list = []
        
        # 1. Constant-ish gaps with small perturbations
        random.seed(42)
        for base in range(5, 20):
            gaps = [base] * 31
            # Perturb
            for _ in range(10):
                i = random.randint(0, 30)
                gaps[i] = max(1, gaps[i] + random.randint(-3, 3))
            total = sum(gaps)
            if total + 1 <= MAX_VAL:
                gaps_list.append(gaps[:])
        
        # 2. Arithmetic progression of gaps
        for start in range(2, 10):
            for step in range(1, 5):
                gaps = [start + i * step for i in range(31)]
                if sum(gaps) + 1 <= MAX_VAL:
                    gaps_list.append(gaps[:])
        
        # 3. Random gaps
        for _ in range(30):
            gaps = [random.randint(1, 15) for _ in range(31)]
            total = sum(gaps)
            if total + 1 <= MAX_VAL:
                gaps_list.append(gaps[:])
            else:
                # Scale down
                scale = (MAX_VAL - 1) / total
                gaps = [max(1, int(g * scale)) for g in gaps]
                # Fix sum
                diff = (MAX_VAL - 1) - sum(gaps)
                if diff > 0:
                    gaps[random.randint(0, 30)] += diff
                elif diff < 0:
                    # Reduce some gaps
                    i = 0
                    while diff < 0 and i < 31:
                        reduce = min(gaps[i] - 1, -diff)
                        gaps[i] -= reduce
                        diff += reduce
                        i += 1
                if sum(gaps) + 1 <= MAX_VAL:
                    gaps_list.append(gaps[:])
        
        # 4. Structured: mix of small and large gaps
        for _ in range(20):
            gaps = []
            for i in range(31):
                if i % 5 == 0:
                    gaps.append(random.randint(5, 15))
                else:
                    gaps.append(random.randint(1, 5))
            total = sum(gaps)
            if total + 1 <= MAX_VAL:
                gaps_list.append(gaps[:])
        
        return gaps_list
    
    def local_search_gaps(initial_gaps, time_limit):
        """
        Local search on gap sequences.
        Moves:
        1. Swap any two gaps
        2. Change a gap by +/- 1 (if valid)
        3. Rotate: move a gap from position i to position j
        """
        current_gaps = initial_gaps[:]
        current_score = score_gaps_fast(current_gaps)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_new_score = current_score
            best_new_gaps = None
            
            # Try all swaps
            for i in range(31):
                for j in range(i + 1, 31):
                    new_gaps = current_gaps[:]
                    new_gaps[i], new_gaps[j] = new_gaps[j], new_gaps[i]
                    new_score = score_gaps_fast(new_gaps)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_gaps = new_gaps
                        improved = True
            
            # Try +/- 1 changes
            if not improved:
                for i in range(31):
                    for delta in [-1, 1]:
                        new_gaps = current_gaps[:]
                        new_gaps[i] = current_gaps[i] + delta
                        if new_gaps[i] < 1:
                            continue
                        total = sum(new_gaps)
                        if total + 1 > MAX_VAL:
                            continue
                        new_score = score_gaps_fast(new_gaps)
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_gaps = new_gaps
                            improved = True
            
            # Try moves (remove from i, insert at j)
            if not improved:
                for i in range(31):
                    for j in range(31):
                        if i == j:
                            continue
                        new_gaps = current_gaps[:]
                        val = new_gaps.pop(i)
                        new_gaps.insert(j, val)
                        total = sum(new_gaps)
                        if total + 1 > MAX_VAL:
                            continue
                        new_score = score_gaps_fast(new_gaps)
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_gaps = new_gaps
                            improved = True
            
            if improved:
                current_gaps = best_new_gaps
                current_score = best_new_score
        
        return current_gaps, current_score
    
    # Main loop
    initial_gaps_list = generate_initial_gaps()
    
    best_overall_score = 0
    best_overall_gaps = None
    
    # Sort by initial score to prioritize
    scored_starts = []
    for gaps in initial_gaps_list:
        sc = score_gaps_fast(gaps)
        scored_starts.append((sc, gaps))
    scored_starts.sort(key=lambda x: -x[0])
    
    for idx, (init_score, init_gaps) in enumerate(scored_starts):
        if time.time() > deadline - 5.0:
            break
        
        time_per_start = max(0.3, 3.0 - idx * 0.05)
        local_deadline = time.time() + time_per_start
        
        gaps, score = local_search_gaps(init_gaps, local_deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_gaps = gaps[:]
    
    # Final polish on best
    if time.time() < deadline - 2.0 and best_overall_gaps:
        final_deadline = deadline - 1.0
        best_overall_gaps, best_overall_score = local_search_gaps(best_overall_gaps, final_deadline)
    
    # Convert to set
    if best_overall_gaps:
        result = gaps_to_set(best_overall_gaps, offset=1)
        # Ensure validity
        if len(result) == 32 and len(set(result)) == 32 and all(1 <= x <= MAX_VAL for x in result):
            return result
    
    # Fallback: return a valid set
    return list(range(1, 33))

if __name__ == "__main__":
    res = construct()
    assert len(res) == 32
    assert len(set(res)) == 32
    assert all(1 <= x <= 512 for x in res)
    print("Score:", score_set(res))
