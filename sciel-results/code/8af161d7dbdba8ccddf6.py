import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    N = 32
    MAX_VAL = 512
    
    def score_set(S):
        """Returns the number of distinct differences d."""
        S_set = set(S)
        diffs = set()
        # To optimize, we can iterate over pairs
        # But N=32 is small, so O(N^2) is fine
        for i in range(N):
            for j in range(i + 1, N):
                a = S[i]
                b = S[j]
                d = b - a
                c = b + d
                if c in S_set:
                    diffs.add(d)
        return len(diffs)
    
    def get_best_solution():
        best_score = 0
        best_set = None
        
        # Heuristic 1: Interval
        s1 = list(range(1, N + 1))
        score1 = score_set(s1)
        if score1 > best_score:
            best_score = score1
            best_set = s1[:]
            
        # Heuristic 2: Random sets
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            s = random.sample(range(1, MAX_VAL + 1), N)
            sc = score_set(s)
            if sc > best_score:
                best_score = sc
                best_set = s[:]
                
        # Simulated Annealing
        # Start from best found so far
        current_set = best_set[:]
        current_score = best_score
        
        # SA Parameters
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9995
        max_iters = 50000
        
        for i in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Generate neighbor
            new_set = current_set[:]
            idx = random.randint(0, N - 1)
            # Try to pick a value that might help
            # Random value
            new_val = random.randint(1, MAX_VAL)
            if new_val in new_set:
                # Avoid duplicates, pick another
                new_val = random.randint(1, MAX_VAL)
                while new_val in new_set:
                    new_val = random.randint(1, MAX_VAL)
            
            new_set[idx] = new_val
            new_score = score_set(new_set)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_set = new_set
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Local Polish: Try to improve by changing one element to a "good" candidate
        # Candidates: Values that form a progression with two existing elements
        # For each pair (a, b) in best_set, potential c = 2b-a or 2a-b.
        # We check if replacing any element with c improves the score.
        
        improved = True
        polish_passes = 0
        while improved and time.time() < deadline - 2 and polish_passes < 5:
            improved = False
            polish_passes += 1
            s = best_set[:]
            s_set = set(s)
            
            # Generate candidates
            candidates = set()
            for i in range(N):
                for j in range(i + 1, N):
                    a = s[i]
                    b = s[j]
                    # a, b, c => c = 2b - a
                    c1 = 2 * b - a
                    if 1 <= c1 <= MAX_VAL:
                        candidates.add(c1)
                    # c, a, b => c = 2a - b
                    c2 = 2 * a - b
                    if 1 <= c2 <= MAX_VAL:
                        candidates.add(c2)
                    # a, c, b => c = (a+b)/2 (if integer)
                    if (a + b) % 2 == 0:
                        c3 = (a + b) // 2
                        if 1 <= c3 <= MAX_VAL:
                            candidates.add(c3)
            
            # Try replacing each element with each candidate
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                old_val = s[i]
                for c in candidates:
                    if c == old_val:
                        continue
                    if c in s_set:
                        # If c is already in set, we can't just add it, we replace old_val.
                        # But if c is in set, and old_val is not c, then we remove old_val and add c?
                        # No, we replace s[i] with c. If c is already in s, we have a duplicate.
                        # So we must ensure c is not in s_set - {old_val}
                        if c in s_set and c != old_val:
                            continue
                    
                    new_s = s[:]
                    new_s[i] = c
                    new_score = score_set(new_s)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_set = new_s
                        s = new_s
                        s_set = set(s)
                        improved = True
                        break # Restart candidate loop for this i? Or just break inner?
                        # Actually, if we improved, we should update s and s_set and continue checking other replacements for the new s
                        # But to keep it simple, we break and restart the whole polish pass?
                        # Or just continue with the updated s.
                        # Let's just break the candidate loop for this i and continue to next i with updated s.
                        # But s_set needs updating.
                        # This is getting complicated. Let's just do a simple hill climb on the best_set found.
        
        # Final simple hill climb
        # For each position, try all values 1..512? Too slow.
        # Try values from candidates + neighbors.
        
        return best_set

    return get_best_solution()
