import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return
    
    # Precompute structures for a given n
    def get_structures(n):
        triples = list(combinations(range(n), 3))
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k4_triples = []
        for k4 in k4s:
            ts = []
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        t = (k4[i], k4[j], k4[k])
                        ts.append(triple_idx[t])
            k4_triples.append(ts)
            
        k5s = list(combinations(range(n), 5))
        k5_triples = []
        for k5 in k5s:
            ts = []
            for t in combinations(k5, 3):
                ts.append(triple_idx[t])
            k5_triples.append(ts)
            
        # Map triple index to list of k4 indices and k5 indices it belongs to
        triple_to_k4 = [[] for _ in range(len(triples))]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(len(triples))]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
                
        return triples, triple_idx, k4_triples, k5_triples, triple_to_k4, triple_to_k5

    def try_solve(n, time_limit):
        if n < 4: return None
        triples, triple_idx, k4_triples, k5_triples, triple_to_k4, triple_to_k5 = get_structures(n)
        num_triples = len(triples)
        num_k4 = len(k4_triples)
        num_k5 = len(k5_triples)
        
        # Initialize random coloring
        # 0: Red, 1: Blue
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        # red_k4_count[i] = number of red triples in k4s[i]
        # blue_k5_count[i] = number of blue triples in k5s[i]
        red_k4_count = [0] * num_k4
        blue_k5_count = [0] * num_k5
        
        for i, ts in enumerate(k4_triples):
            cnt = 0
            for t_idx in ts:
                if colors[t_idx] == 0:
                    cnt += 1
            red_k4_count[i] = cnt
            
        for i, ts in enumerate(k5_triples):
            cnt = 0
            for t_idx in ts:
                if colors[t_idx] == 1:
                    cnt += 1
            blue_k5_count[i] = cnt
            
        # Count bad sets
        bad_k4s = [i for i in range(num_k4) if red_k4_count[i] == 4]
        bad_k5s = [i for i in range(num_k5) if blue_k5_count[i] == 10]
        
        if not bad_k4s and not bad_k5s:
            return colors
            
        # Local Search
        # We maintain lists of bad sets for quick selection
        # To be efficient, we can just pick a random bad set from the list.
        # If the list is large, we might want to sample.
        
        # We will restart if we get stuck or if progress is slow?
        # For now, just run until time limit.
        
        while time.time() < time_limit:
            if not bad_k4s and not bad_k5s:
                return colors
                
            # Pick a bad set
            if bad_k4s and (not bad_k5s or random.random() < 0.5):
                # Pick random bad K4
                k4_idx = random.choice(bad_k4s)
                ts = k4_triples[k4_idx]
                # Pick a random triple in this K4 to flip (0->1)
                t_idx = random.choice(ts)
                # Flip 0 -> 1
                colors[t_idx] = 1
                
                # Update counts
                # This triple was Red (0), now Blue (1)
                # For all K4s containing t_idx:
                for k4_i in triple_to_k4[t_idx]:
                    old_cnt = red_k4_count[k4_i]
                    red_k4_count[k4_i] = old_cnt - 1
                    # Check badness change
                    if old_cnt == 4 and red_k4_count[k4_i] != 4:
                        # Was bad, now not bad
                        if k4_i in bad_k4s:
                            bad_k4s.remove(k4_i) # This is O(N) and slow!
                    elif old_cnt != 4 and red_k4_count[k4_i] == 4:
                        # Was not bad, now bad
                        bad_k4s.append(k4_i)
                        
                # For all K5s containing t_idx:
                for k5_i in triple_to_k5[t_idx]:
                    old_cnt = blue_k5_count[k5_i]
                    blue_k5_count[k5_i] = old_cnt + 1
                    if old_cnt == 10 and blue_k5_count[k5_i] != 10:
                        if k5_i in bad_k5s:
                            bad_k5s.remove(k5_i)
                    elif old_cnt != 10 and blue_k5_count[k5_i] == 10:
                        bad_k5s.append(k5_i)
                        
            else:
                # Pick random bad K5
                if not bad_k5s: continue # Should not happen if bad_k4s empty
                k5_idx = random.choice(bad_k5s)
                ts = k5_triples[k5_idx]
                # Pick a random triple in this K5 to flip (1->0)
                t_idx = random.choice(ts)
                # Flip 1 -> 0
                colors[t_idx] = 0
                
                # Update counts
                # This triple was Blue (1), now Red (0)
                for k4_i in triple_to_k4[t_idx]:
                    old_cnt = red_k4_count[k4_i]
                    red_k4_count[k4_i] = old_cnt + 1
                    if old_cnt == 4 and red_k4_count[k4_i] != 4:
                        if k4_i in bad_k4s:
                            bad_k4s.remove(k4_i)
                    elif old_cnt != 4 and red_k4_count[k4_i] == 4:
                        bad_k4s.append(k4_i)
                        
                for k5_i in triple_to_k5[t_idx]:
                    old_cnt = blue_k5_count[k5_i]
                    blue_k5_count[k5_i] = old_cnt - 1
                    if old_cnt == 10 and blue_k5_count[k5_i] != 10:
                        if k5_i in bad_k5s:
                            bad_k5s.remove(k5_i)
                    elif old_cnt != 10 and blue_k5_count[k5_i] == 10:
                        bad_k5s.append(k5_i)
                        
        return None # Timeout

    # Try n=34, 33, 32...
    # We can allocate time based on difficulty.
    # 34 is the goal.
    
    # To handle the O(N) remove issue, we can use a set for bad sets?
    # Or just accept the slowness? 
    # Number of bad sets is usually small near the end, but large at start.
    # At start, almost all K4s/K5s might be bad.
    # Removing from a list is bad.
    # Let's use a set for bad indices.
    
    def try_solve_v2(n, time_limit):
        if n < 4: return None
        triples, triple_idx, k4_triples, k5_triples, triple_to_k4, triple_to_k5 = get_structures(n)
        num_triples = len(triples)
        num_k4 = len(k4_triples)
        num_k5 = len(k5_triples)
        
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        red_k4_count = [0] * num_k4
        blue_k5_count = [0] * num_k5
        
        for i, ts in enumerate(k4_triples):
            cnt = 0
            for t_idx in ts:
                if colors[t_idx] == 0:
                    cnt += 1
            red_k4_count[i] = cnt
            
        for i, ts in enumerate(k5_triples):
            cnt = 0
            for t_idx in ts:
                if colors[t_idx] == 1:
                    cnt += 1
            blue_k5_count[i] = cnt
            
        bad_k4s = set(i for i in range(num_k4) if red_k4_count[i] == 4)
        bad_k5s = set(i for i in range(num_k5) if blue_k5_count[i] == 10)
        
        if not bad_k4s and not bad_k5s:
            return colors
            
        # Use a list for random access, but update set?
        # Or just iterate over set?
        # random.choice requires sequence.
        # We can keep a list of bad sets, but lazily delete?
        # Or just use random.sample?
        
        # Let's stick to a simple approach:
        # Keep lists, but instead of remove, we can just filter occasionally?
        # No, let's use a set for membership, and a list for random selection.
        # When we update, we add to set/list if becomes bad.
        # If becomes not bad, remove from set. 
        # For list, we can just leave it and check validity on pick?
        
        bad_k4_list = list(bad_k4s)
        bad_k5_list = list(bad_k5s)
        
        while time.time() < time_limit:
            if not bad_k4s and not bad_k5s:
                return colors
                
            # Clean up lists if they are much larger than sets?
            # Or just pick from set directly if small?
            # If set is large, random.sample(set, 1) is slow?
            # Let's just pick from list, and verify.
            
            if bad_k4s and (not bad_k5s or random.random() < 0.5):
                # Pick from bad_k4_list
                if not bad_k4_list:
                    # Resync
                    bad_k4_list = list(bad_k4s)
                # Try a few times to find a valid bad set
                for _ in range(10):
                    if not bad_k4_list: break
                    k4_idx = random.choice(bad_k4_list)
                    if k4_idx in bad_k4s:
                        break
                else:
                    continue
                    
                ts = k4_triples[k4_idx]
                t_idx = random.choice(ts)
                # Flip 0 -> 1
                if colors[t_idx] != 0: continue # Should be 0 if k4 is bad (all red)
                colors[t_idx] = 1
                
                for k4_i in triple_to_k4[t_idx]:
                    old_cnt = red_k4_count[k4_i]
                    red_k4_count[k4_i] = old_cnt - 1
                    if old_cnt == 4 and red_k4_count[k4_i] != 4:
                        bad_k4s.discard(k4_i)
                    elif old_cnt != 4 and red_k4_count[k4_i] == 4:
                        bad_k4s.add(k4_i)
                        bad_k4_list.append(k4_i)
                        
                for k5_i in triple_to_k5[t_idx]:
                    old_cnt = blue_k5_count[k5_i]
                    blue_k5_count[k5_i] = old_cnt + 1
                    if old_cnt == 10 and blue_k5_count[k5_i] != 10:
                        bad_k5s.discard(k5_i)
                    elif old_cnt != 10 and blue_k5_count[k5_i] == 10:
                        bad_k5s.add(k5_i)
                        bad_k5_list.append(k5_i)
                        
            else:
                if not bad_k5s: continue
                if not bad_k5_list:
                    bad_k5_list = list(bad_k5s)
                for _ in range(10):
                    if not bad_k5_list: break
                    k5_idx = random.choice(bad_k5_list)
                    if k5_idx in bad_k5s:
                        break
                else:
                    continue
                    
                ts = k5_triples[k5_idx]
                t_idx = random.choice(ts)
                # Flip 1 -> 0
                if colors[t_idx] != 1: continue
                colors[t_idx] = 0
                
                for k4_i in triple_to_k4[t_idx]:
                    old_cnt = red_k4_count[k4_i]
                    red_k4_count[k4_i] = old_cnt + 1
                    if old_cnt == 4 and red_k4_count[k4_i] != 4:
                        bad_k4s.discard(k4_i)
                    elif old_cnt != 4 and red_k4_count[k4_i] == 4:
                        bad_k4s.add(k4_i)
                        bad_k4_list.append(k4_i)
                        
                for k5_i in triple_to_k5[t_idx]:
                    old_cnt = blue_k5_count[k5_i]
                    blue_k5_count[k5_i] = old_cnt - 1
                    if old_cnt == 10 and blue_k5_count[k5_i] != 10:
                        bad_k5s.discard(k5_i)
                    elif old_cnt != 10 and blue_k5_count[k5_i] == 10:
                        bad_k5s.add(k5_i)
                        bad_k5_list.append(k5_i)
                        
        return None

    # Try to solve for n=34
    # If it fails, try 33, etc.
    # We have ~110s.
    # Let's give 34 most of the time.
    
    # Heuristic: Start with 34. If not solved in 100s, give up and try 33?
    # Or just return the best found?
    # The task requires a VALID coloring.
    # So if we don't find one for 34, we must find one for smaller n.
    
    # We can run multiple restarts for 34.
    
    best_n = 0
    best_colors = None
    
    # Try 34
    t_start = time.time()
    colors = try_solve_v2(34, t_start + 100)
    if colors:
        return {'n': 34, 'colors': colors}
        
    # If 34 failed, try 33 with remaining time
    t_start = time.time()
    remaining = deadline - t_start
    if remaining > 5:
        colors = try_solve_v2(33, t_start + remaining - 2)
        if colors:
            return {'n': 33, 'colors': colors}
            
    # Fallback: Use the seed solution for n=9 (known valid)
    # Or try to construct a small one quickly?
    # The seed uses n=9 with sum mod 3.
    # Let's just return the seed solution if we fail everything.
    # But wait, we can try to solve n=10, 11... quickly?
    # Actually, the seed is valid.
    
    # Let's verify the seed logic is valid.
    # n=9, color 0 if sum%3==0 else 1.
    # Is it valid?
    # The prompt says "The published construction has 34 points".
    # The seed program returns n=9.
    # If my search fails, I should return a valid coloring.
    # I will implement a quick check for the seed coloring and return it if valid.
    
    def check_validity(n, colors):
        triples = list(combinations(range(n), 3))
        t_idx = {t: i for i, t in enumerate(triples)}
        # Check Red K4
        for k4 in combinations(range(n), 4):
            cnt = 0
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        t = (k4[i], k4[j], k4[k])
                        if colors[t_idx[t]] == 0:
                            cnt += 1
            if cnt == 4:
                return False
        # Check Blue K5
        for k5 in combinations(range(n), 5):
            cnt = 0
            for t in combinations(k5, 3):
                if colors[t_idx[t]] == 1:
                    cnt += 1
            if cnt == 10:
                return False
        return True

    # Fallback to seed
    n = 9
    seed_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    if check_validity(n, seed_colors):
        return {'n': n, 'colors': seed_colors}
        
    # If even seed fails (unlikely), return random valid small n?
    # n=4: 4 triples. No K4 red? K4 is the whole set. If all red, bad.
    # So we need at least one blue.
    # n=5: No K5 blue? K5 is whole set. If all blue, bad.
    
    # Just return the seed, it should be valid.
    return {'n': 9, 'colors': seed_colors}
