import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try to find a solution for n=34, then 33, etc.
    # We prioritize higher n.
    
    for n in range(34, 5, -1):
        if time.time() > deadline - 5:
            break
            
        # Precompute triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s and their incidence
        # k4s: list of lists of triple indices
        # k5s: list of lists of triple indices
        
        # For efficiency in Min-Conflicts, we need:
        # 1. For each triple, which K4s and K5s it belongs to.
        # 2. For each K4/K5, the count of red/blue edges.
        
        # Let's build the incidence first.
        # This is O(n^4) and O(n^5) which is fine for n=34.
        
        k4s = []
        for verts in combinations(range(n), 4):
            trs = [triple_to_idx[(a,b,c)] for a,b,c in combinations(verts, 3)]
            k4s.append(trs)
            
        k5s = []
        for verts in combinations(range(n), 5):
            trs = [triple_to_idx[(a,b,c)] for a,b,c in combinations(verts, 3)]
            k5s.append(trs)
            
        num_k4s = len(k4s)
        num_k5s = len(k5s)
        
        # Incidence: triple -> list of k4 indices, list of k5 indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        for i, trs in enumerate(k4s):
            for t in trs:
                triple_to_k4s[t].append(i)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, trs in enumerate(k5s):
            for t in trs:
                triple_to_k5s[t].append(i)
                
        # Min-Conflicts Search
        def min_conflicts_search():
            # Random initialization
            colors = np.random.randint(0, 2, size=num_triples)
            
            # Initialize counts
            # red_count_k4[i] = number of red edges in k4s[i]
            red_count_k4 = np.zeros(num_k4s, dtype=np.int32)
            # blue_count_k5[j] = number of blue edges in k5s[j]
            blue_count_k5 = np.zeros(num_k5s, dtype=np.int32)
            
            # Initial counts
            for t in range(num_triples):
                c = colors[t]
                for k4_idx in triple_to_k4s[t]:
                    if c == 0:
                        red_count_k4[k4_idx] += 1
                for k5_idx in triple_to_k5s[t]:
                    if c == 1:
                        blue_count_k5[k5_idx] += 1
                        
            # Initial conflict counts
            num_red_k4 = np.sum(red_count_k4 == 4)
            num_blue_k5 = np.sum(blue_count_k5 == 10)
            
            total_conflicts = num_red_k4 + num_blue_k5
            
            if total_conflicts == 0:
                return colors.copy()
                
            # Track conflicts for each triple?
            # A triple is in conflict if it is part of any red K4 or blue K5.
            # We can maintain a boolean array `in_conflict` or a count.
            # Let's maintain `conflict_count[t]` = number of monochromatic cliques containing t.
            # But updating this is O(degree).
            # Simpler: Just pick a random triple that is in conflict.
            # To find them efficiently, we can maintain a set/list of conflicting triples.
            
            # Let's use a simpler heuristic: 
            # 1. Identify all triples that are part of at least one conflicting clique.
            # 2. Pick one at random.
            # 3. Flip it.
            # 4. Update counts.
            
            # To avoid O(N) scan for conflicting triples, we can maintain a list.
            # But maintaining the list incrementally is tricky.
            # Given N=6000, scanning is fast.
            
            max_iters = 50000
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                    
                # Find a conflicting triple
                # A triple t is in conflict if:
                # - It is red and part of a K4 with 4 reds.
                # - It is blue and part of a K5 with 10 blues.
                
                # We can iterate through K4s and K5s to find conflicts?
                # No, that's slow.
                
                # Let's maintain `triple_conflict_count` array.
                # If count > 0, it's in conflict.
                
                # Actually, let's just pick a random triple and check if it's in conflict?
                # No, that's inefficient if conflicts are rare.
                
                # Better: Maintain a list of indices of triples that are in conflict.
                # When we flip a triple, we update the conflict status of that triple and its neighbors?
                # No, flipping t changes the status of K4s/K5s containing t.
                # This changes the conflict status of ALL triples in those K4s/K5s.
                # That's too many updates.
                
                # Let's stick to:
                # 1. Pick a random triple t.
                # 2. Check if it is in conflict.
                # 3. If yes, flip.
                # 4. If no, flip anyway? (Random walk)
                # Min-Conflicts usually picks from the conflicting set.
                
                # Let's optimize the "find conflicting triple" step.
                # We can keep a list `conflicting_triples`.
                # Initially, populate it.
                # When flipping t:
                #   - t might leave or enter conflict.
                #   - Neighbors of t (in K4s/K5s) might leave or enter conflict.
                #   - This requires updating the list.
                
                # Given the complexity, let's try a simpler approach:
                # Randomly pick a triple. If it reduces total conflicts, flip.
                # If not, flip with some probability (SA-like) or just flip if it's "bad".
                
                # Let's try: Pick random triple. Calculate delta.
                # If delta < 0, accept.
                # If delta == 0, accept with prob 0.5.
                # If delta > 0, accept with prob exp(-delta/T).
                
                # This is SA. We know SA works but is slow.
                # Let's try to make SA faster by using the precomputed counts.
                
                t = random.randint(0, num_triples - 1)
                c = colors[t]
                
                delta = 0
                
                # Check K4s
                for k4_idx in triple_to_k4s[t]:
                    rc = red_count_k4[k4_idx]
                    if c == 0:
                        # t is red, will become blue
                        # rc decreases by 1
                        if rc == 4:
                            delta -= 1 # Remove a red K4
                        new_rc = rc - 1
                        if new_rc == 4:
                            delta += 1 # Create a red K4 (impossible if rc was 4 and we remove 1? No, rc was 4, new is 3. rc was 3, new is 2. rc was 5? No, max 4.)
                        # Wait, if rc==4, it was a conflict. After flip, rc=3, no conflict. Delta -1.
                        # If rc==3, it was not conflict. After flip, rc=2, not conflict. Delta 0.
                        # So only rc==4 matters for removal.
                        # Can we create a red K4? Only if rc was 3 and we ADD a red.
                        # But we are REMOVING a red (c=0 -> 1).
                        # So we only REMOVE red K4s.
                    else:
                        # t is blue, will become red
                        # rc increases by 1
                        new_rc = rc + 1
                        if new_rc == 4:
                            delta += 1 # Create a red K4
                        # Can we remove? No, rc increases.
                
                # Check K5s
                for k5_idx in triple_to_k5s[t]:
                    bc = blue_count_k5[k5_idx]
                    if c == 1:
                        # t is blue, will become red
                        # bc decreases by 1
                        if bc == 10:
                            delta -= 1 # Remove a blue K5
                    else:
                        # t is red, will become blue
                        # bc increases by 1
                        new_bc = bc + 1
                        if new_bc == 10:
                            delta += 1 # Create a blue K5
                
                # Acceptance
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.5
                else:
                    # Simple: accept with low prob if small, else no
                    accept = random.random() < 0.1
                    
                if accept:
                    colors[t] = 1 - c
                    # Update counts
                    if c == 0:
                        for k4_idx in triple_to_k4s[t]:
                            red_count_k4[k4_idx] -= 1
                        for k5_idx in triple_to_k5s[t]:
                            blue_count_k5[k5_idx] += 1
                    else:
                        for k4_idx in triple_to_k4s[t]:
                            red_count_k4[k4_idx] += 1
                        for k5_idx in triple_to_k5s[t]:
                            blue_count_k5[k5_idx] -= 1
                            
                    total_conflicts += delta
                    if total_conflicts == 0:
                        return colors.copy()
                        
            return None

        # Run a few restarts
        found = None
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            res = min_conflicts_search()
            if res is not None:
                found = res
                break
                
        if found is not None:
            return {'n': n, 'colors': found.tolist()}
            
    # Fallback
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
