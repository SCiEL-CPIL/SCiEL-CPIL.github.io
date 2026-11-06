import itertools
import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Try to find a valid coloring for n=34, then 33, etc.
    # We prioritize finding a valid one for the largest n.
    
    def try_find_coloring(n):
        """
        Attempts to find a valid coloring for n vertices.
        Returns (colors, n) if found, else (None, n).
        """
        if n < 4:
            # Trivially valid
            return [0] * 0, n
            
        triples = list(itertools.combinations(range(n), 3))
        M = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute 4-sets and their edges
        four_sets = list(itertools.combinations(range(n), 4))
        four_sets_edges = []
        edge_to_four_sets = [[] for _ in range(M)]
        for i, s in enumerate(four_sets):
            edges = []
            for t in itertools.combinations(s, 3):
                idx = triple_to_idx[t]
                edges.append(idx)
                edge_to_four_sets[idx].append(i)
            four_sets_edges.append(edges)
            
        # Precompute 5-sets and their edges
        five_sets = list(itertools.combinations(range(n), 5))
        five_sets_edges = []
        edge_to_five_sets = [[] for _ in range(M)]
        for i, s in enumerate(five_sets):
            edges = []
            for t in itertools.combinations(s, 3):
                idx = triple_to_idx[t]
                edges.append(idx)
                edge_to_five_sets[idx].append(i)
            five_sets_edges.append(edges)
            
        # Initialize colors
        # Heuristic: color based on sum mod 3
        # 0 if (a+b+c)%3 == 0 else 1
        colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples]
        
        # Initialize counts
        red_count_4 = [0] * len(four_sets)
        blue_count_5 = [0] * len(five_sets)
        
        for i, edges in enumerate(four_sets_edges):
            cnt = 0
            for e in edges:
                if colors[e] == 0:
                    cnt += 1
            red_count_4[i] = cnt
            
        for i, edges in enumerate(five_sets_edges):
            cnt = 0
            for e in edges:
                if colors[e] == 1:
                    cnt += 1
            blue_count_5[i] = cnt
            
        viol_4_count = sum(1 for c in red_count_4 if c == 4)
        viol_5_count = sum(1 for c in blue_count_5 if c == 10)
        
        if viol_4_count == 0 and viol_5_count == 0:
            return colors, n
            
        best_violations = viol_4_count + viol_5_count
        best_colors = colors[:]
        
        # Local Search
        # We want to reduce violations.
        # Strategy: Pick a random violation, pick a random edge in it, flip.
        # If it reduces total violations, accept. Else, maybe accept with some prob (SA) or reject.
        # Given time, simple hill climbing with random restarts might be better.
        
        # To make it faster, we can maintain a list of violated 4-sets and 5-sets.
        # But updating that list dynamically is complex.
        # Instead, we just pick a random edge that is part of *some* violation?
        # Or just pick a random edge overall?
        # Picking a random edge is O(1).
        # But we need to check if it helps.
        
        # Better: Keep a set of currently violated 4-sets and 5-sets.
        # When a flip happens, update the status of affected 4-sets and 5-sets.
        
        violated_4 = set(i for i, c in enumerate(red_count_4) if c == 4)
        violated_5 = set(i for i, c in enumerate(blue_count_5) if c == 10)
        
        step = 0
        last_improvement = 0
        
        while time.time() - start_time < time_limit:
            if not violated_4 and not violated_5:
                return colors, n
                
            # Pick a random violation
            if violated_4 and (not violated_5 or random.random() < 0.5):
                # Pick random violated 4-set
                s_idx = random.choice(list(violated_4))
                # Pick random edge in it
                e_idx = random.choice(four_sets_edges[s_idx])
            else:
                # Pick random violated 5-set
                s_idx = random.choice(list(violated_5))
                # Pick random edge in it
                e_idx = random.choice(five_sets_edges[s_idx])
                
            # Flip e_idx
            old_c = colors[e_idx]
            new_c = 1 - old_c
            colors[e_idx] = new_c
            
            # Update counts and violation sets
            # 4-sets
            for s in edge_to_four_sets[e_idx]:
                if old_c == 0: # Was Red, now Blue
                    red_count_4[s] -= 1
                    if red_count_4[s] == 3:
                        violated_4.discard(s)
                    # If it was 4, it became 3, so removed from violated_4.
                    # If it was <4, it became <3, no change in violation status (unless it was 4? No, 4->3 is only change to non-violated)
                else: # Was Blue, now Red
                    red_count_4[s] += 1
                    if red_count_4[s] == 4:
                        violated_4.add(s)
                        
            # 5-sets
            for s in edge_to_five_sets[e_idx]:
                if old_c == 1: # Was Blue, now Red
                    blue_count_5[s] -= 1
                    if blue_count_5[s] == 9:
                        violated_5.discard(s)
                else: # Was Red, now Blue
                    blue_count_5[s] += 1
                    if blue_count_5[s] == 10:
                        violated_5.add(s)
                        
            current_violations = len(violated_4) + len(violated_5)
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                last_improvement = step
                
            step += 1
            
            # Restart if stuck for a while
            if step - last_improvement > 10000:
                # Random restart
                colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples]
                # Re-init counts
                for i in range(len(four_sets)):
                    red_count_4[i] = sum(1 for e in four_sets_edges[i] if colors[e] == 0)
                for i in range(len(five_sets)):
                    blue_count_5[i] = sum(1 for e in five_sets_edges[i] if colors[e] == 1)
                violated_4 = set(i for i, c in enumerate(red_count_4) if c == 4)
                violated_5 = set(i for i, c in enumerate(blue_count_5) if c == 10)
                current_violations = len(violated_4) + len(violated_5)
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                last_improvement = step
                step = 0
                
        # If we didn't find a valid one, return the best one found?
        # The task requires a VALID coloring to score > 0.
        # "An invalid result... scores 0".
        # So we must return a valid one.
        # If we didn't find one for this n, we should return None.
        if best_violations == 0:
            return best_colors, n
        else:
            return None, n

    # Try n from 34 down to 9
    for n in range(34, 8, -1):
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
        res = try_find_coloring(n)
        if res[0] is not None:
            return {'n': res[1], 'colors': res[0]}
            
    # Fallback: Return a known small valid coloring
    # n=9 modular coloring is valid?
    # Seed program says n=9 is valid.
    n = 9
    triples = list(itertools.combinations(range(n), 3))
    colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples]
    # Verify validity of n=9
    # Red K4?
    valid = True
    for s in itertools.combinations(range(n), 4):
        reds = 0
        for t in itertools.combinations(s, 3):
            idx = triples.index(t) # Slow, but n=9 is small
            if colors[idx] == 0:
                reds += 1
        if reds == 4:
            valid = False
            break
    if valid:
        for s in itertools.combinations(range(n), 5):
            blues = 0
            for t in itertools.combinations(s, 3):
                idx = triples.index(t)
                if colors[idx] == 1:
                    blues += 1
            if blues == 10:
                valid = False
                break
    if valid:
        return {'n': n, 'colors': colors}
        
    # If even n=9 fails (it shouldn't), return n=1
    return {'n': 1, 'colors': []}
