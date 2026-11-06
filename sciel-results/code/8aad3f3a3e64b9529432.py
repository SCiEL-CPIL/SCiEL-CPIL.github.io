import time
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Precompute weights as Fractions for exact comparison
    # w[a] = 1 / (512 - a)
    w = [None] * 512
    for a in range(1, 512):
        w[a] = Fraction(1, 512 - a)
    
    # Candidates sorted by weight descending
    candidates = list(range(511, 0, -1))
    
    # Greedy Initialization
    S = []
    S_set = set()
    current_score = Fraction(0, 1)
    
    for a in candidates:
        # Check if a is coprime to all in S
        valid = True
        for s in S:
            if gcd(a, s) != 1:
                valid = False
                break
        if valid:
            S.append(a)
            S_set.add(a)
            current_score += w[a]
            
    # Helper to get conflicts of x with set S_list
    def get_conflicts(x, S_list):
        conflicts = []
        for s in S_list:
            if gcd(x, s) != 1:
                conflicts.append(s)
        return conflicts

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = Fraction(0, 1)
        best_move = None
        
        # 1. Check Add Moves
        # Since we iterate candidates in descending weight order, the first valid one is the best Add
        for b in candidates:
            if b in S_set:
                continue
            conflicts = get_conflicts(b, S)
            if len(conflicts) == 0:
                gain = w[b]
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('add', b)
                break # Only the first (highest weight) valid add is the best add
        
        # 2. Check 1-Swap Moves (Remove a, Add b)
        for a in S:
            # We want to find best b not in S, coprime to S\{a}
            # Iterate b in descending weight
            for b in candidates:
                if b in S_set:
                    continue
                if b == a:
                    continue
                # Check if b is coprime to S excluding a
                valid = True
                for s in S:
                    if s == a:
                        continue
                    if gcd(b, s) != 1:
                        valid = False
                        break
                if valid:
                    gain = w[b] - w[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('swap1', a, b)
                    break # First valid b for this a is the best b for this a
        
        # 3. Check 2-out, 1-in Moves (Remove a, c; Add b)
        # Iterate b in candidates
        for b in candidates:
            if b in S_set:
                continue
            conflicts = get_conflicts(b, S)
            if len(conflicts) == 2:
                a, c = conflicts
                # Ensure a and c are distinct (they are from set S, so yes)
                # Check if removing a, c allows adding b
                # This is already true by definition of conflicts
                gain = w[b] - w[a] - w[c]
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('swap21', a, c, b)
            elif len(conflicts) > 2:
                # Cannot be fixed by removing only 2 elements
                pass
            # If len < 2, handled by Add or 1-Swap
        
        # 4. Check 1-out, 2-in Moves (Remove a; Add b, d)
        # Restricted to top 50 candidates for b, d to keep complexity low
        top_candidates = candidates[:50]
        for a in S:
            S_rem = [s for s in S if s != a]
            # Find best pair b, d from top_candidates
            # b, d must be coprime to S_rem and each other
            best_pair_gain = Fraction(0, 1)
            best_pair = None
            for i in range(len(top_candidates)):
                b = top_candidates[i]
                if b in S_set:
                    continue
                # Check if b coprime to S_rem
                valid_b = True
                for s in S_rem:
                    if gcd(b, s) != 1:
                        valid_b = False
                        break
                if not valid_b:
                    continue
                
                for j in range(i + 1, len(top_candidates)):
                    d = top_candidates[j]
                    if d in S_set:
                        continue
                    # Check if d coprime to S_rem
                    valid_d = True
                    for s in S_rem:
                        if gcd(d, s) != 1:
                            valid_d = False
                            break
                    if not valid_d:
                        continue
                    # Check if b, d coprime to each other
                    if gcd(b, d) != 1:
                        continue
                    
                    gain_pair = w[b] + w[d]
                    if gain_pair > best_pair_gain:
                        best_pair_gain = gain_pair
                        best_pair = (b, d)
            
            if best_pair is not None:
                total_gain = best_pair_gain - w[a]
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_move = ('swap12', a, best_pair[0], best_pair[1])
        
        # Apply best move if found
        if best_move is not None:
            improved = True
            if best_move[0] == 'add':
                b = best_move[1]
                S.append(b)
                S_set.add(b)
                current_score += w[b]
            elif best_move[0] == 'swap1':
                a, b = best_move[1], best_move[2]
                S.remove(a)
                S_set.remove(a)
                S.append(b)
                S_set.add(b)
                current_score += w[b] - w[a]
            elif best_move[0] == 'swap21':
                a, c, b = best_move[1], best_move[2], best_move[3]
                S.remove(a)
                S.remove(c)
                S_set.remove(a)
                S_set.remove(c)
                S.append(b)
                S_set.add(b)
                current_score += w[b] - w[a] - w[c]
            elif best_move[0] == 'swap12':
                a, b, d = best_move[1], best_move[2], best_move[3]
                S.remove(a)
                S_set.remove(a)
                S.append(b)
                S.append(d)
                S_set.add(b)
                S_set.add(d)
                current_score += w[b] + w[d] - w[a]
        
    return S
